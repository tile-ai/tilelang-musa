from __future__ import annotations

from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.layout import Fragment, make_gemm_fragment_c_linear, make_linear_layout
from tilelang.utils.language import is_shared, is_fragment
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


GEMM_INST_FMA = "musa.fma"


class GemmFMA(GemmBase):
    def _check_operand_scopes(self):
        if self.is_gemm_ss():
            return
        if self.is_gemm_sr() or self.is_gemm_rs():
            rs = self.is_gemm_rs()
            fragment = self.A if rs else self.B
            offsets = self.A_base_offsets if rs else self.B_base_offsets
            axis_dim = int(self.trans_A) if rs else int(not self.trans_B)
            axis_extent = self.M if rs else self.N
            if (
                len(fragment.shape) == 2
                and int(fragment.shape[axis_dim]) == int(axis_extent)
                and all(isinstance(offset, (int, tirx.IntImm)) and int(offset) == 0 for offset in offsets)
            ):
                return
        raise ValueError(f"Unsupported gemm combination for fma, A: {self.A.scope()}, B: {self.B.scope()}")

    @staticmethod
    def _mixed_scope_lanes(axis_extent, thread_nums):
        # Fragment layout inference needs a rectangular bijection. Use the
        # most threads that divide the operand's spatial extent evenly.
        extent = int(axis_extent)
        return next(lanes for lanes in range(min(extent, int(thread_nums)), 0, -1) if extent % lanes == 0)

    def infer_layout(self, target: Target, thread_nums: int):
        self._check_operand_scopes()
        # Keep the warp partition query in sync with C++ GemmWarpPolicy.
        self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_INST_FMA)

        a_shape = self.A.shape
        b_shape = self.B.shape
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError("GemmFMA expects A/B to be at least 2D buffers")

        a_stride = int(a_shape[-2])
        a_continuous = int(a_shape[-1])
        b_stride = int(b_shape[-2])
        b_continuous = int(b_shape[-1])

        if self.is_gemm_sr() or self.is_gemm_rs():
            # A fragment can only be read by its owning thread. Give it the
            # same row (RS) or column (SR) ownership as the output.
            rs = self.is_gemm_rs()
            axis_extent = self.M if rs else self.N
            fragment = self.A if rs else self.B
            row_is_axis = not self.trans_A if rs else self.trans_B
            lanes = self._mixed_scope_lanes(fragment.shape[0 if row_is_axis else 1], thread_nums)
            slices = T.ceildiv(axis_extent, lanes)
            c_layout = Fragment(
                [self.M, self.N],
                forward_thread_fn=lambda i, j: (i if rs else j) % lanes,
                forward_index_fn=lambda i, j: (j if rs else i) * slices + (i if rs else j) // lanes,
            )
            k_extent = fragment.shape[1 if row_is_axis else 0]
            fragment_layout = Fragment(
                list(fragment.shape),
                forward_thread_fn=lambda x, y: (x if row_is_axis else y) % lanes,
                forward_index_fn=lambda x, y: (
                    (x if row_is_axis else y) // lanes * k_extent + (y if row_is_axis else x)
                ),
            )
            shared = self.B if rs else self.A
            return {
                self.C: c_layout,
                fragment: fragment_layout,
                shared: make_linear_layout([b_stride, b_continuous] if rs else [a_stride, a_continuous]),
            }

        return {
            self.C: make_gemm_fragment_c_linear(self.M, self.N, thread_nums),
            self.A: make_linear_layout([a_stride, a_continuous]),
            self.B: make_linear_layout([b_stride, b_continuous]),
        }

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        self._check_operand_scopes()
        del mbar_phase_expr
        del layout_map  # FMA lowering does not need layout remap input.
        thread_nums = thread_bounds.extent
        accum_dtype = self.accum_dtype
        clear_accum = self.clear_accum

        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        A_buf = A_region.buffer if A_region is not None else self.A
        B_buf = B_region.buffer if B_region is not None else self.B
        C_buf = C_region.buffer if C_region is not None else self.C

        A_base_offsets = self.A_base_offsets
        B_base_offsets = self.B_base_offsets
        C_base_offsets = self.C_base_offsets

        M = self.M
        N = self.N
        K = self.K
        trans_A = self.trans_A
        trans_B = self.trans_B
        sr = self.is_gemm_sr()
        rs = self.is_gemm_rs()
        if sr:
            axis_extent, outer_extent = N, M
        elif rs:
            axis_extent, outer_extent = M, N
        else:
            axis_extent, outer_extent = M * N, 1
        if sr:
            lanes = self._mixed_scope_lanes(B_buf.shape[0 if trans_B else 1], thread_nums)
        elif rs:
            lanes = self._mixed_scope_lanes(A_buf.shape[1 if trans_A else 0], thread_nums)
        else:
            lanes = thread_nums
        slices = T.ceildiv(axis_extent, lanes)

        def region_indices(buf, base_offsets, logical_indices):
            rank = len(buf.shape)
            logical_rank = len(logical_indices)
            if rank < logical_rank:
                raise ValueError(
                    f"GemmFMA expects buffer rank >= logical rank, got buffer {buf.name} rank {rank} and logical rank {logical_rank}"
                )
            full_offsets = [0] * (rank - len(base_offsets)) + list(base_offsets)
            leading_rank = rank - logical_rank
            indices = list(full_offsets[:leading_rank])
            for dim, logical_index in enumerate(logical_indices):
                indices.append(full_offsets[leading_rank + dim] + logical_index)
            return tuple(indices)

        @T.prim_func
        def _gemm_fma() -> None:
            accum = T.alloc_local((1,), accum_dtype)
            local_thread = thread_var - thread_bounds.min
            for outer, inner in T.grid(outer_extent, slices):
                axis = inner * lanes + local_thread
                if local_thread < lanes and axis < axis_extent:
                    if sr:
                        i, j = outer, axis
                    elif rs:
                        i, j = axis, outer
                    else:
                        i, j = axis // N, axis % N

                    if clear_accum:
                        accum[0] = T.cast(0, accum_dtype)
                    else:
                        accum[0] = C_buf[region_indices(C_buf, C_base_offsets, (i, j))]

                    for k_iter in T.serial(0, K):
                        if trans_A:
                            a_val = T.cast(A_buf[region_indices(A_buf, A_base_offsets, (k_iter, i))], accum_dtype)
                        else:
                            a_val = T.cast(A_buf[region_indices(A_buf, A_base_offsets, (i, k_iter))], accum_dtype)
                        if trans_B:
                            b_val = T.cast(B_buf[region_indices(B_buf, B_base_offsets, (j, k_iter))], accum_dtype)
                        else:
                            b_val = T.cast(B_buf[region_indices(B_buf, B_base_offsets, (k_iter, j))], accum_dtype)
                        accum[0] = accum[0] + a_val * b_val
                    C_buf[region_indices(C_buf, C_base_offsets, (i, j))] = accum[0]

        return _Simplify(_gemm_fma, inline_let=True)

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
