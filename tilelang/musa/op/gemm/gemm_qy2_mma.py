from __future__ import annotations

from tilelang import _ffi_api
from tilelang import language as T
from tilelang import tvm
from tilelang.layout import (
    make_gemm_ab_layout,
    make_gemm_qy2_fragment_a_col,
    make_gemm_qy2_fragment_a_row,
    make_gemm_qy2_fragment_b_col,
    make_gemm_qy2_fragment_b_row,
    make_gemm_qy2_fragment_c,
    make_gemm_qy2_wmma_fragment_a,
    make_gemm_qy2_wmma_fragment_b,
    make_gemm_qy2_wmma_fragment_c,
)
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_full_region, is_shared
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target
import tvm_ffi

from .gemm_mma import GemmMusaMMA
from .qy2_shared_layout import (
    QY2SharedLayoutMatch,
    detect_qy2_shared_layout,
    validate_qy2_shared_layouts,
)


_DEFAULT_QY2_SHARED_LAYOUT = QY2SharedLayoutMatch()


class GemmQY2MMA(GemmMusaMMA):
    @staticmethod
    def _get_pointer_ignore_last_ndim(buffer: tirx.Buffer) -> int:
        return 2 if is_shared(buffer) else 0

    @staticmethod
    def _as_const_bool(value, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, tirx.IntImm):
            return bool(value.value)
        if hasattr(value, "value"):
            return bool(value.value)
        raise ValueError(f"{name} must be a constant Bool type, got {value}")

    @staticmethod
    def _as_const_int(value, name: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, tirx.IntImm):
            return int(value.value)
        if hasattr(value, "value"):
            return int(value.value)
        raise ValueError(f"{name} must be a constant integer, got {value}")

    def _select_inst_shape(self, target: Target, block_size: int) -> tuple[int, int, int]:
        inst_shape = _ffi_api.GemmSelectQY2MmaInstShape(self.gemm_node, block_size, target)
        if not inst_shape:
            raise ValueError("No QY2 MMA instruction shape was selected")
        return tuple(int(value) for value in inst_shape)

    def _compute_warp_partition(
        self,
        target: Target,
        block_size: int,
        inst_shape: tuple[int, int, int],
    ) -> tuple[int, int]:
        _ffi_api.MUSAGemmComputeWarpPartition(
            self.policy,
            self.M,
            self.N,
            block_size,
            target,
            list(inst_shape),
        )
        return self.policy.m_warp, self.policy.n_warp

    @staticmethod
    def _fit_fragment_layout_to_region(layout, region: tirx.BufferRegion):
        layout_shape = list(layout.get_input_shape())
        buffer = region.buffer
        buffer_shape = list(buffer.shape)
        # Keep ordinary 2-D operands on the original inference path. Rebuild
        # the fragment layout only when a slice or leading dimension requires it.
        if is_full_region(region) and len(buffer_shape) == len(layout_shape):
            return layout

        if len(buffer_shape) < len(layout_shape):
            raise ValueError(f"QY2 fragment buffer rank must be at least the MMA layout rank, got {buffer_shape} and {layout_shape}")

        leading_ndim = len(buffer_shape) - len(layout_shape)
        matrix_shape = buffer_shape[leading_ndim:]
        repeats = []
        for buffer_extent_expr, layout_extent_expr in zip(matrix_shape, layout_shape):
            buffer_extent = int(buffer_extent_expr)
            layout_extent = int(layout_extent_expr)
            if buffer_extent % layout_extent != 0:
                raise ValueError(
                    "QY2 fragment buffer extents must be multiples of the MMA "
                    f"layout extents in matrix dimensions, got {buffer_shape} and {layout_shape}"
                )
            repeats.append(buffer_extent // layout_extent)

        if any(repeat != 1 for repeat in repeats):
            layout = layout.repeat(
                repeats,
                repeat_on_thread=False,
                lower_dim_first=True,
            )
        if leading_ndim > 0:
            matrix_layout_shape = list(layout.get_input_shape())
            layout = layout.reshape([1] * leading_ndim + matrix_layout_shape)
            layout = layout.repeat(
                buffer_shape[:leading_ndim] + [1] * len(matrix_layout_shape),
                repeat_on_thread=False,
                lower_dim_first=True,
            )
        return layout

    def infer_layout(self, target: Target, thread_nums: int):
        block_size = int(thread_nums)
        inst_shape = self._select_inst_shape(target, block_size)
        m_warp, n_warp = self._compute_warp_partition(target, block_size, inst_shape)
        warp_m = self.M // m_warp
        warp_n = self.N // n_warp
        use_extended_mma = inst_shape[:2] != (32, 32)
        c_bits = int(tvm.DataType(self.C.dtype).bits)
        c_layout = (
            make_gemm_qy2_wmma_fragment_c(self.M, self.N, warp_m, warp_n, c_bits, inst_shape)
            if use_extended_mma
            else make_gemm_qy2_fragment_c(self.M, self.N, warp_m, warp_n, c_bits)
        )
        c_layout = self._fit_fragment_layout_to_region(c_layout, self.CRegion)
        layouts = {self.C: c_layout}

        if self.is_gemm_ss():
            layouts[self.A] = self._make_shared_layout(self.A, not self.trans_A)
            layouts[self.B] = self._make_shared_layout(self.B, self.trans_B)
        elif self.is_gemm_sr():
            layouts[self.A] = self._make_shared_layout(self.A, not self.trans_A)
            layouts[self.B] = self._make_fragment_b_layout(warp_m, warp_n, inst_shape)
        elif self.is_gemm_rs():
            layouts[self.A] = self._make_fragment_a_layout(warp_m, warp_n, inst_shape)
            layouts[self.B] = self._make_shared_layout(self.B, self.trans_B)
        elif self.is_gemm_rr():
            layouts[self.A] = self._make_fragment_a_layout(warp_m, warp_n, inst_shape)
            layouts[self.B] = self._make_fragment_b_layout(warp_m, warp_n, inst_shape)
        else:
            raise ValueError(f"Unsupported QY2 GEMM combination, A: {self.A.scope()}, B: {self.B.scope()}")
        return layouts

    @staticmethod
    def _make_shared_layout(buffer, k_inner: bool):
        shape = list(buffer.shape)
        return make_gemm_ab_layout(
            shape,
            int(shape[-1]),
            int(tvm.DataType(buffer.dtype).bits),
            k_inner,
        )

    def _make_fragment_a_layout(
        self,
        warp_m: int,
        warp_n: int,
        inst_shape: tuple[int, int, int],
    ):
        element_bits = int(tvm.DataType(self.A.dtype).bits)
        if inst_shape[:2] != (32, 32):
            layout = make_gemm_qy2_wmma_fragment_a(
                self.M,
                self.N,
                self.K,
                warp_m,
                warp_n,
                element_bits,
                self.trans_A,
                inst_shape,
            )
        else:
            make_layout = make_gemm_qy2_fragment_a_col if self.trans_A else make_gemm_qy2_fragment_a_row
            layout = make_layout(self.M, self.N, self.K, warp_m, warp_n, element_bits)
        return self._fit_fragment_layout_to_region(layout, self.ARegion)

    def _make_fragment_b_layout(
        self,
        warp_m: int,
        warp_n: int,
        inst_shape: tuple[int, int, int],
    ):
        element_bits = int(tvm.DataType(self.B.dtype).bits)
        if inst_shape[:2] != (32, 32):
            layout = make_gemm_qy2_wmma_fragment_b(
                self.M,
                self.N,
                self.K,
                warp_m,
                warp_n,
                element_bits,
                self.trans_B,
                inst_shape,
            )
        else:
            make_layout = make_gemm_qy2_fragment_b_row if self.trans_B else make_gemm_qy2_fragment_b_col
            layout = make_layout(self.M, self.N, self.K, warp_m, warp_n, element_bits)
        return self._fit_fragment_layout_to_region(layout, self.BRegion)

    def _build_op_instance(
        self,
        warp_m: int,
        warp_n: int,
        inst_shape: tuple[int, int, int],
        subgroup_size: int = 0,
        a_shared_layout: QY2SharedLayoutMatch = _DEFAULT_QY2_SHARED_LAYOUT,
        b_shared_layout: QY2SharedLayoutMatch = _DEFAULT_QY2_SHARED_LAYOUT,
    ) -> str:
        if self.is_gemm_ss():
            op_name = "tl::gemm_ss"
        elif self.is_gemm_sr():
            op_name = "tl::gemm_sr"
        elif self.is_gemm_rs():
            op_name = "tl::gemm_rs"
        elif self.is_gemm_rr():
            op_name = "tl::gemm_rr"
        else:
            raise ValueError(f"Unsupported QY2 GEMM combination, A: {self.A.scope()}, B: {self.B.scope()}")

        clear_accum = self._as_const_bool(self.clear_accum, "clear_accum")
        stride_a = self._as_const_int(self.stride_A, "stride_A")
        stride_b = self._as_const_int(self.stride_B, "stride_B")
        offset_a = self._as_const_int(self.offset_A, "offset_A")
        offset_b = self._as_const_int(self.offset_B, "offset_B")
        wg_wait = self._as_const_int(self.wg_wait, "wg_wait")
        if wg_wait != 0:
            raise ValueError(f"wg_wait must be 0 for QY2 MMA, got {wg_wait}")

        op_instance = (
            f"{op_name}<"
            f"{self.M}, {self.N}, {self.K}, "
            f"{warp_m}, {warp_n}, "
            f"{int(bool(self.trans_A))}, {int(bool(self.trans_B))}, "
            f"{int(clear_accum)}, {stride_a}, {stride_b}, "
            f"{offset_a}, {offset_b}"
        )
        use_optimized_shared_layout = a_shared_layout.optimized or b_shared_layout.optimized
        if inst_shape[:2] != (32, 32) or subgroup_size > 0 or use_optimized_shared_layout:
            op_instance += f", {inst_shape[0]}, {inst_shape[1]}, {inst_shape[2]}"
        if subgroup_size > 0 or use_optimized_shared_layout:
            op_instance += f", {subgroup_size}, 0"
        if use_optimized_shared_layout:
            op_instance += (
                f", {int(a_shared_layout.inverse)}"
                f", {int(b_shared_layout.inverse)}"
                f", {int(a_shared_layout.packed)}"
                f", {int(b_shared_layout.packed)}"
                f", {a_shared_layout.physical_offset}"
                f", {b_shared_layout.physical_offset}"
                f", {a_shared_layout.atom_x_stride}"
                f", {a_shared_layout.atom_k_stride}"
                f", {b_shared_layout.atom_x_stride}"
                f", {b_shared_layout.atom_k_stride}"
            )
        return f"{op_instance}>"

    def _detect_shared_layout(
        self,
        layout_map,
        operand: str,
        inst_shape: tuple[int, int, int],
    ) -> QY2SharedLayoutMatch:
        is_a = operand == "A"
        region = self.ARegion if is_a else self.BRegion
        transposed = bool(self.trans_A if is_a else self.trans_B)
        return detect_qy2_shared_layout(
            layout_map,
            region,
            operand,
            transposed,
            inst_shape,
        )

    def _detect_shared_layouts(
        self,
        layout_map,
        inst_shape: tuple[int, int, int],
    ) -> tuple[QY2SharedLayoutMatch, QY2SharedLayoutMatch]:
        a_shared_layout = self._detect_shared_layout(layout_map, "A", inst_shape)
        b_shared_layout = self._detect_shared_layout(layout_map, "B", inst_shape)
        validate_qy2_shared_layouts(
            self,
            inst_shape,
            a_shared_layout,
            b_shared_layout,
        )
        return a_shared_layout, b_shared_layout

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tirx.Var,
        mbar_phase_expr: tirx.PrimExpr | None = None,
        thread_y_extent: tirx.PrimExpr | None = None,
    ):
        del thread_var, mbar_phase_expr
        block_size = int(thread_bounds.extent)
        subgroup_size = 0
        if thread_y_extent is not None and self._as_const_int(thread_y_extent, "thread_y_extent") > 1:
            subgroup_size = block_size
        inst_shape = self._select_inst_shape(target, block_size)
        m_warp, n_warp = self._compute_warp_partition(target, block_size, inst_shape)
        a_shared_layout, b_shared_layout = self._detect_shared_layouts(
            layout_map,
            inst_shape,
        )
        op_instance = self._build_op_instance(
            m_warp,
            n_warp,
            inst_shape,
            subgroup_size,
            a_shared_layout,
            b_shared_layout,
        )
        a_region = self.ARegion
        b_region = self.BRegion
        c_region = self.CRegion
        a_ignore_last_ndim = self._get_pointer_ignore_last_ndim(self.A)
        b_ignore_last_ndim = self._get_pointer_ignore_last_ndim(self.B)

        @T.prim_func
        def _gemm_qy2_mma() -> None:
            # The template applies the logical offsets of the last two matrix
            # dimensions for shared operands. Fragment operands are consumed
            # directly and therefore need their complete region offsets.
            a_ptr = T.access_ptr(
                a_region,
                "r",
                ignore_last_ndim=a_ignore_last_ndim,
            )
            b_ptr = T.access_ptr(
                b_region,
                "r",
                ignore_last_ndim=b_ignore_last_ndim,
            )
            c_ptr = T.access_ptr(c_region, "rw")
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tvm.ir.Op.get("tl.tl_gemm"),
                    tirx.StringImm(op_instance),
                    a_ptr,
                    b_ptr,
                    c_ptr,
                )
            )

        return _Simplify(_gemm_qy2_mma, inline_let=True)


@tvm_ffi.register_global_func("tl.musa.qy2.gemm.reuse_shared_layout")
def gemm_qy2_reuse_shared_layout(
    gemm,
    layout_map,
    buffer,
    target: Target,
    block_size: int,
):
    impl = GemmQY2MMA(gemm)
    inst_shape = impl._select_inst_shape(target, int(block_size))
    if buffer.same_as(impl.A):
        match = impl._detect_shared_layout(layout_map, "A", inst_shape)
    elif buffer.same_as(impl.B):
        match = impl._detect_shared_layout(layout_map, "B", inst_shape)
    else:
        return False
    return match.optimized


@tvm_ffi.register_global_func("tl.musa.qy2.gemm.lower_subgroup")
def gemm_qy2_lower_subgroup(
    gemm,
    layout_map,
    target: Target,
    thread_bounds: Range,
    thread_var: tirx.Var,
    mbar_phase_expr: tirx.PrimExpr,
    thread_y_extent: tirx.PrimExpr,
):
    thread_nums = thread_bounds.extent
    gemm_inst = gemm._select_gemm_instruction(thread_nums, target)
    impl_class = gemm._get_implementation_class(gemm_inst, target)
    if impl_class is not GemmQY2MMA:
        raise ValueError(f"QY2 threadIdx.y subgroup GEMM requires the QY2 MMA implementation, but selected {impl_class.__name__}")
    impl = impl_class(gemm)
    return impl.lower(
        layout_map,
        target,
        thread_bounds,
        thread_var,
        mbar_phase_expr,
        thread_y_extent,
    )
