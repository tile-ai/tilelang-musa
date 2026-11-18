from __future__ import annotations

from tilelang.tileop.gemm_sp.gemm_sp_base import GemmSPBase
from tilelang.layout import make_gemm_fragment_c_linear, make_linear_layout
from tilelang import tvm as tvm
from tilelang.utils.language import is_shared, is_fragment
from tvm.target import Target
from tvm.ir import Range
from tvm import tirx
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


GEMM_SP_INST_MUSA_FMA_SP = "musa.fma.sp"


class GemmSPFMA(GemmSPBase):
    @staticmethod
    def _as_const_bool(value, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, tirx.IntImm):
            return bool(value.value)
        if hasattr(value, "value"):
            return bool(value.value)
        raise ValueError(f"{name} must be a constant Bool type, got {value}")

    def infer_layout(self, target: Target, thread_nums: int):
        self.policy.compute_warp_partition(self.M, self.N, thread_nums, target, GEMM_SP_INST_MUSA_FMA_SP)

        if not self.is_gemm_ss():
            raise ValueError(f"Unsupported sparse gemm combination for fma, A: {self.A.scope()}, B: {self.B.scope()}")
        if not is_shared(self.E):
            raise ValueError(f"Sparse FMA expects E in shared scope, got {self.E.scope()}")

        a_shape = self.A.shape
        b_shape = self.B.shape
        if len(a_shape) < 2 or len(b_shape) < 2:
            raise ValueError("GemmSPFMA expects A/B to be at least 2D buffers")

        a_stride = int(a_shape[-2])
        a_continuous = int(a_shape[-1])
        b_stride = int(b_shape[-2])
        b_continuous = int(b_shape[-1])

        return {
            self.C: make_gemm_fragment_c_linear(self.M, self.N, thread_nums),
            self.A: make_linear_layout([a_stride, a_continuous]),
            self.B: make_linear_layout([b_stride, b_continuous]),
        }

    def lower(self, layout_map: dict, target: Target, thread_bounds: Range, thread_var: tirx.Var):
        del layout_map
        block_size = int(thread_bounds.extent)
        warp_m, warp_n = self.policy.compute_warp_partition(
            self.M, self.N, block_size, target, GEMM_SP_INST_MUSA_FMA_SP
        )

        if not self.is_gemm_ss():
            raise ValueError(f"Unsupported sparse gemm combination for fma, A: {self.A.scope()}, B: {self.B.scope()}")
        if not is_shared(self.E):
            raise ValueError(f"Sparse FMA expects E in shared scope, got {self.E.scope()}")
        if self.wg_wait != 0:
            raise ValueError(f"Sparse FMA expects wg_wait == 0, got {self.wg_wait}")

        a_region = self.ARegion
        e_region = self.ERegion
        b_region = self.BRegion
        c_region = self.CRegion
        trans_A = self.trans_A
        trans_B = self.trans_B
        trans_E = self.trans_E
        clear_accum = self._as_const_bool(self.clear_accum, "clear_accum")

        lda = self.M if trans_A else self.K
        ldb = self.K if trans_B else self.N
        op_instance = (
            f"tl::gemm_sp_ss<{self.M}, {self.N}, {self.K}, "
            f"{warp_m}, {warp_n}, {int(bool(trans_A))}, {int(bool(trans_E))}, "
            f"{int(bool(trans_B))}, {int(clear_accum)}, {lda}, {ldb}, 0, 0>"
        )

        @T.prim_func
        def _gemm_sp_ss() -> None:
            a_ptr = T.access_ptr(a_region, "r")
            b_ptr = T.access_ptr(b_region, "r")
            c_ptr = T.access_ptr(c_region, "rw")
            e_ptr = T.access_ptr(e_region, "r")
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tvm.ir.Op.get("tl.tl_gemm_sp"),
                    tirx.StringImm(op_instance),
                    a_ptr,
                    b_ptr,
                    c_ptr,
                    e_ptr,
                )
            )

        return _Simplify(_gemm_sp_ss, inline_let=True)

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_sr(self) -> bool:
        return is_shared(self.A) and is_fragment(self.B)

    def is_gemm_rs(self) -> bool:
        return is_fragment(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
