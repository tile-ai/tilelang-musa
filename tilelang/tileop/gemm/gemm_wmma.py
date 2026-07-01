from __future__ import annotations

from .gemm_base import GemmBase
from tilelang.layout import (
    make_ph1_wmma_ab_layout,
    make_ph1_wmma_fragment_a,
    make_ph1_wmma_fragment_b,
    make_ph1_wmma_fragment_c,
)
from tilelang.utils.language import is_fragment, is_shared
from tilelang import _ffi_api
from tilelang import tvm as tvm
from tvm.target import Target
from tvm.ir import Range
from tvm import tir
from tilelang import language as T
from tilelang.transform.simplify import _Simplify


GEMM_INST_WMMA = "musa.ph1wmma"


class GemmWMMA(GemmBase):
    @staticmethod
    def _as_const_bool(value, name: str) -> bool:
        if isinstance(value, bool):
            return value
        if isinstance(value, tir.IntImm):
            return bool(value.value)
        if hasattr(value, "value"):
            return bool(value.value)
        raise ValueError(f"{name} must be a constant Bool type, got {value}")

    @staticmethod
    def _as_const_int(value, name: str) -> int:
        if isinstance(value, int):
            return value
        if isinstance(value, tir.IntImm):
            return int(value.value)
        if hasattr(value, "value"):
            return int(value.value)
        raise ValueError(f"{name} must be a constant integer, got {value}")

    def _select_wmma_inst_shape(self, block_size: int, target: Target) -> tuple[int, int, int]:
        inst_shape = _ffi_api.GemmPySelectPH1WmmaInstShape(self.gemm_node, int(block_size), target)
        if len(inst_shape) != 3:
            raise ValueError("PH1 WMMA is selected but no valid instruction shape is found")
        return int(inst_shape[0]), int(inst_shape[1]), int(inst_shape[2])

    def _validate_operand_scopes(self) -> None:
        if self.is_gemm_ss() or self.is_gemm_rr():
            return
        raise ValueError(
            f"PH1 WMMA requires A/B to both be in shared memory or both be fragments, got A={self.A.scope()}, B={self.B.scope()}"
        )

    def infer_layout(self, target: Target, thread_nums: int):
        self._validate_operand_scopes()
        if not is_fragment(self.C):
            raise ValueError(f"PH1 WMMA requires C in local.fragment scope, got {self.C.scope()}")

        warp_m, warp_n = self.policy.compute_warp_partition(
            self.M,
            self.N,
            thread_nums,
            target,
            GEMM_INST_WMMA,
        )
        inst_shape = self._select_wmma_inst_shape(int(thread_nums), target)
        c_bits = int(tvm.DataType(self.C.dtype).bits)
        layouts = {
            self.C: make_ph1_wmma_fragment_c(
                self.M,
                self.N,
                warp_m,
                warp_n,
                c_bits,
                inst_shape,
            )
        }

        if self.is_gemm_ss():
            a_shape = list(self.A.shape)
            b_shape = list(self.B.shape)
            a_bits = int(tvm.DataType(self.A.dtype).bits)
            b_bits = int(tvm.DataType(self.B.dtype).bits)
            layouts[self.A] = make_ph1_wmma_ab_layout(
                a_shape,
                int(a_shape[-1]),
                a_bits,
                not self.trans_A,
            )
            layouts[self.B] = make_ph1_wmma_ab_layout(
                b_shape,
                int(b_shape[-1]),
                b_bits,
                self.trans_B,
            )
        else:
            a_bits = int(tvm.DataType(self.A.dtype).bits)
            b_bits = int(tvm.DataType(self.B.dtype).bits)
            layouts[self.A] = make_ph1_wmma_fragment_a(
                self.M,
                self.N,
                self.K,
                warp_m,
                warp_n,
                a_bits,
                self.trans_A,
                inst_shape,
            )
            layouts[self.B] = make_ph1_wmma_fragment_b(
                self.M,
                self.N,
                self.K,
                warp_m,
                warp_n,
                b_bits,
                self.trans_B,
                inst_shape,
            )
        return layouts

    def _build_op_instance(
        self,
        warp_m: int,
        warp_n: int,
        inst_shape: tuple[int, int, int],
    ) -> tuple[str, bool]:
        self._validate_operand_scopes()
        is_shared_shared = self.is_gemm_ss()
        op_name = "tl::gemm_ss" if is_shared_shared else "tl::gemm_rr"

        clear_accum = self._as_const_bool(self.clear_accum, "clear_accum")
        stride_a = self._as_const_int(self.stride_A, "stride_A")
        stride_b = self._as_const_int(self.stride_B, "stride_B")
        offset_a = self._as_const_int(self.offset_A, "offset_A")
        offset_b = self._as_const_int(self.offset_B, "offset_B")
        wg_wait = self._as_const_int(self.wg_wait, "wg_wait")
        if not is_shared_shared and wg_wait != 0:
            raise ValueError("PH1 WMMA gemm_rr requires wg_wait == 0")

        op_instance = (
            f"{op_name}<"
            f"{self.M}, {self.N}, {self.K}, "
            f"{warp_m}, {warp_n}, "
            f"{int(bool(self.trans_A))}, {int(bool(self.trans_B))}, {int(clear_accum)}, "
            f"{stride_a}, {stride_b}, "
            f"{offset_a}, {offset_b}, "
            f"false, {inst_shape[0]}, {inst_shape[1]}, {inst_shape[2]}"
        )
        if wg_wait != 0:
            op_instance += f", {wg_wait}"
        return op_instance + ">", is_shared_shared

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_var: tir.Var,
        mbar_phase_expr: tir.PrimExpr | None = None,
    ):
        del layout_map, thread_var, mbar_phase_expr
        if not is_fragment(self.C):
            raise ValueError(f"PH1 WMMA requires C in local.fragment scope, got {self.C.scope()}")

        block_size = int(thread_bounds.extent)
        warp_m, warp_n = self.policy.compute_warp_partition(
            self.M,
            self.N,
            block_size,
            target,
            GEMM_INST_WMMA,
        )
        inst_shape = self._select_wmma_inst_shape(block_size, target)
        op_instance, is_shared_shared = self._build_op_instance(warp_m, warp_n, inst_shape)
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        if is_shared_shared:

            @T.prim_func
            def _gemm_wmma_ss() -> None:
                A_ptr = T.access_ptr(A_region, "r")
                B_ptr = T.access_ptr(B_region, "r")
                C_ptr = T.access_ptr(C_region, "rw")
                T.evaluate(
                    T.call_intrin(
                        "handle",
                        tir.op.Op.get("tl.tl_gemm"),
                        tir.StringImm(op_instance),
                        A_ptr,
                        B_ptr,
                        C_ptr,
                        thread_bounds.min,
                    )
                )

            return _Simplify(_gemm_wmma_ss, inline_let=True)

        @T.prim_func
        def _gemm_wmma_rr() -> None:
            A_ptr = T.access_ptr(A_region, "r")
            B_ptr = T.access_ptr(B_region, "r")
            C_ptr = T.access_ptr(C_region, "rw")
            T.evaluate(
                T.call_intrin(
                    "handle",
                    tir.op.Op.get("tl.tl_gemm"),
                    tir.StringImm(op_instance),
                    A_ptr,
                    B_ptr,
                    C_ptr,
                )
            )

        return _Simplify(_gemm_wmma_rr, inline_let=True)

    def is_gemm_ss(self) -> bool:
        return is_shared(self.A) and is_shared(self.B)

    def is_gemm_rr(self) -> bool:
        return is_fragment(self.A) and is_fragment(self.B)
