"""Explicit MP31 WMMA register-register GEMM lowering."""

from __future__ import annotations

from tilelang import _ffi_api
from tilelang import language as T
from tilelang import tvm as tvm
from tilelang.musa.layout import make_mp31_wmma_fragment_a, make_mp31_wmma_fragment_b, make_mp31_wmma_fragment_c
from tilelang.tileop.gemm.gemm_base import GemmBase
from tilelang.transform.simplify import _Simplify
from tilelang.utils.language import is_fragment
from tvm import tirx
from tvm.ir import Range
from tvm.target import Target


GEMM_INST_WMMA = "musa.wmma"


def _as_int(value, name: str) -> int:
    if isinstance(value, int):
        return value
    if hasattr(value, "value"):
        return int(value.value)
    raise ValueError(f"{name} must be a constant integer, got {value}")


_DTYPE_TO_WMMA_TAG = {
    "float16": "f16",
    "bfloat16": "bf16",
    "custom[tfloat32]": "tf32",
    "tfloat32": "tf32",
    "int8": "s8",
    "uint8": "u8",
    "int4": "s4",
    "float8_e4m3": "e4m3",
    "float8_e4m3fn": "e4m3",
    "float8_e5m2": "e5m2",
    "float32": "f32",
    "int32": "s32",
    "uint32": "u32",
}


def _wmma_tag(dtype) -> str:
    name = str(dtype)
    if name in _DTYPE_TO_WMMA_TAG:
        return _DTYPE_TO_WMMA_TAG[name]
    raise ValueError(f"Unsupported MP31 WMMA dtype: {name}")


class GemmMP31WMMA(GemmBase):
    def __post_init__(self) -> None:
        if not self.is_gemm_rr():
            raise ValueError(f"Unsupported WMMA combination, A: {self.A.scope()}, B: {self.B.scope()}")

    def _warp_partition(self, target: Target, block_size: int) -> tuple[int, int]:
        m_warp, n_warp = self.policy.compute_warp_partition(self.M, self.N, block_size, target, GEMM_INST_WMMA)
        return int(m_warp), int(n_warp)

    def _get_inst_shape(self, target: Target, block_size: int, m_warp: int, n_warp: int) -> tuple[int, int, int]:
        inst_m, inst_n, inst_k = _ffi_api.get_mp31_wmma_inst_shape(self.gemm_node, int(block_size), int(m_warp), int(n_warp), target)
        return int(inst_m), int(inst_n), int(inst_k)

    def infer_layout(self, target: Target, thread_nums: int):
        if not self.is_gemm_rr() or not is_fragment(self.C):
            raise ValueError(f"Unsupported WMMA combination, A: {self.A.scope()}, B: {self.B.scope()}")
        thread_nums = _as_int(thread_nums, "thread_nums")
        m_warp, n_warp = self._warp_partition(target, thread_nums)
        inst_shape = self._get_inst_shape(target, thread_nums, m_warp, n_warp)
        a_bits = int(tvm.DataType(self.a_dtype).bits)
        b_bits = int(tvm.DataType(self.b_dtype).bits)
        return {
            self.A: make_mp31_wmma_fragment_a(self.A, m_warp, n_warp, a_bits, bool(self.trans_A), inst_shape),
            self.B: make_mp31_wmma_fragment_b(self.B, m_warp, n_warp, b_bits, bool(self.trans_B), inst_shape),
            self.C: make_mp31_wmma_fragment_c(self.C, m_warp, n_warp, inst_shape),
        }

    def lower(
        self,
        layout_map: dict,
        target: Target,
        thread_bounds: Range,
        thread_index: tirx.PrimExpr,
        mbar_phase_expr: tirx.PrimExpr | None = None,
    ):
        del layout_map, thread_index, mbar_phase_expr
        if not self.is_gemm_rr() or not is_fragment(self.C):
            raise ValueError(f"Unsupported WMMA combination, A: {self.A.scope()}, B: {self.B.scope()}")
        block_size = _as_int(thread_bounds.extent, "thread_bounds.extent")
        m_warp, n_warp = self._warp_partition(target, block_size)
        inst_m, inst_n, inst_k = self._get_inst_shape(target, block_size, m_warp, n_warp)
        a_tag = _wmma_tag(self.a_dtype)
        b_tag = _wmma_tag(self.b_dtype)
        c_tag = _wmma_tag(self.accum_dtype)
        func = (
            f"tl::wmma_rr<{self.M}, {self.N}, {self.K}, "
            f"{m_warp}, {n_warp}, {inst_m}, {inst_n}, {inst_k}, "
            f"tl::wmma::{a_tag}, tl::wmma::{b_tag}, tl::wmma::{c_tag}, "
            f"{int(bool(self.trans_A))}, {int(bool(self.trans_B))}>"
        )
        A_region = self.ARegion
        B_region = self.BRegion
        C_region = self.CRegion

        @T.prim_func
        def _gemm_rr() -> None:
            T.call_extern(
                "handle",
                func,
                T.access_ptr(A_region, "r", ignore_last_ndim=2),
                T.access_ptr(B_region, "r", ignore_last_ndim=2),
                T.access_ptr(C_region, "rw", ignore_last_ndim=2),
                self.clear_accum,
            )

        return _Simplify(_gemm_rr, inline_let=True)
