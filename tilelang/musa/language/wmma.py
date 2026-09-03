from __future__ import annotations

from tvm import tirx

from tilelang.language.gemm_op import GemmWarpPolicy, _gemm_impl
from tilelang.language.utils import BufferLikeType


def wmma_gemm(
    A: BufferLikeType,
    B: BufferLikeType,
    C: BufferLikeType,
    transpose_A: bool = False,
    transpose_B: bool = False,
    policy: GemmWarpPolicy = GemmWarpPolicy.Square,
    clear_accum: bool = False,
) -> tirx.Call:
    """Perform explicit MP31 WMMA GEMM on A/B fragments.

    The operation intentionally exposes only the register-register path and
    uses 32-thread warps; larger warp-local tiles are tiled by the runtime
    helper. It never changes the target selection behavior of ``T.gemm``.
    """
    return _gemm_impl(
        "tl.tileop.wmma_gemm",
        A,
        B,
        C,
        transpose_A,
        transpose_B,
        policy,
        clear_accum,
        wg_wait=0,
    )


__all__ = ["wmma_gemm"]
