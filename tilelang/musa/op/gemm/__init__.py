from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from .gemm_fma import GEMM_INST_FMA, GemmFMA
from .gemm_mma import GEMM_INST_MUSA_MMA, GemmMusaMMA
from .gemm_sqmma import GEMM_INST_SQMMA, GemmSQMMA
from .gemm_wmma import GEMM_INST_WMMA, GemmWMMA
from tilelang.utils.target import target_is_musa


def _match_musa(target) -> bool:
    return target_is_musa(target)


register_gemm_impl("musa.mma", GEMM_INST_MUSA_MMA, _match_musa, GemmMusaMMA)
register_gemm_impl("musa.fma", GEMM_INST_FMA, _match_musa, GemmFMA)
register_gemm_impl("musa.sqmma", GEMM_INST_SQMMA, _match_musa, GemmSQMMA)
register_gemm_impl("musa.ph1wmma", GEMM_INST_WMMA, _match_musa, GemmWMMA)
