from __future__ import annotations

from tilelang.tileop.gemm.registry import register_gemm_impl
from .gemm_fma import GEMM_INST_FMA, GemmFMA
from .gemm_mma import GEMM_INST_MUSA_MMA, GemmMusaMMA
from .gemm_qy2_mma import GemmQY2MMA
from .gemm_sqmma import GEMM_INST_SQMMA, GemmSQMMA
from .gemm_wmma import GEMM_INST_WMMA, GemmWMMA
from tilelang.musa.target import target_is_musa, target_is_qy2


def _match_musa_mma(target) -> bool:
    return target_is_musa(target) and not target_is_qy2(target)


def _match_qy2_mma(target) -> bool:
    return target_is_qy2(target)


register_gemm_impl("musa.mma", GEMM_INST_MUSA_MMA, _match_musa_mma, GemmMusaMMA)
register_gemm_impl("musa.qy2_mma", GEMM_INST_MUSA_MMA, _match_qy2_mma, GemmQY2MMA)
register_gemm_impl("musa.fma", GEMM_INST_FMA, target_is_musa, GemmFMA)
register_gemm_impl("musa.sqmma", GEMM_INST_SQMMA, target_is_musa, GemmSQMMA)
register_gemm_impl("musa.ph1wmma", GEMM_INST_WMMA, target_is_musa, GemmWMMA)
