from __future__ import annotations

from tilelang.backend.gemm import register_gemm_impl
from tilelang.tileop.gemm.gemm_fma import GEMM_INST_FMA, GemmFMA
from tilelang.tileop.gemm.gemm_mma import GemmMMA
from tilelang.tileop.gemm.gemm_sqmma import GEMM_INST_SQMMA, GemmSQMMA
from tilelang.tileop.gemm.gemm_wmma import GEMM_INST_WMMA, GemmWMMA
from tilelang.utils.target import target_is_musa


GEMM_INST_MUSA_MMA = "musa.mma"


def _match_musa(target) -> bool:
    return target_is_musa(target)


register_gemm_impl("musa.mma", GEMM_INST_MUSA_MMA, _match_musa, GemmMMA)
register_gemm_impl("musa.fma", GEMM_INST_FMA, _match_musa, GemmFMA)
register_gemm_impl("musa.sqmma", GEMM_INST_SQMMA, _match_musa, GemmSQMMA)
register_gemm_impl("musa.ph1wmma", GEMM_INST_WMMA, _match_musa, GemmWMMA)
