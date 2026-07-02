from __future__ import annotations

from tilelang.cuda.op.gemm_sp.gemm_sp_mma import GemmSPMMA
from tilelang.tileop.gemm_sp.registry import register_gemm_sp_impl
from tilelang.utils.target import target_is_musa, target_is_qy2


GEMM_SP_INST_MUSA_MMA_SP = "musa.mma.sp"


class GemmSPMUSAMMA(GemmSPMMA):
    GEMM_SP_INST_MMA_SP = GEMM_SP_INST_MUSA_MMA_SP


def _match_musa_mma_sp(target) -> bool:
    return target_is_musa(target) and target_is_qy2(target)


register_gemm_sp_impl(
    "musa.mma.sp",
    GEMM_SP_INST_MUSA_MMA_SP,
    _match_musa_mma_sp,
    GemmSPMUSAMMA,
)
