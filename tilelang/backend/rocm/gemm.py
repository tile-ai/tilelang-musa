from __future__ import annotations

from tilelang.backend.gemm import register_gemm_impl
from tilelang.tileop.gemm.gemm_mfma import GEMM_INST_MFMA, GemmMFMA
from tilelang.utils.target import target_is_hip


def _match_mfma(target) -> bool:
    return target_is_hip(target)


register_gemm_impl("rocm.mfma", GEMM_INST_MFMA, _match_mfma, GemmMFMA)
