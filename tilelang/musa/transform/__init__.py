"""MUSA-specific transformation frontends."""

from .. import _ffi_api


def LowerLDGSTG():
    """Lower global memory load/store to MUSA ldg/stg intrinsics."""
    return _ffi_api.LowerLDGSTG()  # type: ignore


def LowerFastDivmod():
    """Lower explicit MUSA fast integer division operations."""
    return _ffi_api.LowerFastDivmod()  # type: ignore


def LowerTMEIntrin():
    """Lower MP31 TME descriptors and barrier handles."""
    return _ffi_api.LowerTMEIntrin()  # type: ignore


def ProducerConsumerWarpSpecialized():
    """Split eligible MP31 TME pipelines into producer and consumer warps."""
    return _ffi_api.ProducerConsumerWarpSpecialized()  # type: ignore


def LowerPartialThreadSync():
    """Lower partial shared-memory sync to MP31 TME barriers."""
    return _ffi_api.LowerPartialThreadSync()  # type: ignore


__all__ = [
    "LowerLDGSTG",
    "LowerFastDivmod",
    "LowerTMEIntrin",
    "LowerPartialThreadSync",
    "ProducerConsumerWarpSpecialized",
]
