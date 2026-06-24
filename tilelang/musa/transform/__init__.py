"""MUSA-specific transformation frontends."""

from tilelang.transform import _ffi_api


def ProducerConsumerWarpSpecialized():
    """Apply the MUSA-aware warp-specialized producer/consumer rewrite."""
    return _ffi_api.ProducerConsumerWarpSpecialized()  # type: ignore


def FuseMBarrierArriveExpectTx():
    """Fuse adjacent MUSA mbarrier arrive/expect-tx operations."""
    return _ffi_api.FuseMBarrierArriveExpectTx()  # type: ignore


def InjectFenceProxy():
    """Inject backend-safe async proxy fences for MUSA lowering."""
    return _ffi_api.InjectFenceProxy()  # type: ignore


def LowerLDGSTG():
    """Lower MUSA global/shared vector copies to LDGSTG forms."""
    return _ffi_api.LowerLDGSTG()  # type: ignore


def LowerAsyncCopy():
    """Lower MUSA global-to-shared copies into async-copy intrinsics."""
    return _ffi_api.LowerMUSAAsyncCopy()  # type: ignore


def LowerSharedBarrier():
    """Lower shared barriers using MUSA barrier semantics."""
    return _ffi_api.LowerSharedBarrier()  # type: ignore


def LowerSharedTmem():
    """Lower shared tensor-memory allocations when supported."""
    return _ffi_api.LowerSharedTmem()  # type: ignore


def LowerPHIntrin():
    """LowerPHIntrin"""
    if hasattr(_ffi_api, "LowerPHIntrin"):
        return _ffi_api.LowerPHIntrin()  # type: ignore
    return lambda f: f


def LowerL2Persistent():
    """LowerL2Persistent"""
    return _ffi_api.LowerL2Persistent()  # type: ignore


def PersistThreadblock():
    """PersistThreadblock"""
    return _ffi_api.PersistThreadblock()  # type: ignore


__all__ = [
    "ProducerConsumerWarpSpecialized",
    "FuseMBarrierArriveExpectTx",
    "InjectFenceProxy",
    "LowerLDGSTG",
    "LowerAsyncCopy",
    "LowerSharedBarrier",
    "LowerSharedTmem",
    "LowerPHIntrin",
    "LowerL2Persistent",
    "PersistThreadblock",
]
