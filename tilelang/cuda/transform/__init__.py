"""CUDA-specific transformation frontends."""

from tilelang.transform import _ffi_api


def LowerL2Persistent():
    """LowerL2Persistent"""
    return _ffi_api.LowerL2Persistent()  # type: ignore


def PersistThreadblock():
    """PersistThreadblock"""
    return _ffi_api.PersistThreadblock()  # type: ignore


__all__ = [
    "LowerL2Persistent",
    "PersistThreadblock",
]
