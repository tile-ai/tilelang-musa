"""MUSA-specific transformation frontends."""

from tilelang.transform import _ffi_api


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
    "LowerPHIntrin",
    "LowerL2Persistent",
    "PersistThreadblock",
]
