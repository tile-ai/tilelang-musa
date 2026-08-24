"""MUSA-specific transformation frontends."""

from .. import _ffi_api


def LowerLDGSTG():
    """Lower global memory load/store to MUSA ldg/stg intrinsics."""
    return _ffi_api.LowerLDGSTG()  # type: ignore


def LowerFastDivmod():
    """Lower explicit MUSA fast integer division operations."""
    return _ffi_api.LowerFastDivmod()  # type: ignore


__all__ = ["LowerLDGSTG", "LowerFastDivmod"]
