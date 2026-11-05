from __future__ import annotations

import torch

from tilelang.dtypes import get_tvm_dtype

_FLOAT8_E4M3FN = getattr(torch, "float8_e4m3fn", None)


def get_tilelang_type(elem_type: torch.dtype) -> str:
    """Convert torch dtype to the TileLang dtype string used by tests."""
    if elem_type is _FLOAT8_E4M3FN:
        return "float8_e4m3"
    return str(get_tvm_dtype(elem_type))
