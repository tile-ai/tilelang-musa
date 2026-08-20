"""MUSA language dialect: common TileLang plus MUSA extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.builtin import get_warp_group_idx, shuffle_elect  # noqa: F401

from .print import *  # noqa: F401,F403
from .print import __all__ as _PRINT_ALL

_MUSA_API_ALL = (
    "get_warp_group_idx",
    "shuffle_elect",
)

__tilelang_dialect__ = "musa"
__all__ = tuple(dict.fromkeys((*_COMMON_ALL, *_MUSA_API_ALL, *_PRINT_ALL)))

del _COMMON_ALL, _MUSA_API_ALL, _PRINT_ALL
