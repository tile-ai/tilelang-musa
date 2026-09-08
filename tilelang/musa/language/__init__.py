"""MUSA language dialect: common TileLang plus MUSA extensions."""

from __future__ import annotations

from tilelang.language.common import *  # noqa: F401,F403
from tilelang.language.common import __all__ as _COMMON_ALL
from tilelang.language.builtin import (  # noqa: F401
    get_warp_group_idx,
    ldg128,
    ldg256,
    ldg32,
    ldg64,
    shuffle_elect,
    stg128,
    stg256,
    stg32,
    stg64,
)

from .print import *  # noqa: F401,F403
from .print import __all__ as _PRINT_ALL
from .fast_divmod import *  # noqa: F401,F403
from .fast_divmod import __all__ as _FAST_DIVMOD_ALL
from .sqmma import *  # noqa: F401,F403
from .sqmma import __all__ as _SQMMA_ALL
from .wmma import *  # noqa: F401,F403
from .wmma import __all__ as _WMMA_ALL
from .copy_ext import *  # noqa: F401,F403
from .copy_ext import __all__ as _COPY_ALL
from .distributed import *  # noqa: F401,F403
from .distributed import __all__ as _DISTRIBUTED_ALL
from .memory import *  # noqa: F401,F403
from .memory import __all__ as _MEMORY_ALL

_MUSA_API_ALL = (
    "get_warp_group_idx",
    "ldg128",
    "ldg256",
    "ldg32",
    "ldg64",
    "shuffle_elect",
    "stg128",
    "stg256",
    "stg32",
    "stg64",
)

__tilelang_dialect__ = "musa"
__all__ = tuple(
    dict.fromkeys(
        (
            *_COMMON_ALL,
            *_MUSA_API_ALL,
            *_PRINT_ALL,
            *_FAST_DIVMOD_ALL,
            *_SQMMA_ALL,
            *_WMMA_ALL,
            *_COPY_ALL,
            *_DISTRIBUTED_ALL,
            *_MEMORY_ALL,
        )
    )
)

del _COMMON_ALL, _MUSA_API_ALL, _PRINT_ALL, _FAST_DIVMOD_ALL, _SQMMA_ALL, _WMMA_ALL, _COPY_ALL, _DISTRIBUTED_ALL, _MEMORY_ALL
