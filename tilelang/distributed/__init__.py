"""Public facade for the active backend's distributed-memory support."""

import sys

from tilelang.musa import distributed as _impl
from tilelang.musa.distributed import *  # noqa: F401,F403

sys.modules[f"{__name__}.shared_memory"] = _impl.shared_memory

__all__ = _impl.__all__
