"""Public facade for the active backend's distributed-memory support."""

from tilelang.musa import distributed as _impl
from tilelang.musa.distributed import *  # noqa: F401,F403

__all__ = _impl.__all__
