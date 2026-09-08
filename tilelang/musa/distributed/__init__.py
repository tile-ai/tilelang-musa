"""MUSA distributed-memory building blocks."""

from .dlpack import MUSA_DLPACK_DEVICE_TYPE, get_musa_dlpack_device

__all__ = ["MUSA_DLPACK_DEVICE_TYPE", "get_musa_dlpack_device"]
