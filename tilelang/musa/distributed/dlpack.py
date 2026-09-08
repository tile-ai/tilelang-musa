"""Compatibility helpers for MUSA tensors on older PyTorch DLPack builds."""

from __future__ import annotations

from typing import Any


MUSA_DLPACK_DEVICE_TYPE = 12


def get_musa_dlpack_device(tensor: Any) -> tuple[int, int]:
    """Return a DLPack device pair with a fail-closed MUSA fallback."""

    try:
        device_type, device_id = tensor.__dlpack_device__()
    except (AttributeError, ValueError):
        device = getattr(tensor, "device", None)
        if getattr(device, "type", None) != "musa":
            raise
        device_id = getattr(device, "index", None)
        if device_id is None:
            import torch

            device_id = torch.musa.current_device()
        return MUSA_DLPACK_DEVICE_TYPE, int(device_id)
    return int(device_type), int(device_id)


__all__ = ["MUSA_DLPACK_DEVICE_TYPE", "get_musa_dlpack_device"]
