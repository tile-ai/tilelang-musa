"""MUSA runtime device hooks."""

from __future__ import annotations

import torch

from tilelang import tvm
from tilelang.backend.runtime_device import RuntimeDevice, register_runtime_device


def _current_device() -> torch.device:
    torch.musa._lazy_init()
    current_device = getattr(torch._C, "_musa_getDevice", None)
    device_id = current_device() if current_device is not None else torch.musa.current_device()
    return torch.device("musa", device_id)


def _device_guard(device: torch.device):
    return torch.musa.device(device)


def _tvm_device(device_id: int):
    return tvm.musa(device_id)


register_runtime_device(
    RuntimeDevice(
        target_kind="musa",
        current_device=_current_device,
        device_guard=_device_guard,
        tvm_device=_tvm_device,
    ),
    override=True,
)
