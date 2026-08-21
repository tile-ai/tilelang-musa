"""Backend-owned runtime device integration for generic JIT adapters."""

from __future__ import annotations

from collections.abc import Callable
from contextlib import AbstractContextManager
from dataclasses import dataclass
from typing import Any

from tvm.target import Target

CurrentDeviceFunc = Callable[[], Any]
DeviceGuardFunc = Callable[[Any], AbstractContextManager[Any]]
TVMDeviceFunc = Callable[[int], Any]


@dataclass(frozen=True, slots=True)
class RuntimeDevice:
    """Runtime hooks owned by one device backend."""

    target_kind: str
    current_device: CurrentDeviceFunc
    device_guard: DeviceGuardFunc
    tvm_device: TVMDeviceFunc

    def normalize_device(self, device: Any | None) -> Any:
        if device is None or getattr(device, "index", None) is None:
            return self.current_device()
        return device


_RUNTIME_DEVICES: dict[str, RuntimeDevice] = {}


def register_runtime_device(runtime_device: RuntimeDevice, *, override: bool = False) -> RuntimeDevice:
    target_kind = runtime_device.target_kind
    if target_kind in _RUNTIME_DEVICES and not override:
        raise ValueError(f"Runtime device for target kind {target_kind!r} is already registered")
    _RUNTIME_DEVICES[target_kind] = runtime_device
    return runtime_device


def resolve_runtime_device(target: str | Target, *, allow_missing: bool = False) -> RuntimeDevice | None:
    target_kind = target if isinstance(target, str) else target.kind.name
    runtime_device = _RUNTIME_DEVICES.get(target_kind)
    if runtime_device is None and not allow_missing:
        raise ValueError(f"No runtime device registered for target kind {target_kind!r}")
    return runtime_device
