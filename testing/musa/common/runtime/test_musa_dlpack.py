from types import SimpleNamespace

import pytest
import tilelang.testing

from tilelang.musa.distributed import (
    MUSA_DLPACK_DEVICE_TYPE,
    get_musa_dlpack_device,
)
from tilelang.distributed import get_musa_dlpack_device as public_dlpack_device


class _Tensor:
    def __init__(self, device_type: str, device_index: int | None, result=None):
        self.device = SimpleNamespace(type=device_type, index=device_index)
        self._result = result

    def __dlpack_device__(self):
        if self._result is None:
            raise ValueError(f"Unknown device type {self.device.type}")
        return self._result


@tilelang.testing.requires_musa
def test_musa_dlpack_device_uses_native_result():
    assert get_musa_dlpack_device(_Tensor("musa", 3, (12, 3))) == (12, 3)


@tilelang.testing.requires_musa
def test_distributed_facade_reexports_dlpack_helper():
    assert public_dlpack_device is get_musa_dlpack_device


@tilelang.testing.requires_musa
def test_musa_dlpack_device_falls_back_only_for_musa():
    assert get_musa_dlpack_device(_Tensor("musa", 5)) == (
        MUSA_DLPACK_DEVICE_TYPE,
        5,
    )
    with pytest.raises(ValueError, match="Unknown device type cpu"):
        get_musa_dlpack_device(_Tensor("cpu", 0))


@tilelang.testing.requires_musa
def test_musa_dlpack_device_uses_current_device_when_index_is_missing(monkeypatch):
    import torch

    monkeypatch.setattr(torch.musa, "current_device", lambda: 7)
    assert get_musa_dlpack_device(_Tensor("musa", None)) == (
        MUSA_DLPACK_DEVICE_TYPE,
        7,
    )
