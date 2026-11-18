from types import SimpleNamespace

import pytest

from tilelang.distributed.musa_dlpack import (
    MUSA_DLPACK_DEVICE_TYPE,
    get_musa_dlpack_device,
)


class _Tensor:
    def __init__(self, device_type: str, device_index: int, result=None):
        self.device = SimpleNamespace(type=device_type, index=device_index)
        self._result = result

    def __dlpack_device__(self):
        if self._result is None:
            raise ValueError(f"Unknown device type {self.device.type}")
        return self._result


def test_musa_dlpack_device_uses_native_result_when_available():
    assert get_musa_dlpack_device(_Tensor("musa", 3, (12, 3))) == (12, 3)


def test_musa_dlpack_device_falls_back_only_for_musa():
    assert get_musa_dlpack_device(_Tensor("musa", 5)) == (
        MUSA_DLPACK_DEVICE_TYPE,
        5,
    )
    with pytest.raises(ValueError, match="Unknown device type cpu"):
        get_musa_dlpack_device(_Tensor("cpu", 0))
