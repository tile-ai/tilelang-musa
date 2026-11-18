import threading
from contextlib import nullcontext

import torch

from tilelang.jit.adapter.tvm_ffi import TVMFFIKernelAdapter


def _make_adapter(monkeypatch):
    adapter = TVMFFIKernelAdapter.__new__(TVMFFIKernelAdapter)
    adapter.executable = None
    adapter._executables_by_device = {}
    adapter._last_executable_by_device = None
    adapter._executable_lock = threading.Lock()
    created = []

    def make_executable():
        executable = object()
        created.append(executable)
        return executable

    monkeypatch.setattr(adapter, "_make_executable", make_executable)
    return adapter, created


def test_tvm_ffi_executable_cache_tracks_input_device(monkeypatch):
    adapter, created = _make_adapter(monkeypatch)
    musa0 = torch.device("musa", 0)
    musa1 = torch.device("musa", 1)
    device_guards = []

    def device_guard(device):
        device_guards.append(device)
        return nullcontext()

    monkeypatch.setattr(torch.musa, "device", device_guard)

    first = adapter._get_executable(musa0)
    assert adapter._get_executable(musa0) is first
    second = adapter._get_executable(musa1)
    assert second is not first
    assert adapter._get_executable(musa0) is first

    assert created == [first, second]
    assert device_guards == [musa0, musa1]
    assert set(adapter._executables_by_device) == {musa0, musa1}


def test_tvm_ffi_executable_cache_avoids_global_device_query(monkeypatch):
    adapter, _ = _make_adapter(monkeypatch)

    def fail_current_device():
        raise AssertionError("an indexed input device should avoid a global device query")

    monkeypatch.setattr(torch.musa, "current_device", fail_current_device)
    adapter._get_executable(torch.device("musa", 0))


def test_tvm_ffi_restored_executable_bypasses_per_device_cache(monkeypatch):
    adapter, created = _make_adapter(monkeypatch)
    restored = object()
    adapter.executable = restored

    assert adapter._get_executable(torch.device("musa", 0)) is restored
    assert created == []
