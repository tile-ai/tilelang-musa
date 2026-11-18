"""Stub-safe checks for the optional MUSA IPC/VMM runtime.

These tests only assert the call-time contract that must hold whether or not a
MUSA driver/GPU is present: the runtime entry points are always registered,
capability probes never crash, and allocation either succeeds (when the driver
reports support) or fails with a Python exception instead of aborting the
process.
"""

from pathlib import Path
import ctypes

import pytest
import tvm_ffi


@pytest.fixture(scope="module", autouse=True)
def _load_tilelang_runtime():
    library = Path(__file__).parents[3] / "build" / "lib" / "libtilelang.so"
    if not library.exists():
        pytest.skip("TileLang native library is not built")
    ctypes.CDLL(str(library), mode=ctypes.RTLD_GLOBAL)


def _global_func(name):
    func = tvm_ffi.get_global_func(name, allow_missing=True)
    assert func is not None, f"{name} is not registered"
    return func


def test_musa_shared_memory_capabilities_are_optional():
    supports_vmm = _global_func("tl.shared_memory.supports_vmm_fabric")
    supports_multicast = _global_func("tl.shared_memory.supports_multicast")

    # Capability probes must be call-time safe: they return a bool on every
    # machine, including ones without a MUSA driver.
    assert isinstance(supports_vmm(), bool)
    # Multicast is not implemented on MUSA yet.
    assert supports_multicast() is False


def test_musa_shared_memory_teardown_rejects_unknown_handles_cleanly():
    close_vmm = _global_func("tl.shared_memory.close_vmm_handle")
    close_ipc = _global_func("tl.shared_memory.close_ipc_handle")
    with pytest.raises((ValueError, RuntimeError)):
        close_vmm(1)
    with pytest.raises((ValueError, RuntimeError)):
        close_ipc(1)


def test_musa_shared_memory_allocation_is_call_time_safe():
    supports_vmm = _global_func("tl.shared_memory.supports_vmm_fabric")
    vmm_malloc = _global_func("tl.shared_memory.vmm_malloc")
    vmm_free = _global_func("tl.shared_memory.vmm_free")

    if supports_vmm():
        # Driver present: allocation must succeed and be releasable.
        ptr = vmm_malloc(4096)
        assert ptr != 0
        vmm_free(ptr)
    else:
        # No driver: allocation must fail with a Python exception instead of
        # crashing the process.
        with pytest.raises((RuntimeError, ValueError)):
            vmm_malloc(4096)
