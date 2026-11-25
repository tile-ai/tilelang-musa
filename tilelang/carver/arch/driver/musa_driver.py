from __future__ import annotations

import ctypes

try:
    from torch_musa._MUSAC import _MusaDeviceProperties
except ImportError:
    _MusaDeviceProperties = type("DummyMusaDeviceProperties", (), {})


class musaDeviceAttrNames:
    r"""
    refer to MUSA runtime attribute ids.
    Keep ids aligned with CUDA-compatible values used by tests.
    """

    musaDevAttrMaxThreadsPerBlock: int = 1
    musaDevAttrMaxSharedMemoryPerBlock: int = 8
    musaDevAttrMaxRegistersPerBlock: int = 12
    musaDevAttrMultiProcessorCount: int = 16
    musaDevAttrMaxSharedMemoryPerMultiprocessor: int = 81
    musaDevAttrMaxPersistingL2CacheSize: int = 108


def get_musa_device_properties(device_id: int = 0) -> _MusaDeviceProperties | None:
    try:
        import torch

        musa = getattr(torch, "musa", None)
        if musa is None or not musa.is_available():
            return None
        return musa.get_device_properties(device_id)
    except (ImportError, RuntimeError):
        return None


def get_device_name(device_id: int = 0) -> str | None:
    prop = get_musa_device_properties(device_id)
    if prop is not None:
        return getattr(prop, "name", None)
    return None


def get_device_attribute(attr: int, device_id: int = 0) -> int | None:
    try:
        libmusart = ctypes.cdll.LoadLibrary("libmusart.so")
        value = ctypes.c_int()
        musa_device_get_attribute = libmusart.musaDeviceGetAttribute
        musa_device_get_attribute.argtypes = [
            ctypes.POINTER(ctypes.c_int),
            ctypes.c_int,
            ctypes.c_int,
        ]
        musa_device_get_attribute.restype = ctypes.c_int
        result = musa_device_get_attribute(ctypes.byref(value), attr, device_id)
        if result != 0:
            return None
        return value.value
    except (AttributeError, OSError):
        return None


def get_shared_memory_per_block(device_id: int = 0, format: str = "bytes") -> int | None:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    shared_mem = get_device_attribute(musaDeviceAttrNames.musaDevAttrMaxSharedMemoryPerBlock, device_id)
    if shared_mem is None:
        return None
    if format == "bytes":
        return shared_mem
    if format == "kb":
        return shared_mem // 1024
    if format == "mb":
        return shared_mem // (1024 * 1024)
    raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def get_max_dynamic_shared_size_bytes(device_id: int = 0, format: str = "bytes") -> int | None:
    assert format in ["bytes", "kb", "mb"], "Invalid format. Must be one of: bytes, kb, mb"
    shared_mem = get_device_attribute(musaDeviceAttrNames.musaDevAttrMaxSharedMemoryPerMultiprocessor, device_id)
    if shared_mem is None:
        return None
    if format == "bytes":
        return shared_mem
    if format == "kb":
        return shared_mem // 1024
    if format == "mb":
        return shared_mem // (1024 * 1024)
    raise RuntimeError("Invalid format. Must be one of: bytes, kb, mb")


def get_persisting_l2_cache_max_size(device_id: int = 0) -> int | None:
    return get_device_attribute(musaDeviceAttrNames.musaDevAttrMaxPersistingL2CacheSize, device_id)


def get_num_sms(device_id: int = 0) -> int | None:
    return get_device_attribute(musaDeviceAttrNames.musaDevAttrMultiProcessorCount, device_id)


def get_registers_per_block(device_id: int = 0) -> int | None:
    return get_device_attribute(musaDeviceAttrNames.musaDevAttrMaxRegistersPerBlock, device_id)
