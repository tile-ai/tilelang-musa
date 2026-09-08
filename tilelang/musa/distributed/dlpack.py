"""Compatibility helpers for MUSA tensors on older PyTorch DLPack builds."""

from __future__ import annotations

import ctypes
from math import prod
from typing import Any

import torch


MUSA_DLPACK_DEVICE_TYPE = 12


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [
        ("code", ctypes.c_uint8),
        ("bits", ctypes.c_uint8),
        ("lanes", ctypes.c_uint16),
    ]


class _DLTensor(ctypes.Structure):
    _fields_ = [
        ("data", ctypes.c_void_p),
        ("device", _DLDevice),
        ("ndim", ctypes.c_int32),
        ("dtype", _DLDataType),
        ("shape", ctypes.POINTER(ctypes.c_int64)),
        ("strides", ctypes.POINTER(ctypes.c_int64)),
        ("byte_offset", ctypes.c_uint64),
    ]


class _DLManagedTensor(ctypes.Structure):
    pass


_DLManagedTensorDeleter = ctypes.CFUNCTYPE(None, ctypes.POINTER(_DLManagedTensor))
_DLManagedTensor._fields_ = [
    ("dl_tensor", _DLTensor),
    ("manager_ctx", ctypes.c_void_p),
    ("deleter", _DLManagedTensorDeleter),
]

_DLPACK_DTYPE = {
    torch.float32: (2, 32),
    torch.float16: (2, 16),
    torch.bfloat16: (4, 16),
    torch.float64: (2, 64),
    torch.int8: (0, 8),
    torch.int16: (0, 16),
    torch.int32: (0, 32),
    torch.int64: (0, 64),
    torch.uint8: (1, 8),
    torch.uint16: (1, 16),
    torch.uint32: (1, 32),
    torch.uint64: (1, 64),
    torch.bool: (6, 8),
}
_DLPACK_VIEWS: dict[int, tuple[object, ...]] = {}


@_DLManagedTensorDeleter
def _release_dlpack_view(managed: ctypes.POINTER(_DLManagedTensor)) -> None:
    _DLPACK_VIEWS.pop(ctypes.addressof(managed.contents), None)


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


def tensor_from_musa_ptr(
    ptr: int,
    shape: tuple[int, ...],
    dtype: torch.dtype,
    device: int,
    owner: Any = None,
) -> torch.Tensor:
    """Create a contiguous non-owning MUSA tensor using a DLPack capsule."""

    shape = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape):
        raise ValueError(f"MUSA tensor dimensions must be non-negative, got {shape}")
    if prod(shape) == 0:
        return torch.empty(shape, dtype=dtype, device=f"musa:{device}")
    if ptr <= 0:
        raise ValueError(f"MUSA tensor pointer must be positive, got {ptr}")
    dtype_fields = _DLPACK_DTYPE.get(dtype)
    if dtype_fields is None:
        raise ValueError(f"unsupported MUSA DLPack dtype: {dtype}")

    device_type = MUSA_DLPACK_DEVICE_TYPE
    if owner is not None:
        device_type, device = get_musa_dlpack_device(owner)
    shape_storage = (ctypes.c_int64 * len(shape))(*shape)
    managed = _DLManagedTensor()
    managed.dl_tensor = _DLTensor(
        data=ctypes.c_void_p(ptr),
        device=_DLDevice(int(device_type), int(device)),
        ndim=len(shape),
        dtype=_DLDataType(dtype_fields[0], dtype_fields[1], 1),
        shape=shape_storage,
        strides=None,
        byte_offset=0,
    )
    managed.manager_ctx = None
    managed.deleter = _release_dlpack_view
    address = ctypes.addressof(managed)
    _DLPACK_VIEWS[address] = (managed, shape_storage, owner)

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(address, b"dltensor", None)
    try:
        return torch.utils.dlpack.from_dlpack(capsule)
    except Exception:
        _DLPACK_VIEWS.pop(address, None)
        raise


__all__ = [
    "MUSA_DLPACK_DEVICE_TYPE",
    "get_musa_dlpack_device",
    "tensor_from_musa_ptr",
]
