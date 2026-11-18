"""MUSA IPC peer-pointer table built on TileScale's shared-memory FFI."""

from __future__ import annotations

import ctypes
from dataclasses import dataclass
from math import prod
from typing import Optional, Sequence

import torch
import torch.distributed as dist

from .shared_memory import _close_ipc_handle, _create_ipc_handle, _open_ipc_handle
from .musa_dlpack import get_musa_dlpack_device


class _DLDevice(ctypes.Structure):
    _fields_ = [("device_type", ctypes.c_int32), ("device_id", ctypes.c_int32)]


class _DLDataType(ctypes.Structure):
    _fields_ = [("code", ctypes.c_uint8), ("bits", ctypes.c_uint8), ("lanes", ctypes.c_uint16)]


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
    torch.int32: (0, 32),
    torch.int64: (0, 64),
    torch.uint8: (1, 8),
    torch.int8: (0, 8),
    torch.bool: (6, 8),
}
_DLPACK_VIEWS: dict[int, tuple[object, ...]] = {}


@_DLManagedTensorDeleter
def _release_dlpack_view(managed: ctypes.POINTER(_DLManagedTensor)) -> None:
    _DLPACK_VIEWS.pop(ctypes.addressof(managed.contents), None)


def _musa_tensor_from_ptr(
    ptr: int,
    shape: Sequence[int],
    dtype: torch.dtype,
    device_type: int,
    device: int,
) -> torch.Tensor:
    """Create a non-owning torch_musa tensor view using a DLPack capsule."""

    shape = tuple(int(dim) for dim in shape)
    if any(dim < 0 for dim in shape):
        raise ValueError(f"peer tensor dimensions must be non-negative, got {shape}")
    if prod(shape) == 0:
        return torch.empty(shape, dtype=dtype, device=f"musa:{device}")
    if ptr <= 0:
        raise ValueError(f"peer tensor pointer must be positive, got {ptr}")
    dtype_fields = _DLPACK_DTYPE.get(dtype)
    if dtype_fields is None:
        raise ValueError(f"unsupported MUSA DLPack dtype: {dtype}")

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
    _DLPACK_VIEWS[address] = (managed, shape_storage)

    capsule_new = ctypes.pythonapi.PyCapsule_New
    capsule_new.restype = ctypes.py_object
    capsule_new.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_void_p]
    capsule = capsule_new(address, b"dltensor", None)
    try:
        return torch.utils.dlpack.from_dlpack(capsule)
    except Exception:
        _DLPACK_VIEWS.pop(address, None)
        raise


@dataclass
class MUSAIpcPeerTable:
    """A device-visible peer pointer table and its imported IPC mappings.

    ``close`` is collective.  Call it before destroying the process group and
    before allowing any rank's owner tensor to be released.
    """

    rank: int
    group_size: int
    local_tensor: torch.Tensor
    peer_ptrs: torch.Tensor
    opened_ptrs: list[int]
    group: dist.ProcessGroup
    _closed: bool = False

    def peer_ptr(self, rank: int) -> int:
        if rank < 0 or rank >= self.group_size:
            raise IndexError(f"peer rank must be in [0, {self.group_size}), got {rank}")
        return int(self.peer_ptrs[rank].item())

    def peer_tensor(self, rank: int, shape: Sequence[int], dtype: torch.dtype) -> torch.Tensor:
        """Return a bounded non-owning MUSA tensor view of one peer mapping."""

        if self._closed:
            raise RuntimeError("cannot create a peer tensor after the IPC table is closed")
        if dtype != self.local_tensor.dtype:
            raise ValueError(
                f"peer dtype must match symmetric allocation dtype {self.local_tensor.dtype}, got {dtype}"
            )
        shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape):
            raise ValueError(f"peer tensor dimensions must be non-negative, got {shape}")
        requested_bytes = prod(shape) * self.local_tensor.element_size()
        allocation_bytes = self.local_tensor.numel() * self.local_tensor.element_size()
        if requested_bytes > allocation_bytes:
            raise ValueError(
                f"peer view requires {requested_bytes} bytes, allocation has {allocation_bytes} bytes"
            )
        ptr = self.peer_ptr(rank)
        if ptr == 0:
            raise RuntimeError(f"peer rank {rank} pointer is null")
        dlpack_device_type, dlpack_device = get_musa_dlpack_device(
            self.local_tensor
        )
        tensor = _musa_tensor_from_ptr(
            ptr,
            shape,
            dtype,
            int(dlpack_device_type),
            int(dlpack_device),
        )
        tensor._tilelang_ipc_peer_table = self
        return tensor

    def close(self) -> None:
        if self._closed:
            return

        local_error = None
        try:
            torch.musa.synchronize(self.local_tensor.device)
        except Exception as error:  # noqa: BLE001 - propagated collectively below
            local_error = error
        self._raise_collective_errors("synchronize device work", local_error)

        local_error = None
        for peer_rank, peer_ptr in enumerate(self.opened_ptrs):
            if peer_rank != self.rank and peer_ptr != 0:
                # The native closer consumes its registry entry even when the
                # runtime close reports an error, so never retry this pointer.
                self.opened_ptrs[peer_rank] = 0
                try:
                    _close_ipc_handle(peer_ptr)
                except Exception as error:  # noqa: BLE001 - collect every rank's failure
                    if local_error is None:
                        local_error = error
        self._raise_collective_errors("close imported IPC mappings", local_error)
        dist.barrier(group=self.group)
        self._closed = True

    def _raise_collective_errors(self, stage: str, local_error: Optional[Exception]) -> None:
        local_status = None
        if local_error is not None:
            local_status = f"{type(local_error).__name__}: {local_error}"
        statuses = [None] * self.group_size
        dist.all_gather_object(statuses, local_status, group=self.group)
        failures = [
            f"rank {rank}: {status}" for rank, status in enumerate(statuses) if status is not None
        ]
        if failures:
            raise RuntimeError(f"MUSA IPC peer-table stage '{stage}' failed ({'; '.join(failures)})")

    def __enter__(self) -> "MUSAIpcPeerTable":
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_musa_ipc_peer_table(
    local_tensor: torch.Tensor,
    *,
    group: Optional[dist.ProcessGroup] = None,
    device_ids: Optional[Sequence[int]] = None,
) -> MUSAIpcPeerTable:
    """Collectively export local allocations and import every peer mapping."""

    if not local_tensor.is_contiguous():
        raise ValueError("local_tensor must be contiguous")
    if local_tensor.device.type != "musa":
        raise ValueError("local_tensor must live on a MUSA device")
    if not dist.is_initialized():
        raise RuntimeError("torch.distributed must be initialized before building an IPC table")

    group = dist.group.WORLD if group is None else group
    rank = dist.get_rank(group=group)
    world_size = dist.get_world_size(group=group)
    local_device = int(local_tensor.device.index or 0)

    if device_ids is None:
        device_ids = list(range(world_size))
    device_ids = [int(device) for device in device_ids]
    if len(device_ids) != world_size:
        raise ValueError(f"device_ids length must equal world_size={world_size}")
    if len(set(device_ids)) != world_size:
        raise ValueError(f"device_ids must be unique, got {device_ids}")
    device_count = int(torch.musa.device_count())
    if any(device < 0 or device >= device_count for device in device_ids):
        raise ValueError(f"device_ids must be in [0, {device_count}), got {device_ids}")
    if local_device != device_ids[rank]:
        raise ValueError(
            f"rank {rank} tensor is on MUSA device {local_device}, "
            f"expected device_ids[{rank}]={device_ids[rank]}"
        )

    local_record = {
        "handle": bytes(_create_ipc_handle(int(local_tensor.data_ptr()))),
        "shape": tuple(int(dim) for dim in local_tensor.shape),
        "dtype": str(local_tensor.dtype),
        "bytes": local_tensor.numel() * local_tensor.element_size(),
    }
    records = [None] * world_size
    dist.all_gather_object(records, local_record, group=group)

    expected = (local_record["shape"], local_record["dtype"], local_record["bytes"])
    for peer_rank, record in enumerate(records):
        actual = (record["shape"], record["dtype"], record["bytes"])
        if actual != expected:
            raise ValueError(
                f"rank {peer_rank} allocation is not symmetric: expected {expected}, got {actual}"
            )

    opened = [0] * world_size
    opened[rank] = int(local_tensor.data_ptr())
    try:
        for peer_rank, record in enumerate(records):
            if peer_rank != rank:
                opened[peer_rank] = int(_open_ipc_handle(record["handle"]))
        peer_ptrs = torch.tensor(opened, dtype=torch.uint64, device=local_tensor.device)
    except Exception as import_error:
        cleanup_errors = []
        for peer_rank, peer_ptr in enumerate(opened):
            if peer_rank != rank and peer_ptr != 0:
                try:
                    _close_ipc_handle(peer_ptr)
                except Exception as cleanup_error:
                    cleanup_errors.append(f"rank {peer_rank}: {cleanup_error}")
        if cleanup_errors:
            raise RuntimeError(
                "IPC table construction failed and imported mapping cleanup failed "
                f"({'; '.join(cleanup_errors)})"
            ) from import_error
        raise

    return MUSAIpcPeerTable(
        rank=rank,
        group_size=world_size,
        local_tensor=local_tensor,
        peer_ptrs=peer_ptrs,
        opened_ptrs=opened,
        group=group,
    )


__all__ = ["MUSAIpcPeerTable", "build_musa_ipc_peer_table"]
