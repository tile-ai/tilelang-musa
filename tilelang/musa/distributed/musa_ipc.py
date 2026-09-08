"""Symmetric MUSA IPC peer tables and sub-allocation."""

from __future__ import annotations

from dataclasses import dataclass
from math import prod
from collections.abc import Sequence

import torch
import torch.distributed as dist

from .dlpack import tensor_from_musa_ptr
from .shared_memory import close_ipc_handle, create_ipc_handle, open_ipc_handle


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

    def peer_tensor(
        self,
        rank: int,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        byte_offset: int = 0,
    ) -> torch.Tensor:
        """Return a bounded non-owning MUSA tensor view of one peer mapping."""

        if self._closed:
            raise RuntimeError("cannot create a peer tensor after the IPC table is closed")
        shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape):
            raise ValueError(f"peer tensor dimensions must be non-negative, got {shape}")
        if byte_offset < 0:
            raise ValueError(f"byte_offset must be non-negative, got {byte_offset}")
        element_size = torch.empty((), dtype=dtype).element_size()
        requested_bytes = prod(shape) * element_size
        allocation_bytes = self.local_tensor.numel() * self.local_tensor.element_size()
        if byte_offset + requested_bytes > allocation_bytes:
            raise ValueError(f"peer view ends at byte {byte_offset + requested_bytes}, allocation has {allocation_bytes} bytes")
        ptr = self.peer_ptr(rank) + byte_offset
        if ptr == 0:
            raise RuntimeError(f"peer rank {rank} pointer is null")
        tensor = tensor_from_musa_ptr(
            ptr,
            shape,
            dtype,
            int(self.local_tensor.device.index or 0),
            self.local_tensor,
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
                    close_ipc_handle(peer_ptr)
                except Exception as error:  # noqa: BLE001 - collect every rank's failure
                    if local_error is None:
                        local_error = error
        self._raise_collective_errors("close imported IPC mappings", local_error)
        dist.barrier(group=self.group)
        self._closed = True

    def _raise_collective_errors(self, stage: str, local_error: Exception | None) -> None:
        local_status = None
        if local_error is not None:
            local_status = f"{type(local_error).__name__}: {local_error}"
        statuses = [None] * self.group_size
        dist.all_gather_object(statuses, local_status, group=self.group)
        failures = [f"rank {rank}: {status}" for rank, status in enumerate(statuses) if status is not None]
        if failures:
            raise RuntimeError(f"MUSA IPC peer-table stage '{stage}' failed ({'; '.join(failures)})")

    def __enter__(self) -> MUSAIpcPeerTable:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


def build_musa_ipc_peer_table(
    local_tensor: torch.Tensor,
    *,
    group: dist.ProcessGroup | None = None,
    device_ids: Sequence[int] | None = None,
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
        raise ValueError(f"rank {rank} tensor is on MUSA device {local_device}, expected device_ids[{rank}]={device_ids[rank]}")

    local_record = {
        "handle": bytes(create_ipc_handle(int(local_tensor.data_ptr()))),
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
            raise ValueError(f"rank {peer_rank} allocation is not symmetric: expected {expected}, got {actual}")

    opened = [0] * world_size
    opened[rank] = int(local_tensor.data_ptr())
    try:
        for peer_rank, record in enumerate(records):
            if peer_rank != rank:
                opened[peer_rank] = int(open_ipc_handle(record["handle"]))
        peer_ptrs = torch.tensor(opened, dtype=torch.uint64, device=local_tensor.device)
    except Exception:
        for peer_rank, peer_ptr in enumerate(opened):
            if peer_rank != rank and peer_ptr != 0:
                close_ipc_handle(peer_ptr)
        raise

    return MUSAIpcPeerTable(
        rank=rank,
        group_size=world_size,
        local_tensor=local_tensor,
        peer_ptrs=peer_ptrs,
        opened_ptrs=opened,
        group=group,
    )


@dataclass(frozen=True)
class MUSASymmetricAllocation:
    """One identically placed sub-allocation on every rank."""

    local: torch.Tensor
    peers: tuple[torch.Tensor, ...]
    byte_offset: int
    nbytes: int


class MUSASymmetricAllocator:
    """Collective bump allocator backed by one symmetric MUSA IPC arena."""

    def __init__(
        self,
        total_bytes: int,
        *,
        group: dist.ProcessGroup | None = None,
        device_ids: Sequence[int] | None = None,
        device: int | None = None,
    ) -> None:
        total_bytes = int(total_bytes)
        if total_bytes <= 0:
            raise ValueError(f"total_bytes must be positive, got {total_bytes}")
        if not dist.is_initialized():
            raise RuntimeError("torch.distributed must be initialized before creating a symmetric allocator")
        if device is None:
            device = int(torch.musa.current_device())
        self.total_bytes = total_bytes
        self._offset = 0
        self._closed = False
        self._arena = torch.empty(total_bytes, dtype=torch.uint8, device=f"musa:{device}")
        self.peer_table = build_musa_ipc_peer_table(self._arena, group=group, device_ids=device_ids)

    def allocate(
        self,
        shape: Sequence[int],
        dtype: torch.dtype,
        *,
        alignment: int = 256,
    ) -> MUSASymmetricAllocation:
        """Collectively reserve an identically placed tensor on all ranks."""

        if self._closed:
            raise RuntimeError("cannot allocate from a closed symmetric allocator")
        shape = tuple(int(dim) for dim in shape)
        if any(dim < 0 for dim in shape):
            raise ValueError(f"allocation dimensions must be non-negative, got {shape}")
        alignment = int(alignment)
        if alignment <= 0 or alignment & (alignment - 1):
            raise ValueError(f"alignment must be a positive power of two, got {alignment}")
        element_size = torch.empty((), dtype=dtype).element_size()
        nbytes = prod(shape) * element_size
        byte_offset = (self._offset + alignment - 1) & -alignment
        end = byte_offset + nbytes
        if end > self.total_bytes:
            raise MemoryError(f"symmetric allocation ends at byte {end}, arena has {self.total_bytes} bytes")
        peers = tuple(
            self.peer_table.peer_tensor(rank, shape, dtype, byte_offset=byte_offset) for rank in range(self.peer_table.group_size)
        )
        self._offset = end
        return MUSASymmetricAllocation(
            local=peers[self.peer_table.rank],
            peers=peers,
            byte_offset=byte_offset,
            nbytes=nbytes,
        )

    def close(self) -> None:
        """Collectively close imported peer mappings."""

        if self._closed:
            return
        self.peer_table.close()
        self._closed = True

    def __enter__(self) -> MUSASymmetricAllocator:
        return self

    def __exit__(self, exc_type, exc, traceback) -> None:
        self.close()


__all__ = [
    "MUSAIpcPeerTable",
    "MUSASymmetricAllocation",
    "MUSASymmetricAllocator",
    "build_musa_ipc_peer_table",
]
