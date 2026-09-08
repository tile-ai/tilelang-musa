"""MUSA distributed-memory building blocks."""

from .dlpack import (
    MUSA_DLPACK_DEVICE_TYPE,
    get_musa_dlpack_device,
    tensor_from_musa_ptr,
)
from . import shared_memory as shared_memory
from .shared_memory import (
    close_ipc_handle,
    create_ipc_handle,
    open_ipc_handle,
    supports_multicast,
    supports_vmm_fabric,
    tensor_from_ptr,
)
from . import musa_ipc as musa_ipc
from .musa_ipc import (
    MUSAIpcPeerTable,
    MUSASymmetricAllocation,
    MUSASymmetricAllocator,
    build_musa_ipc_peer_table,
)

__all__ = [
    "MUSA_DLPACK_DEVICE_TYPE",
    "MUSAIpcPeerTable",
    "MUSASymmetricAllocation",
    "MUSASymmetricAllocator",
    "build_musa_ipc_peer_table",
    "close_ipc_handle",
    "create_ipc_handle",
    "get_musa_dlpack_device",
    "open_ipc_handle",
    "supports_multicast",
    "supports_vmm_fabric",
    "tensor_from_musa_ptr",
    "tensor_from_ptr",
]
