"""MUSA shared-memory primitives for distributed communication (IPC + VMM).

All ops are registered via TVM FFI under the ``tl.musa.shared_memory.*`` namespace.
"""

import operator

import torch
import tvm_ffi

from .dlpack import tensor_from_musa_ptr

# ---------- TVM FFI function handles ----------


def _missing_shared_memory_func(name):
    def _missing(*args, **kwargs):
        raise RuntimeError(
            f"TileLang MUSA shared-memory FFI function '{name}' is unavailable. "
            "This usually means TileLang was built without MUSA shared-memory support. "
            "Build the MUSA runtime with optional IPC/VMM support enabled."
        )

    return _missing


def _get_required_global_func(name):
    func = tvm_ffi.get_global_func(name, allow_missing=True)
    if func is None:
        return _missing_shared_memory_func(name)
    return func


def _get_capability_global_func(name):
    func = tvm_ffi.get_global_func(name, allow_missing=True)
    if func is None:
        return lambda *args, **kwargs: False
    return func


_vmm_malloc = _get_required_global_func("tl.musa.shared_memory.vmm_malloc")
_vmm_free = _get_required_global_func("tl.musa.shared_memory.vmm_free")
_create_vmm_handle = _get_required_global_func("tl.musa.shared_memory.create_vmm_handle")
_open_vmm_handle = _get_required_global_func("tl.musa.shared_memory.open_vmm_handle")
_close_vmm_handle = _get_required_global_func("tl.musa.shared_memory.close_vmm_handle")
_open_vmm_handles_contiguous = _get_required_global_func("tl.musa.shared_memory.open_vmm_handles_contiguous")
_close_vmm_handles_contiguous = _get_required_global_func("tl.musa.shared_memory.close_vmm_handles_contiguous")
_sync_vmm_handles_raw = _get_required_global_func("tl.musa.shared_memory.sync_vmm_handles")

_create_ipc_handle = _get_required_global_func("tl.musa.shared_memory.create_ipc_handle")
_open_ipc_handle = _get_required_global_func("tl.musa.shared_memory.open_ipc_handle")
_close_ipc_handle = _get_required_global_func("tl.musa.shared_memory.close_ipc_handle")
_sync_ipc_handles_raw = _get_required_global_func("tl.musa.shared_memory.sync_ipc_handles")

_supports_vmm_fabric = _get_capability_global_func("tl.musa.shared_memory.supports_vmm_fabric")
_supports_multicast = _get_capability_global_func("tl.musa.shared_memory.supports_multicast")

# ---------- tensor_from_ptr (pure Python, no C++ torch dependency) ----------

_dtype_str_to_torch = {
    "float32": torch.float32,
    "float": torch.float32,
    "float16": torch.float16,
    "half": torch.float16,
    "bfloat16": torch.bfloat16,
    "float64": torch.float64,
    "double": torch.float64,
    "int32": torch.int32,
    "int": torch.int32,
    "int64": torch.int64,
    "long": torch.int64,
    "uint8": torch.uint8,
    "byte": torch.uint8,
    "uint16": torch.uint16,
    "int8": torch.int8,
    "bool": torch.bool,
    "uint32": torch.uint32,
    "uint64": torch.uint64,
}


def tensor_from_ptr(
    ptr_val: int,
    shape: list,
    dtype_str: str = "float32",
    device: int = 0,
    take_ownership: bool = False,
    _owner=None,
) -> torch.Tensor:
    """Create a MUSA tensor viewing external device memory (zero-copy)."""
    if take_ownership:
        raise NotImplementedError("tensor_from_ptr does not yet support ownership transfer")
    if ptr_val == 0:
        raise RuntimeError("Received null pointer (0).")

    dtype = _dtype_str_to_torch.get(dtype_str)
    if dtype is None:
        raise ValueError(f"Unsupported dtype string: '{dtype_str}'")

    if not isinstance(shape, (list, tuple)):
        shape = (shape,)
    try:
        shape = tuple(operator.index(dim) for dim in shape)
    except TypeError as exc:
        raise TypeError("shape dimensions must be integers") from exc
    if any(dim < 0 for dim in shape):
        raise ValueError("shape dimensions must be non-negative")

    numel = 1
    for s in shape:
        numel *= s
    if numel == 0:
        return torch.empty(shape, dtype=dtype, device=f"musa:{device}")

    return tensor_from_musa_ptr(ptr_val, shape, dtype, int(device), _owner)


# ---------- Higher-level Python wrappers ----------


def _sync_vmm_handles(rank, device_ids, buffer_ptrs_gpu_addr, all_gathered_handles):
    """Compatibility wrapper: packs handles into a single bytes blob and calls FFI."""
    num = len(device_ids)
    # all_gathered_handles is a list of bytearrays (or bytes)
    # Pack into single contiguous bytes blob
    # handle_size = len(all_gathered_handles[0]) if all_gathered_handles[0] is not None else 0
    packed = b""
    for h in all_gathered_handles:
        packed += bytes(h)
    _sync_vmm_handles_raw(rank, num, buffer_ptrs_gpu_addr, packed)


def _sync_ipc_handles(rank, device_ids, buffer_ptrs_gpu_addr, all_gathered_handles, root_unique_id_opt=None):
    """Compatibility wrapper for IPC handle sync."""
    num = len(device_ids)
    packed = b""
    for h in all_gathered_handles:
        packed += bytes(h)
    _sync_ipc_handles_raw(rank, num, buffer_ptrs_gpu_addr, packed)


def _create_tensor(shape, dtype):
    """Create a MUSA tensor (simple musaMalloc-backed)."""
    return torch.empty(shape, dtype=dtype, device="musa")


class _ManagedAllocation:
    """Compatibility holder for an externally owned MUSA allocation."""

    def __init__(self, ptr: int, releaser=None):
        self.ptr = ptr
        self.releaser = releaser

    def __del__(self):
        ptr = getattr(self, "ptr", 0)
        if not ptr:
            return
        # Consume the pointer before release so finalization can never retry a
        # pointer whose state is uncertain after a runtime error.
        self.ptr = 0


def create_host_device_tensor(shape, dtype):
    """Reject CUDA-only managed allocation; use VMM/IPC handles instead."""
    raise NotImplementedError(
        "MUSA does not expose a CUDA-style managed host/device allocation here; "
        "allocate with _vmm_malloc or _create_tensor and exchange an IPC/VMM handle."
    )


def supports_vmm_fabric() -> bool:
    """Return whether the active driver completes a VMM fabric round trip."""

    return bool(_supports_vmm_fabric())


def supports_multicast() -> bool:
    """Return whether the MUSA shared-memory runtime supports multicast."""

    return bool(_supports_multicast())


def create_ipc_handle(ptr: int) -> bytes:
    """Export one MUSA allocation as an IPC handle."""

    return bytes(_create_ipc_handle(int(ptr)))


def open_ipc_handle(handle: bytes) -> int:
    """Import one MUSA IPC handle into the current process."""

    return int(_open_ipc_handle(bytes(handle)))


def close_ipc_handle(ptr: int) -> None:
    """Close a pointer returned by :func:`open_ipc_handle`."""

    _close_ipc_handle(int(ptr))


__all__ = [
    "close_ipc_handle",
    "create_ipc_handle",
    "open_ipc_handle",
    "supports_multicast",
    "supports_vmm_fabric",
    "tensor_from_ptr",
    "_create_tensor",
    "_create_ipc_handle",
    "_open_ipc_handle",
    "_close_ipc_handle",
    "_sync_ipc_handles",
    "create_host_device_tensor",
    "_supports_vmm_fabric",
    "_vmm_malloc",
    "_vmm_free",
    "_create_vmm_handle",
    "_open_vmm_handle",
    "_close_vmm_handle",
    "_open_vmm_handles_contiguous",
    "_close_vmm_handles_contiguous",
    "_sync_vmm_handles",
    "_supports_multicast",
]
