"""Shared memory allocator for distributed communication (IPC + VMM/fabric).

All ops registered via TVM FFI under tl.shared_memory.* namespace.
"""

import ctypes
import operator

import torch
import tvm_ffi

# ---------- TVM FFI function handles ----------


def _missing_shared_memory_func(name):
    def _missing(*args, **kwargs):
        raise RuntimeError(
            f"TileScale shared-memory FFI function '{name}' is unavailable. "
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


_vmm_malloc = _get_required_global_func("tl.shared_memory.vmm_malloc")
_vmm_free = _get_required_global_func("tl.shared_memory.vmm_free")
_create_vmm_handle = _get_required_global_func("tl.shared_memory.create_vmm_handle")
_open_vmm_handle = _get_required_global_func("tl.shared_memory.open_vmm_handle")
_close_vmm_handle = _get_required_global_func("tl.shared_memory.close_vmm_handle")
_open_vmm_handles_contiguous = _get_required_global_func(
    "tl.shared_memory.open_vmm_handles_contiguous"
)
_close_vmm_handles_contiguous = _get_required_global_func(
    "tl.shared_memory.close_vmm_handles_contiguous"
)
_sync_vmm_handles_raw = _get_required_global_func("tl.shared_memory.sync_vmm_handles")

_create_ipc_handle = _get_required_global_func("tl.shared_memory.create_ipc_handle")
_open_ipc_handle = _get_required_global_func("tl.shared_memory.open_ipc_handle")
_close_ipc_handle = _get_required_global_func("tl.shared_memory.close_ipc_handle")
_sync_ipc_handles_raw = _get_required_global_func("tl.shared_memory.sync_ipc_handles")

_supports_vmm_fabric = _get_capability_global_func("tl.shared_memory.supports_vmm_fabric")
_supports_multicast = _get_capability_global_func("tl.shared_memory.supports_multicast")

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

# torch_musa accepts the CUDA array interface for device-pointer interop.
_torch_dtype_to_typestr = {
    torch.float32: "<f4",
    torch.float64: "<f8",
    torch.float16: "<f2",
    torch.int8: "<i1",
    torch.int16: "<i2",
    torch.int32: "<i4",
    torch.int64: "<i8",
    torch.uint8: "<u1",
    torch.bool: "|b1",
    # bfloat16/uint32/uint64 handled specially
}


class _ExternalMUSAArray:
    """Minimal CUDA-array-interface provider for a MUSA device pointer."""

    def __init__(self, ptr: int, shape: tuple, typestr: str, owner=None):
        # PyTorch retains the producer for the lifetime of the resulting
        # storage. Keeping the allocation here therefore also protects aliases.
        self._owner = owner
        self.__cuda_array_interface__ = {
            "data": (ptr, False),
            "shape": shape,
            "typestr": typestr,
            "version": 3,
            "strides": None,
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

    typestr = _torch_dtype_to_typestr.get(dtype)
    if typestr is not None:
        # Standard path via torch_musa's CUDA-array-interface compatibility.
        arr = _ExternalMUSAArray(ptr_val, shape, typestr, _owner)
        tensor = torch.as_tensor(arr, device=f"musa:{device}")
    else:
        # bfloat16 / uint32 / uint64: create as matching-size int type, then view
        element_size = torch.empty((), dtype=dtype).element_size()
        if element_size == 2:
            # proxy_dtype = torch.int16
            proxy_typestr = "<i2"
        elif element_size == 4:
            # proxy_dtype = torch.int32
            proxy_typestr = "<i4"
        elif element_size == 8:
            # proxy_dtype = torch.int64
            proxy_typestr = "<i8"
        else:
            raise ValueError(f"Cannot handle dtype {dtype} with element_size={element_size}")

        arr = _ExternalMUSAArray(ptr_val, shape, proxy_typestr, _owner)
        t = torch.as_tensor(arr, device=f"musa:{device}")
        tensor = t.view(dtype)

    if _owner is not None:
        tensor.untyped_storage()._tilelang_managed_allocation = _owner
    return tensor


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


__all__ = [
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
