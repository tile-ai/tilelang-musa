import gc
import struct
import tempfile
from pathlib import Path

import pytest
import tilelang.testing
import torch
import torch.distributed as dist
import torch.multiprocessing as mp
import tvm_ffi

import tilelang.distributed.shared_memory as public_shared_memory
import tilelang.musa.distributed.shared_memory as musa_shared_memory
from tilelang.musa.distributed import (
    close_ipc_handle,
    create_ipc_handle,
    open_ipc_handle,
    supports_multicast,
    supports_vmm_fabric,
    tensor_from_ptr,
)


def _ipc_suballocation_worker(rank: int, world_size: int, store_path: str) -> None:
    torch.musa.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    peer_view = None
    peer_ptr = None
    try:
        storage = torch.arange(300, dtype=torch.int32, device=f"musa:{rank}")
        storage.add_(rank * 1000)
        tensor = storage[16:273]
        handle = create_ipc_handle(tensor.data_ptr())
        offset, allocation_size = struct.unpack("<QQ", handle[:16])
        assert offset >= 16 * tensor.element_size()
        assert offset + tensor.numel() * tensor.element_size() <= allocation_size

        handles = [None] * world_size
        dist.all_gather_object(handles, handle)
        peer_rank = (rank + 1) % world_size
        peer_ptr = open_ipc_handle(handles[peer_rank])
        peer_view = tensor_from_ptr(
            peer_ptr,
            list(tensor.shape),
            dtype_str="int32",
            device=rank,
        )
        torch.musa.synchronize(rank)
        expected = torch.arange(16, 273, dtype=torch.int32, device=f"musa:{rank}")
        expected.add_(peer_rank * 1000)
        torch.testing.assert_close(peer_view, expected, atol=0, rtol=0)
    finally:
        del peer_view
        gc.collect()
        if peer_ptr is not None:
            close_ipc_handle(peer_ptr)
        dist.destroy_process_group()


@tilelang.testing.requires_musa
def test_shared_memory_compatibility_path_uses_same_module():
    assert public_shared_memory is musa_shared_memory


@tilelang.testing.requires_musa
def test_musa_shared_memory_ffi_is_registered():
    names = (
        "create_ipc_handle",
        "open_ipc_handle",
        "close_ipc_handle",
        "vmm_malloc",
        "vmm_free",
        "supports_vmm_fabric",
        "supports_multicast",
    )
    for name in names:
        assert tvm_ffi.get_global_func(f"tl.musa.shared_memory.{name}", allow_missing=True) is not None


@tilelang.testing.requires_musa
def test_musa_shared_memory_rejects_invalid_arguments():
    with pytest.raises(RuntimeError, match="non-zero address"):
        create_ipc_handle(0)
    vmm_malloc = tvm_ffi.get_global_func("tl.musa.shared_memory.vmm_malloc")
    with pytest.raises(RuntimeError, match="must be positive"):
        vmm_malloc(0)


@tilelang.testing.requires_musa
def test_musa_ipc_export_and_pointer_view():
    source = torch.arange(32, dtype=torch.float32, device="musa")
    handle = create_ipc_handle(source.data_ptr())
    assert len(handle) == 80

    view = tensor_from_ptr(
        source.data_ptr(),
        list(source.shape),
        dtype_str="float32",
        device=source.device.index or 0,
        _owner=source,
    )
    torch.testing.assert_close(view, source, atol=0, rtol=0)
    view.add_(1)
    torch.testing.assert_close(source, torch.arange(1, 33, dtype=torch.float32, device="musa"))


@tilelang.testing.requires_musa
def test_musa_ipc_preserves_suballocation_offset():
    if torch.musa.device_count() < 2:
        pytest.skip("requires two MUSA GPUs")
    with tempfile.TemporaryDirectory() as directory:
        store_path = str(Path(directory) / "gloo_store")
        mp.spawn(
            _ipc_suballocation_worker,
            args=(2, store_path),
            nprocs=2,
            join=True,
        )


@tilelang.testing.requires_musa
def test_musa_shared_memory_capabilities_are_fail_closed():
    assert isinstance(supports_vmm_fabric(), bool)
    assert supports_multicast() is False


if __name__ == "__main__":
    tilelang.testing.main()
