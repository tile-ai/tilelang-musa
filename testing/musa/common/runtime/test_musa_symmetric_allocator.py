import gc
import tempfile
from pathlib import Path

import pytest
import tilelang.testing
import torch
import torch.distributed as dist
import torch.multiprocessing as mp

import tilelang.distributed.musa_ipc as public_musa_ipc
import tilelang.musa.distributed.musa_ipc as musa_ipc
from tilelang.musa.distributed import MUSASymmetricAllocator


@tilelang.testing.requires_musa
def test_musa_ipc_compatibility_path_uses_same_module():
    assert public_musa_ipc is musa_ipc


def _symmetric_allocator_worker(rank: int, world_size: int, store_path: str) -> None:
    torch.musa.set_device(rank)
    dist.init_process_group(
        "gloo",
        init_method=f"file://{store_path}",
        rank=rank,
        world_size=world_size,
    )
    allocator = None
    first = None
    second = None
    try:
        allocator = MUSASymmetricAllocator(4096, device=rank, device_ids=list(range(world_size)))
        first = allocator.allocate((16,), torch.float32)
        second = allocator.allocate((8,), torch.int32, alignment=512)
        assert first.byte_offset == 0
        assert second.byte_offset == 512
        with pytest.raises(ValueError, match="positive power of two"):
            allocator.allocate((1,), torch.float32, alignment=3)
        with pytest.raises(MemoryError, match="arena has 4096 bytes"):
            allocator.allocate((4096,), torch.float32)

        first.local.fill_(rank + 1)
        second.local.fill_(100 + rank)
        torch.musa.synchronize(rank)
        dist.barrier()

        for peer_rank in range(world_size):
            torch.testing.assert_close(
                first.peers[peer_rank],
                torch.full_like(first.local, peer_rank + 1),
                atol=0,
                rtol=0,
            )
            torch.testing.assert_close(
                second.peers[peer_rank],
                torch.full_like(second.local, 100 + peer_rank),
                atol=0,
                rtol=0,
            )
        torch.musa.synchronize(rank)
        dist.barrier()
    finally:
        del first, second
        gc.collect()
        if allocator is not None:
            allocator.close()
            with pytest.raises(RuntimeError, match="closed symmetric allocator"):
                allocator.allocate((1,), torch.float32)
        dist.destroy_process_group()


@tilelang.testing.requires_musa
def test_symmetric_allocator_rejects_invalid_size_before_collective_init():
    with pytest.raises(ValueError, match="total_bytes must be positive"):
        MUSASymmetricAllocator(0)


@tilelang.testing.requires_musa
def test_symmetric_allocator_two_rank_ipc_correctness():
    if torch.musa.device_count() < 2:
        pytest.skip("requires two MUSA GPUs")
    with tempfile.TemporaryDirectory() as directory:
        store_path = str(Path(directory) / "gloo_store")
        mp.spawn(
            _symmetric_allocator_worker,
            args=(2, store_path),
            nprocs=2,
            join=True,
        )


if __name__ == "__main__":
    tilelang.testing.main()
