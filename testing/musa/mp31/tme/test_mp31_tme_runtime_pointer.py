import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm


@T.prim_func
def _runtime_pointer_tme_load(
    peer_ptrs: T.Tensor((1,), T.int64),
    output: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        peer = T.make_tensor(peer_ptrs[0], (128,), T.float32)
        shared = T.alloc_shared((128,), T.float32)
        barrier = T.alloc_barrier(128)
        T.tma_copy(peer[0], shared[0], barrier=barrier)
        T.barrier_arrive(barrier)
        T.barrier_wait(barrier, 0)
        for i in T.Parallel(128):
            output[i] = shared[i]


@T.prim_func
def _runtime_pointer_tme_store(
    peer_ptrs: T.Tensor((1,), T.int64),
    source: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        peer = T.make_tensor(peer_ptrs[0], (128,), T.float32)
        shared = T.alloc_shared((128,), T.float32)
        for i in T.Parallel(128):
            shared[i] = source[i]
        T.sync_threads()
        T.tma_copy(shared[0], peer[0])
        T.tma_store_wait(0)


@T.prim_func
def _runtime_pointer_tme_load_2d(
    peer_ptrs: T.Tensor((1,), T.int64),
    output: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        peer = T.make_tensor(peer_ptrs[0], (3, 128), T.float32)
        shared = T.alloc_shared((128,), T.float32)
        barrier = T.alloc_barrier(128)
        T.tma_copy(peer[1, 0], shared[0], barrier=barrier)
        T.barrier_arrive(barrier)
        T.barrier_wait(barrier, 0)
        for i in T.Parallel(128):
            output[i] = shared[i]


@T.prim_func
def _runtime_pointer_tme_store_2d(
    peer_ptrs: T.Tensor((1,), T.int64),
    source: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        peer = T.make_tensor(peer_ptrs[0], (3, 128), T.float32)
        shared = T.alloc_shared((128,), T.float32)
        for i in T.Parallel(128):
            shared[i] = source[i]
        T.sync_threads()
        T.tma_copy(shared[0:128], peer[1, 0:128])
        T.tma_store_wait(0)


@T.prim_func
def _runtime_pointer_tme_noncontiguous_column(
    peer_ptrs: T.Tensor((1,), T.int64),
    output: T.Tensor((8,), T.float32),
):
    with T.Kernel(1, threads=32):
        peer = T.make_tensor(peer_ptrs[0], (8, 8), T.float32)
        shared = T.alloc_shared((8,), T.float32)
        barrier = T.alloc_barrier(32)
        T.tma_copy(peer[0:8, 1], shared[0:8], barrier=barrier)
        T.barrier_arrive(barrier)
        T.barrier_wait(barrier, 0)
        for i in T.Parallel(8):
            output[i] = shared[i]


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_runtime_pointer_codegen():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        artifact = tilelang.lower(
            _runtime_pointer_tme_load,
            target=target,
            enable_device_compile=False,
        )
    assert "tl::tme_load_runtime_pointer(" in artifact.kernel_source

    with target:
        artifact = tilelang.lower(
            _runtime_pointer_tme_store,
            target=target,
            enable_device_compile=False,
        )
    assert "tl::tme_store_runtime_pointer(" in artifact.kernel_source

    with target:
        artifact = tilelang.lower(
            _runtime_pointer_tme_load_2d,
            target=target,
            enable_device_compile=False,
        )
    assert "tl::tme_load_runtime_pointer(" in artifact.kernel_source

    with target:
        artifact = tilelang.lower(
            _runtime_pointer_tme_store_2d,
            target=target,
            enable_device_compile=False,
        )
    assert "tl::tme_store_runtime_pointer(" in artifact.kernel_source


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_runtime_pointer_correctness():
    kernel = tilelang.compile(
        _runtime_pointer_tme_load,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    source = torch.randn(128, dtype=torch.float32, device="musa")
    peer_ptrs = torch.tensor([source.data_ptr()], dtype=torch.int64, device="musa")
    result = kernel(peer_ptrs)
    torch.musa.synchronize()
    torch.testing.assert_close(result, source, atol=0, rtol=0)

    destination = torch.empty_like(source)
    destination_ptrs = torch.tensor([destination.data_ptr()], dtype=torch.int64, device="musa")
    store_kernel = tilelang.compile(
        _runtime_pointer_tme_store,
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    store_kernel(destination_ptrs, source)
    torch.musa.synchronize()
    torch.testing.assert_close(destination, source, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_runtime_pointer_2d_row_correctness():
    load_kernel = tilelang.compile(
        _runtime_pointer_tme_load_2d,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    source = torch.randn((3, 128), dtype=torch.float32, device="musa")
    source_ptrs = torch.tensor([source.data_ptr()], dtype=torch.int64, device="musa")
    result = load_kernel(source_ptrs)
    torch.musa.synchronize()
    torch.testing.assert_close(result, source[1], atol=0, rtol=0)

    store_kernel = tilelang.compile(
        _runtime_pointer_tme_store_2d,
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    destination = torch.zeros_like(source)
    destination_ptrs = torch.tensor([destination.data_ptr()], dtype=torch.int64, device="musa")
    store_kernel(destination_ptrs, source[2])
    torch.musa.synchronize()
    expected = torch.zeros_like(source)
    expected[1].copy_(source[2])
    torch.testing.assert_close(destination, expected, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_runtime_pointer_rejects_noncontiguous_column():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target, pytest.raises(tvm.TVMError, match="must be contiguous"):
        tilelang.lower(
            _runtime_pointer_tme_noncontiguous_column,
            target=target,
            enable_device_compile=False,
        )


if __name__ == "__main__":
    tilelang.testing.main()
