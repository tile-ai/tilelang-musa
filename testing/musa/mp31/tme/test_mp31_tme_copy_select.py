import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@T.prim_func
def _copy_select_tme_load(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float32)
        T.copy(src, smem, prefer_instruction="tma")
        T.copy(smem, dst)


@T.prim_func
def _copy_select_tme_store(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float32)
        T.copy(src, smem)
        T.copy(smem, dst, prefer_instruction="tma")


@T.prim_func
def _copy_auto_tme_store(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float32)
        T.copy(src, smem)
        T.copy(smem, dst)


@T.prim_func
def _copy_auto_tme_store_singleton_aligned_regions(
    src: T.Tensor((8, 16), T.float32),
    dst: T.Tensor((2, 8, 3, 16), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((8, 16), T.float32)
        T.copy(src, smem)
        T.copy(smem, dst[1, 0:8, 2:3, 0:16])


@T.prim_func
def _copy_disable_tme_store(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float32)
        T.copy(src, smem)
        T.copy(smem, dst, disable_tma=True)


@T.prim_func
def _copy_prefer_sync_store(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float32)
        T.copy(src, smem)
        T.copy(smem, dst, prefer_instruction="sync")


@T.prim_func
def _copy_invalid_preference(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float32)
        T.copy(src, smem)
        T.copy(smem, dst, prefer_instruction="invalid")


def _lower(func):
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        return tilelang.lower(func, target=target, enable_device_compile=False)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_copy_select_tme_load_codegen_and_runtime():
    artifact = _lower(_copy_select_tme_load)
    source = artifact.kernel_source
    assert "tl::tme_load(" in source
    assert "tl::tme_barrier_arrive(" in source
    assert "tl::tme_barrier_wait(" in source

    kernel = tilelang.compile(
        _copy_select_tme_load,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(128, dtype=torch.float32)
    dst = kernel(src.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(dst.cpu(), src, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_copy_select_tme_store_codegen_and_runtime():
    artifact = _lower(_copy_select_tme_store)
    source = artifact.kernel_source
    assert "tl::tme_store(" in source
    assert "tl::tme_store_commit()" in source
    assert "tl::tme_store_read_wait()" in source

    kernel = tilelang.compile(
        _copy_select_tme_store,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(128, dtype=torch.float32)
    dst = kernel(src.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(dst.cpu(), src, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_copy_auto_tme_store_codegen_and_runtime():
    artifact = _lower(_copy_auto_tme_store)
    assert "tl::tme_store(" in artifact.kernel_source

    kernel = tilelang.compile(
        _copy_auto_tme_store,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(128, dtype=torch.float32)
    dst = kernel(src.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(dst.cpu(), src, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_copy_auto_tme_store_singleton_aligned_regions():
    artifact = _lower(_copy_auto_tme_store_singleton_aligned_regions)
    assert "tl::tme_store(" in artifact.kernel_source

    kernel = tilelang.compile(
        _copy_auto_tme_store_singleton_aligned_regions,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(8 * 16, dtype=torch.float32).reshape(8, 16)
    dst = kernel(src.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(dst[1, :, 2, :].cpu(), src, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_copy_disable_tma_forces_simt():
    artifact = _lower(_copy_disable_tme_store)
    assert "tl::tme_store(" not in artifact.kernel_source


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_copy_prefer_sync_forces_simt():
    artifact = _lower(_copy_prefer_sync_store)
    assert "tl::tme_store(" not in artifact.kernel_source


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_copy_rejects_unknown_preference():
    with pytest.raises(tvm.TVMError, match="Unknown copy prefer_instruction"):
        _lower(_copy_invalid_preference)
