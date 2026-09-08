import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx


@T.prim_func
def _warp_specialized_copy(
    src: T.Tensor((4, 128), T.float32),
    dst: T.Tensor((4, 128), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float32)
        for k in T.Pipelined(4, num_stages=2):
            T.copy(src[k, :], smem)
            T.copy(smem, dst[k, :])


@T.prim_func
def _small_pipeline_copy(
    src: T.Tensor((4, 32), T.float32),
    dst: T.Tensor((4, 32), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((32,), T.float32)
        for k in T.Pipelined(4, num_stages=2):
            T.copy(src[k, :], smem)
            T.copy(smem, dst[k, :])


@T.prim_func
def _pipeline_copy_disable_tme(
    src: T.Tensor((4, 128), T.float32),
    dst: T.Tensor((4, 128), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float32)
        for k in T.Pipelined(4, num_stages=2):
            T.copy(src[k, :], smem, disable_tma=True)
            T.copy(smem, dst[k, :], disable_tma=True)


@T.prim_func
def _pipeline_copy_prefer_sync(
    src: T.Tensor((4, 128), T.float32),
    dst: T.Tensor((4, 128), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float32)
        for k in T.Pipelined(4, num_stages=2):
            T.copy(src[k, :], smem, prefer_instruction="sync")
            T.copy(smem, dst[k, :], disable_tma=True)


@T.prim_func
def _warp_specialized_gemm(
    A: T.Tensor((128, 128), T.float16),
    B: T.Tensor((128, 128), T.float16),
    C: T.Tensor((128, 128), T.float32),
):
    with T.Kernel(1, threads=512):
        A_shared = T.alloc_shared((128, 32), T.float16)
        B_shared = T.alloc_shared((128, 32), T.float16)
        C_local = T.alloc_fragment((128, 128), T.float32)
        C_shared = T.alloc_shared((128, 128), T.float32)
        T.clear(C_local)
        for k in T.Pipelined(4, num_stages=2):
            T.copy(A[:, k * 32 : (k + 1) * 32], A_shared)
            T.copy(B[:, k * 32 : (k + 1) * 32], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_B=True)
        T.copy(C_local, C_shared)
        T.copy(C_shared, C, prefer_instruction="tma")


@T.prim_func
def _warp_specialized_mixed_copy(
    A: T.Tensor((4, 128), T.float32),
    B: T.Tensor((4, 128), T.float32),
    C: T.Tensor((4, 128), T.float32),
):
    with T.Kernel(1, threads=128):
        A_shared = T.alloc_shared((128,), T.float32)
        B_shared = T.alloc_shared((128,), T.float32)
        for k in T.Pipelined(4, num_stages=2):
            T.copy(A[k, :], A_shared)
            T.async_copy(B[k, :], B_shared)
            for i in T.Parallel(128):
                C[k, i] = A_shared[i] + B_shared[i]


def _unchecked_sqmma(A, B, C):
    """Construct a SQMMA tile op without the public T.gemm shape checks."""
    region_op = tirx.op.Op.get("tl.region")
    A_region = T.call_intrin("handle", region_op, A[0, 0, 0], 1, 2, 128, 32)
    B_region = T.call_intrin("handle", region_op, B[0, 0, 0], 1, 2, 128, 32)
    C_region = T.call_intrin("handle", region_op, C[0, 0], 3, 128, 128)
    return T.call_intrin(
        "handle",
        tirx.op.Op.get("tl.tileop.sqmma_gemm"),
        A_region,
        B_region,
        C_region,
        False,
        True,
        128,
        128,
        32,
        0,
        False,
        32,
        32,
        0,
        0,
        1,
        0,
        0,
        0,
        0,
    )


@T.prim_func
def _sqmma_with_unsliced_leading_dimension():
    with T.Kernel(1, threads=128):
        A_shared = T.alloc_shared((2, 128, 32), T.float16)
        B_shared = T.alloc_shared((2, 128, 32), T.float16)
        C_local = T.alloc_fragment((128, 128), T.float32)
        T.clear(A_shared)
        T.clear(B_shared)
        T.clear(C_local)
        T.evaluate(_unchecked_sqmma(A_shared, B_shared, C_local))


def _lower(func):
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        return tilelang.lower(func, target=target, enable_device_compile=False)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_sqmma_rejects_unsliced_leading_dimension():
    with pytest.raises(tvm.TVMError, match="A leading region dimensions must have extent one"):
        _lower(_sqmma_with_unsliced_leading_dimension)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_warp_specialized_copy_codegen_and_runtime():
    source = _lower(_warp_specialized_copy).kernel_source
    assert "tl::tme_load(" in source
    assert "tl::tme_barrier_wait(" in source
    assert "tl::tme_barrier_arrive(" in source
    assert "cp_async_gs" not in source

    kernel = tilelang.compile(
        _warp_specialized_copy,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(512, dtype=torch.float32).reshape(4, 128)
    dst = kernel(src.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(dst.cpu(), src, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_small_pipeline_copy_selects_tme():
    source = _lower(_small_pipeline_copy).kernel_source
    assert "tl::tme_load(" in source

    kernel = tilelang.compile(
        _small_pipeline_copy,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(128, dtype=torch.float32).reshape(4, 32)
    dst = kernel(src.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(dst.cpu(), src, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_pipeline_copy_disable_tma_selects_async():
    source = _lower(_pipeline_copy_disable_tme).kernel_source
    assert "tl::tme_load(" not in source
    assert "cp_async_gs" in source


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_pipeline_copy_prefer_sync_stays_simt():
    source = _lower(_pipeline_copy_prefer_sync).kernel_source
    assert "tl::tme_load(" not in source
    assert "cp_async_gs" not in source


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_warp_specialized_gemm_codegen_and_runtime():
    source = _lower(_warp_specialized_gemm).kernel_source
    assert source.count("tl::tme_load(") == 2
    assert "tl::sqmma_ss<" in source
    assert "tl::tme_barrier_wait(" in source
    assert "tl::tme_barrier_arrive(" in source
    assert "cp_async_gs" not in source

    kernel = tilelang.compile(
        _warp_specialized_gemm,
        out_idx=[2],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    A = torch.randn(128, 128, dtype=torch.float16)
    B = torch.randn(128, 128, dtype=torch.float16)
    C = kernel(A.to("musa"), B.to("musa"))
    torch.musa.synchronize()
    ref = A.float() @ B.float().T
    torch.testing.assert_close(C.cpu(), ref, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_warp_specialized_mixed_copy_codegen_and_runtime():
    source = _lower(_warp_specialized_mixed_copy).kernel_source
    assert "tl::tme_load(" in source
    assert "tl::cp_async_gs<4>" in source
    assert "tl::cp_async_wait<0>()" in source
    assert "tl::tme_barrier_arrive(" in source

    kernel = tilelang.compile(
        _warp_specialized_mixed_copy,
        out_idx=[2],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    A = torch.randn(4, 128, dtype=torch.float32)
    B = torch.randn(4, 128, dtype=torch.float32)
    C = kernel(A.to("musa"), B.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(C.cpu(), A + B, atol=0, rtol=0)
