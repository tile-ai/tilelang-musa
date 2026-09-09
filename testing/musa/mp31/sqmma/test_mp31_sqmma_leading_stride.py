import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

from .test_mp31_sqmma import (
    ACCUM_DTYPE,
    OUTPUT_DTYPE,
    TOLERANCE_BY_DTYPE,
    TORCH_DTYPE_TO_TILELANG,
    matmul as copy_matmul,
)
from .test_mp31_sqmma_async_copy import matmul as async_copy_matmul
from .test_mp31_sqmma_tma_copy import matmul as tma_copy_matmul


_MATMUL_BY_COPY_KIND = {
    "copy": copy_matmul,
    "async_copy": async_copy_matmul,
    "tma_copy": tma_copy_matmul,
}

# Each case makes one SQMMA operand's physical continuous dimension wider than
# 256 bytes.  Together they cover A/B and K-major/non-K-major descriptors.
_LEADING_STRIDE_CASES = [
    (64, 64, 512, 128, False, False),
    (512, 64, 64, 512, True, False),
    (64, 64, 512, 128, False, True),
    (64, 512, 64, 512, False, False),
]

_LEADING_STRIDE_2048B_CASES = [
    (64, 16, 1024, 128, False, False),
    (1024, 16, 64, 128, True, False),
    (16, 64, 1024, 128, False, True),
    (64, 1024, 64, 512, False, False),
]


@pytest.mark.parametrize("elem_type", [torch.float16, torch.float8_e4m3fn])
@pytest.mark.parametrize("copy_kind", list(_MATMUL_BY_COPY_KIND))
@pytest.mark.parametrize(
    "block_M, block_N, block_K, threads, trans_A, trans_B",
    _LEADING_STRIDE_CASES,
)
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_sqmma_leading_stride(
    elem_type,
    copy_kind,
    block_M,
    block_N,
    block_K,
    threads,
    trans_A,
    trans_B,
):
    M = N = K = 1024
    a_shape = (K, M) if trans_A else (M, K)
    b_shape = (N, K) if trans_B else (K, N)
    A = torch.randn(a_shape, dtype=torch.float16, device="musa").to(elem_type)
    B = torch.randn(b_shape, dtype=torch.float16, device="musa").to(elem_type)
    program = _MATMUL_BY_COPY_KIND[copy_kind](
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        threads,
        dtype=TORCH_DTYPE_TO_TILELANG[elem_type],
        out_dtype=OUTPUT_DTYPE[elem_type],
        accum_dtype=ACCUM_DTYPE[elem_type],
        trans_A=trans_A,
        trans_B=trans_B,
    )
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )

    logical_A = A.T if trans_A else A
    logical_B = B.T if trans_B else B
    expected = logical_A.float().cpu() @ logical_B.float().cpu()
    actual = kernel(A, B).cpu().float()
    rtol, atol = TOLERANCE_BY_DTYPE[elem_type]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@pytest.mark.parametrize("copy_kind", list(_MATMUL_BY_COPY_KIND))
@pytest.mark.parametrize(
    "block_M, block_N, block_K, threads, trans_A, trans_B",
    _LEADING_STRIDE_2048B_CASES,
)
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_sqmma_leading_stride_2048b(
    copy_kind,
    block_M,
    block_N,
    block_K,
    threads,
    trans_A,
    trans_B,
):
    M = N = K = 1024
    a_shape = (K, M) if trans_A else (M, K)
    b_shape = (N, K) if trans_B else (K, N)
    A = torch.randn(a_shape, dtype=torch.float16, device="musa")
    B = torch.randn(b_shape, dtype=torch.float16, device="musa")
    program = _MATMUL_BY_COPY_KIND[copy_kind](
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        threads,
        dtype="float16",
        out_dtype="float16",
        accum_dtype="float32",
        trans_A=trans_A,
        trans_B=trans_B,
    )
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )

    logical_A = A.T if trans_A else A
    logical_B = B.T if trans_B else B
    expected = logical_A.float().cpu() @ logical_B.float().cpu()
    actual = kernel(A, B).cpu().float()
    rtol, atol = TOLERANCE_BY_DTYPE[torch.float16]
    torch.testing.assert_close(actual, expected, rtol=rtol, atol=atol)


@T.prim_func
def _warp_specialized_panelized_gemm(
    A: T.Tensor((64, 512), T.float16),
    B: T.Tensor((512, 64), T.float16),
    C: T.Tensor((64, 64), T.float32),
):
    with T.Kernel(1, threads=512):
        A_shared = T.alloc_shared((64, 256), T.float16)
        B_shared = T.alloc_shared((256, 64), T.float16)
        C_local = T.alloc_fragment((64, 64), T.float32)
        T.clear(C_local)
        for k in T.Pipelined(2, num_stages=2):
            T.copy(A[:, k * 256 : (k + 1) * 256], A_shared)
            T.copy(B[k * 256 : (k + 1) * 256, :], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_sqmma_leading_stride_warp_specialization():
    kernel = tilelang.compile(
        _warp_specialized_panelized_gemm,
        out_idx=[2],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    source = kernel.get_kernel_source()
    assert source.count("tl::tme_load(") == 3
    assert "tl::sqmma_ss<" in source

    A = torch.randn((64, 512), dtype=torch.float16)
    B = torch.randn((512, 64), dtype=torch.float16)
    actual = kernel(A.to("musa"), B.to("musa")).cpu()
    torch.testing.assert_close(actual, A.float() @ B.float(), rtol=1e-3, atol=1e-3)
