import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


FP8_DTYPES = [
    (torch.float8_e4m3fn, "float8_e4m3fn"),
    (torch.float8_e5m2, "float8_e5m2"),
]


def _make_tme_fp8_copy(dtype):
    @T.prim_func
    def kernel(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), dtype)
            barrier = T.alloc_barrier(128)
            T.tma_copy(A[0], shared, barrier=barrier)
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)
            T.tma_copy(shared, C[0])
            T.tma_store_wait(0)

    return kernel


@pytest.mark.parametrize("torch_dtype, tilelang_dtype", FP8_DTYPES)
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_fp8_load_store(torch_dtype, tilelang_dtype):
    kernel = tilelang.compile(
        _make_tme_fp8_copy(tilelang_dtype),
        out_idx=-1,
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    values = torch.randn((128,), dtype=torch.float16, device="musa").to(torch_dtype)
    result = kernel(values)
    torch.musa.synchronize()
    torch.testing.assert_close(
        result.cpu().to(torch.float32),
        values.cpu().to(torch.float32),
        rtol=0.0,
        atol=0.0,
    )


@pytest.mark.parametrize("torch_dtype, tilelang_dtype", FP8_DTYPES)
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_fp8_ws_gemm(torch_dtype, tilelang_dtype):
    M = N = K = 256

    @T.prim_func
    def fp8_gemm(
        A: T.Tensor((M, K), tilelang_dtype),
        B: T.Tensor((N, K), tilelang_dtype),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, 128), T.ceildiv(M, 128), threads=256) as (bx, by):
            As = T.alloc_shared((128, 128), tilelang_dtype)
            Bs = T.alloc_shared((128, 128), tilelang_dtype)
            Cl = T.alloc_fragment((128, 128), "float32")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, 128), num_stages=2):
                T.copy(A[by * 128 : (by + 1) * 128, k * 128 : (k + 1) * 128], As)
                T.copy(B[bx * 128 : (bx + 1) * 128, k * 128 : (k + 1) * 128], Bs)
                T.gemm(As, Bs, Cl, transpose_B=True)
            T.copy(Cl, C[by * 128 : (by + 1) * 128, bx * 128 : (bx + 1) * 128])

    kernel = tilelang.compile(
        fp8_gemm,
        out_idx=[2],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    src = kernel.get_kernel_source()
    assert "tl::tme_load(" in src
    assert "cp_async_gs" not in src

    a = (torch.randn((M, K), dtype=torch.float16, device="musa") * 0.1).to(torch_dtype)
    b = (torch.randn((N, K), dtype=torch.float16, device="musa") * 0.1).to(torch_dtype)
    c = kernel(a, b)
    torch.musa.synchronize()
    ref = (a.float() @ b.float().T).to(torch.float16)
    torch.testing.assert_close(c.cpu(), ref.cpu(), rtol=2e-2, atol=2e-2)
