import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
import torch
from tilelang.tileop.base import GemmWarpPolicy

tilelang.disable_cache()


@tilelang.jit(target="musa")
def matmul_trans_b(A, B, block_M, block_N, block_K, dtype, accum_dtype, num_warp, policy):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[N, K], dtype]
    C = T.empty((M, N), dtype)

    threads = num_warp * 32
    if policy == "square":
        warp_policy = GemmWarpPolicy.Square
    elif policy == "m":
        warp_policy = GemmWarpPolicy.FullRow
    elif policy == "n":
        warp_policy = GemmWarpPolicy.FullCol
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        a_shared = T.alloc_shared((block_M, block_K), dtype)
        b_shared = T.alloc_shared((block_N, block_K), dtype)
        c_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(c_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], a_shared)
            T.copy(B[bx * block_N, k * block_K], b_shared)
            T.gemm(a_shared, b_shared, c_local, transpose_B=True, policy=warp_policy)

        T.copy(c_local, C[by * block_M, bx * block_N])

    return C


def _make_inputs(M, N, K, dtype):
    torch_dtype = T.dtype(dtype).as_torch()
    if torch_dtype is torch.float8_e4m3fn:
        a = torch.randint(-128, 128, (M, K), device="musa", dtype=torch.int8).to(torch_dtype)
        b = torch.randint(-128, 128, (N, K), device="musa", dtype=torch.int8).to(torch_dtype)
    else:
        a = torch.randn(M, K, device="musa", dtype=torch_dtype)
        b = torch.randn(N, K, device="musa", dtype=torch_dtype)
    return a, b


def _assert_gemm_trans_b_case(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, num_warp, policy):
    kernel = matmul_trans_b.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
        num_warp=num_warp,
        policy=policy,
    )
    a, b = _make_inputs(M, N, K, dtype)
    c = kernel(a, b)
    ref = a @ b.T

    if a.dtype is torch.float8_e4m3fn:
        torch.testing.assert_close(c.float(), ref.float(), rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    return kernel


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16", "float8_e4m3"])
def test_gemm_trans_b(dtype):
    _assert_gemm_trans_b_case(
        M=256,
        N=256,
        K=256,
        block_M=128,
        block_N=128,
        block_K=64,
        dtype=dtype,
        accum_dtype="float32",
        num_warp=4,
        policy="square",
    )


def main():
    kernel = _assert_gemm_trans_b_case(
        M=256,
        N=256,
        K=256,
        block_M=128,
        block_N=128,
        block_K=64,
        dtype="float16",
        accum_dtype="float32",
        num_warp=4,
        policy="square",
    )
    print(kernel.get_kernel_source())
    print("pass")


if __name__ == "__main__":
    main()
