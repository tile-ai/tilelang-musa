import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def parallel_shared_gemm(A, B, block_M, block_N, block_K, dtype="bfloat16", accum_dtype="float32"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[N, K], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        a_shared = T.alloc_shared((block_M, block_K), dtype)
        b_shared = T.alloc_shared((block_N, block_K), dtype)
        c_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(c_local)
        for k_tile in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
            for i, kk in T.Parallel(block_M, block_K):
                global_m = by * block_M + i
                global_k = k_tile * block_K + kk
                a_shared[i, kk] = T.if_then_else(
                    (global_m < M) & (global_k < K),
                    A[global_m, global_k],
                    0,
                )

            for j, kk in T.Parallel(block_N, block_K):
                global_n = bx * block_N + j
                global_k = k_tile * block_K + kk
                b_shared[j, kk] = T.if_then_else(
                    (global_n < N) & (global_k < K),
                    B[global_n, global_k],
                    0,
                )

            T.gemm(a_shared, b_shared, c_local, transpose_B=True)

        T.copy(c_local, C[by * block_M, bx * block_N])

    return C


def _assert_case(M, N, K, block_M, block_N, block_K):
    kernel = parallel_shared_gemm.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype="bfloat16",
        accum_dtype="float32",
    )

    a = torch.randn((M, K), device="musa", dtype=torch.bfloat16)
    b = torch.randn((N, K), device="musa", dtype=torch.bfloat16)
    c = kernel(a, b)
    ref = a @ b.T
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    return kernel


def test_parallel_shared_gemm():
    _assert_case(
        M=512,
        N=512,
        K=512,
        block_M=128,
        block_N=128,
        block_K=64,
    )


def main():
    kernel = _assert_case(
        M=512,
        N=512,
        K=512,
        block_M=128,
        block_N=128,
        block_K=64,
    )
    print(kernel.get_kernel_source())
    print("pass")


if __name__ == "__main__":
    main()
