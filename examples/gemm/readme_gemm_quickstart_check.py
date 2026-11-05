import tilelang
import tilelang.language as T
import torch


tilelang.disable_cache()


@tilelang.jit(target="musa")
def matmul(A, B, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        a_shared = T.alloc_shared((block_M, block_K), dtype)
        b_shared = T.alloc_shared((block_K, block_N), dtype)
        c_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(c_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, ko * block_K], a_shared)
            T.copy(B[ko * block_K, bx * block_N], b_shared)
            T.gemm(a_shared, b_shared, c_local)

        for i, j in T.Parallel(block_M, block_N):
            c_local[i, j] = T.max(c_local[i, j], 0)

        T.copy(c_local, C[by * block_M, bx * block_N])

    return C


def run_case(M=512, N=512, K=512, block_M=128, block_N=128, block_K=64):
    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )

    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)

    c = kernel(a, b)
    ref_c = torch.relu(a @ b)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("README GEMM example check passed.")


if __name__ == "__main__":
    run_case()
