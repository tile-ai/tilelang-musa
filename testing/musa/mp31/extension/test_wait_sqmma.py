import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa")
def matmul(A, B, block_M, block_N, block_K, dtype, accum_dtype, num_warp):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)
    threads = num_warp * 32

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        a_shared = T.alloc_shared((block_M, block_K), dtype)
        b_shared = T.alloc_shared((block_K, block_N), dtype)
        c_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(c_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], a_shared)
            T.copy(B[k * block_K, bx * block_N], b_shared)
            T.gemm(a_shared, b_shared, c_local, wg_wait=-1)
            T.wait_wgmma()
        T.copy(c_local, C[by * block_M, bx * block_N])

    return C


@tilelang.jit(target="musa")
def matmul_with_independent_compute(A, B, block_M, block_N, block_K, dtype, accum_dtype, num_warp):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)
    D = T.empty((M, N), accum_dtype)
    threads = num_warp * 32

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        a_shared = T.alloc_shared((block_M, block_K), dtype)
        b_shared = T.alloc_shared((block_K, block_N), dtype)
        c_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        d_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(c_local)
        T.clear(d_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], a_shared)
            T.copy(B[k * block_K, bx * block_N], b_shared)
            T.gemm(a_shared, b_shared, c_local, wg_wait=-1)
            for i, j in T.Parallel(block_M, block_N):
                a_val = T.cast(a_shared[i, 0], accum_dtype)
                b_val = T.cast(b_shared[0, j], accum_dtype)
                d_local[i, j] += a_val + b_val
            T.wait_wgmma()

        T.copy(c_local, C[by * block_M, bx * block_N])
        T.copy(d_local, D[by * block_M, bx * block_N])

    return C, D


def independent_compute_reference(a, b, block_K):
    M, K = a.shape
    _, N = b.shape
    a_fp32 = a.to(torch.float32)
    b_fp32 = b.to(torch.float32)

    ref_d = torch.zeros((M, N), device=a.device, dtype=torch.float32)
    for k_base in range(0, K, block_K):
        ref_d += a_fp32[:, k_base].unsqueeze(1)
        ref_d += b_fp32[k_base, :].unsqueeze(0)
    return ref_d


def test_wait_wgmma():
    M = 512
    N = 512
    K = 512
    block_M = 128
    block_N = 128
    block_K = 64

    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype="float16",
        accum_dtype="float32",
        num_warp=4,
    )

    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)
    c = kernel(a, b)
    ref = a @ b
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


def test_wait_wgmma_with_independent_compute():
    M = 512
    N = 512
    K = 512
    block_M = 128
    block_N = 128
    block_K = 64

    kernel = matmul_with_independent_compute.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype="float16",
        accum_dtype="float32",
        num_warp=4,
    )

    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)
    c, d = kernel(a, b)
    ref_c = a @ b
    ref_d = independent_compute_reference(a, b, block_K)

    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)


def main():
    M = 512
    N = 512
    K = 512
    block_M = 128
    block_N = 128
    block_K = 64

    kernel_0 = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype="float16",
        accum_dtype="float32",
        num_warp=4,
    )
    print(kernel_0.get_kernel_source())

    kernel_1 = matmul_with_independent_compute.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype="float16",
        accum_dtype="float32",
        num_warp=4,
    )
    print(kernel_1.get_kernel_source())

    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)
    c, d = kernel_1(a, b)
    ref_c = a @ b
    ref_d = independent_compute_reference(a, b, block_K)
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    torch.testing.assert_close(d, ref_d, rtol=1e-2, atol=1e-2)
    print("pass")


if __name__ == "__main__":
    main()
