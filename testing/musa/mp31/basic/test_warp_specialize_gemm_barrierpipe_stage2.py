import tilelang
import tilelang.language as T
import tilelang.testing
import torch


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)
    num_stages = 2
    mbarrier_list = [128, 128] * num_stages
    # Initialize Kernel Context
    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=256) as (bx, by):
        A_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
        B_shared = T.alloc_shared((num_stages, block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        # create mbarrier for tma
        mbars = T.alloc_barrier(mbarrier_list)

        with T.ws(0):
            T.clear(C_local)

        for ko in range(T.ceildiv(K, block_K)):
            with T.ws(1):
                T.mbarrier_wait_parity(
                    mbarrier=mbars[ko % num_stages + num_stages],
                    parity=((ko // num_stages) % num_stages) ^ 1,
                )
                T.copy(
                    A[by * block_M : (by + 1) * block_M, ko * block_K : (ko + 1) * block_K],
                    A_shared[ko % num_stages, :, :],
                )
                T.copy(
                    B[ko * block_K : (ko + 1) * block_K, bx * block_N : (bx + 1) * block_N],
                    B_shared[ko % num_stages, :, :],
                )
                T.mbarrier_arrive(mbarrier=mbars[ko % num_stages])
            with T.ws(0):
                T.mbarrier_wait_parity(
                    mbarrier=mbars[ko % num_stages],
                    parity=(ko // num_stages) % num_stages,
                )
                T.gemm(A_shared[ko % num_stages, :, :], B_shared[ko % num_stages, :, :], C_local)
                T.mbarrier_arrive(mbarrier=mbars[ko % num_stages + num_stages])

        with T.ws(0):
            T.copy(C_local, C[by * block_M, bx * block_N])

    return C


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_warp_specialize_gemm_barrierpipe_stage2():
    M = 256
    N = 256
    K = 256
    block_M = 128
    block_N = 128
    block_K = 64

    jit_kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )

    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)
    c = jit_kernel(a, b)

    ref_c = a @ b
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)


def main():
    M = 256
    N = 256
    K = 256
    block_M = 128
    block_N = 128
    block_K = 64
    jit_kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
    )

    print(jit_kernel.get_kernel_source())

    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)

    # Run the kernel through the Profiler
    c = jit_kernel(a, b)

    # Reference multiplication using PyTorch
    ref_c = a @ b

    # Validate correctness
    torch.testing.assert_close(c, ref_c, rtol=1e-2, atol=1e-2)
    print("Kernel output matches PyTorch reference.")

    # Profile latency with kernel
    profiler = jit_kernel.get_profiler(tensor_supply_type=tilelang.TensorSupplyType.Normal)

    latency = profiler.do_bench()

    print(f"Latency: {latency} ms")


if __name__ == "__main__":
    main()
