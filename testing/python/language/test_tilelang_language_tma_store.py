"""Tests for TMA store (shared -> global).

Explicit T.tma_copy(shared_buf, global_buf) emits tma_store + tma_store_arrive
(no wait). The user must explicitly call T.tma_store_wait() for
synchronization.

Plain T.copy(shared_buf, global_buf) may also auto-lower to tma_store when the
store-side TMA constraints are satisfied. In that case lowering emits both
tma_store_arrive and tma_store_wait automatically.
"""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch


def matmul_tma_store(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    in_dtype,
    out_dtype,
    accum_dtype,
    threads,
    num_stages,
):
    """GEMM with explicit TMA loads and T.tma_copy for the final shared -> global store."""
    A_shape = (M, K)
    B_shape = (K, N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K), in_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_shared = T.alloc_shared((block_M, block_N), out_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            mbar_A = T.alloc_barrier(threads)
            mbar_B = T.alloc_barrier(threads)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.tma_copy(A[by * block_M, k * block_K], A_shared, barrier=mbar_A)
                T.barrier_arrive(mbar_A)
                T.tma_copy(B[k * block_K, bx * block_N], B_shared, barrier=mbar_B)
                T.barrier_arrive(mbar_B)
                T.mbarrier_wait_parity(mbar_A, k % 2)
                T.mbarrier_wait_parity(mbar_B, k % 2)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C_shared)
            T.tma_copy(C_shared, C[by * block_M, bx * block_N])
            T.tma_store_wait()

    return main


def run_gemm_tma_store(num_stages, verbose=False):
    M, N, K = 32, 32, 32
    block_M, block_N, block_K = 16, 16, 16
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    threads = 32

    program = matmul_tma_store(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        in_dtype,
        out_dtype,
        accum_dtype,
        threads,
        num_stages,
    )
    kernel = tilelang.compile(
        program,
        out_idx=[2],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    kernel_source = kernel.get_kernel_source()
    if verbose:
        print(kernel_source)
    assert "tma_store_arrive" in kernel_source, "Expected tma_store_arrive in kernel source"

    profiler = kernel.get_profiler()

    def ref_program(A, B):
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        return C.to(torch.__getattribute__(out_dtype))

    rtol, atol = tilelang.testing.get_tolerance(torch.float16, profile="gemm_algorithm")
    profiler.assert_allclose(ref_program, atol=atol, rtol=rtol)


def auto_tma_store_copy(M, N, block_M, block_N, dtype, threads):
    @T.prim_func
    def main(
        A: T.Tensor((M, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_N), dtype)
            T.copy(A[by * block_M, bx * block_N], A_shared)
            T.copy(A_shared, C[by * block_M, bx * block_N])

    return main


def run_auto_tma_store_copy():
    M = N = 256
    block_M = block_N = 128
    dtype = T.float16
    threads = 128

    program = auto_tma_store_copy(M, N, block_M, block_N, dtype, threads)
    kernel = tilelang.compile(
        program,
        out_idx=[1],
        pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
    )
    kernel_source = kernel.get_kernel_source()
    assert "tma_store_arrive" in kernel_source, "Expected auto tma_store_arrive in kernel source"
    assert "tma_store_wait" in kernel_source, "Expected auto tma_store_wait in kernel source"

    profiler = kernel.get_profiler()

    def ref_program(A):
        return A

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_tma_store_2_stages():
    run_gemm_tma_store(num_stages=2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_tma_store_3_stages():
    run_gemm_tma_store(num_stages=3)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_plain_copy_auto_tma_store():
    run_auto_tma_store_copy()


if __name__ == "__main__":
    tilelang.testing.main()
