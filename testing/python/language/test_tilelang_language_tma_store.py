"""Test T.tma_copy() for TMA store on MUSA.

T.tma_copy(shared_buf, global_buf) emits tma_store + tma_store_arrive.
The user must explicitly call T.tma_store_wait() for synchronization.
No barrier argument is needed for stores.
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
    """GEMM with T.copy loads and T.tma_copy for the final shared -> global store."""
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
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
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
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        },
    )
    kernel_source = kernel.get_kernel_source()
    if verbose:
        print(kernel_source)
    assert "tl::tma_store" in kernel_source
    assert "tl::tma_store_arrive()" in kernel_source
    assert kernel_source.count("tl::tma_store_wait<0>()") == 1

    profiler = kernel.get_profiler()

    def ref_program(A, B):
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        return C.to(torch.__getattribute__(out_dtype))

    rtol, atol = tilelang.testing.get_tolerance(torch.float16, profile="gemm_algorithm")
    profiler.assert_allclose(ref_program, atol=atol, rtol=rtol)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_tma_store_2_stages():
    run_gemm_tma_store(num_stages=2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_tma_store_3_stages():
    run_gemm_tma_store(num_stages=3)


if __name__ == "__main__":
    tilelang.testing.main()
