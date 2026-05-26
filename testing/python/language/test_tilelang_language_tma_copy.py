"""Test T.tma_copy() with user-managed mbarrier synchronization on MUSA.

For TMA loads (global -> shared):
  T.tma_copy() emits only expect_tx + tma_load (no arrive, no wait).
  The user must explicitly call T.barrier_arrive() and T.mbarrier_wait_parity().
  This allows multiple tma_copy operations to share a single barrier arrive.
  Pipeline buffer versioning expands the barrier to num_stages versions automatically.

For TMA stores (shared -> global):
  T.tma_copy() emits tma_store + tma_store_arrive (no wait).
  The user must explicitly call T.tma_store_wait() for synchronization.
  No barrier argument is needed for stores.
"""

import tilelang
import tilelang.language as T
import tilelang.testing
import torch
import pytest


def matmul_tma_copy(
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
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            mbar_A = T.alloc_barrier(threads)
            mbar_B = T.alloc_barrier(threads)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                T.tma_copy(A[by * block_M, k * block_K], A_shared, barrier=mbar_A)
                T.barrier_arrive(mbar_A)
                T.tma_copy(B[k * block_K, bx * block_N], B_shared, barrier=mbar_B)
                T.barrier_arrive(mbar_B)
                T.barrier_wait(mbar_A, k % 2)
                T.barrier_wait(mbar_B, k % 2)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_tma_copy(num_stages, verbose=False):
    M, N, K = 256, 256, 1024
    block_M, block_N, block_K = 128, 128, 32
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    threads = 128

    program = matmul_tma_copy(
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
    if verbose:
        print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        return C.to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_tma_copy_pipeline_2_stages():
    run_gemm_tma_copy(num_stages=2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_tma_copy_pipeline_3_stages():
    run_gemm_tma_copy(num_stages=3)


def per_warp_tma_copy_kernel():
    M = 32
    K = 1024
    num_threads = 256
    warp_size = 32
    num_warps = num_threads // warp_size

    @T.prim_func
    def main(A: T.Tensor((M, K), T.float32), B: T.Tensor((M,), T.float32)):
        with T.Kernel(T.ceildiv(M, num_warps), threads=num_threads) as pid:
            tid = T.get_thread_binding()
            warp_idx = tid // warp_size
            row = pid * num_warps + warp_idx
            a_shared = T.alloc_shared((num_warps, K), dtype=T.float32)
            mbars = T.alloc_barrier([warp_size] * num_warps)
            T.tma_copy(
                A[row, 0:K],
                a_shared[warp_idx, 0:K],
                barrier=mbars[warp_idx],
                leader_scope_threads=warp_size,
            )
            T.mbarrier_arrive(mbarrier=mbars[warp_idx])
            T.mbarrier_wait_parity(mbarrier=mbars[warp_idx], parity=0)
            if tid % warp_size == 0:
                B[row] = a_shared[warp_idx, 0]

    return main


def block_tma_copy_kernel():
    M = 256
    K = 1024
    num_threads = 256

    @T.prim_func
    def main(A: T.Tensor((M, K), T.float32), B: T.Tensor((1,), T.float32)):
        with T.Kernel(T.ceildiv(M, num_threads), threads=num_threads) as pid:
            tid = T.get_thread_binding()
            a_shared = T.alloc_shared((1, K), dtype=T.float32)
            mbar = T.alloc_barrier([num_threads])
            T.tma_copy(A[pid, 0:K], a_shared[0, 0:K], barrier=mbar)
            T.mbarrier_arrive(mbarrier=mbar)
            T.mbarrier_wait_parity(mbarrier=mbar, parity=0)
            if tid == 0:
                B[pid] = a_shared[0, 0]

    return main


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_copy_per_warp_leader_scope_codegen():
    kernel = tilelang.compile(per_warp_tma_copy_kernel(), out_idx=[1])
    source = kernel.get_kernel_source()
    assert "tl_shuffle_elect<32>()" in source, "Expected per-warp elect<32>"


@tilelang.testing.requires_cuda
@tilelang.testing.requires_cuda_compute_version_ge(9, 0)
def test_tma_copy_default_block_leader_scope_codegen():
    kernel = tilelang.compile(block_tma_copy_kernel(), out_idx=[1])
    source = kernel.get_kernel_source()
    assert "tl_shuffle_elect<256>()" in source, "Expected block-wide elect<256>"


def matmul_tma_copy_store(
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
    """GEMM using T.tma_copy for both load (global->shared) and store (shared->global)."""
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
            # Store result: fragment -> shared -> global via T.tma_copy (no barrier needed)
            T.copy(C_local, C_shared)
            T.tma_copy(C_shared, C[by * block_M, bx * block_N])
            T.tma_store_wait()

    return main


def run_gemm_tma_copy_store(num_stages, verbose=False):
    M, N, K = 256, 256, 1024
    block_M, block_N, block_K = 128, 128, 32
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    threads = 128

    program = matmul_tma_copy_store(
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
    assert "tl::tma_store" in kernel_source
    assert "tl::tma_store_arrive" in kernel_source
    assert kernel_source.count("tl::tma_store_wait<0>()") == 1
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        return C.to(torch.__getattribute__(out_dtype))

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


def test_copy_prefer_tma_lowers_as_synchronous_tma_load():
    @T.prim_func
    def main(x: T.Tensor((128, 32), T.float32)):
        with T.Kernel(threads=128):
            x_shared = T.alloc_shared((128, 32), T.float32)
            T.copy(x, x_shared, annotations={"prefer_instruction": "tma"})

    target = {"kind": "cuda", "arch": "sm_90"}
    artifact = tilelang.lower(main, target=target, enable_device_compile=False)
    device_source = str(artifact.kernel_source)
    assert "tl::tma_load" in device_source
    assert "x_to_x_shared_mbarrier_mem" in device_source
    assert "x_to_x_shared_mbarrier[0]" in device_source
    assert "arrive_and_expect_tx" in device_source
    assert ".wait(0)" in device_source


def test_copy_prefer_tma_lowers_as_synchronous_musa_tma_load():
    @T.prim_func
    def main(x: T.Tensor((128, 32), T.float32)):
        with T.Kernel(threads=128):
            x_shared = T.alloc_shared((128, 32), T.float32)
            T.copy(x, x_shared, prefer_instruction="tma")

    target = {"kind": "musa", "arch": "mp_31"}
    artifact = tilelang.lower(main, target=target, enable_device_compile=False)
    device_source = str(artifact.kernel_source)
    assert "tl::tma_load" in device_source
    assert "__musa_async_bar_record(1)" in device_source
    assert "__musa_async_init_arrival(1" in device_source
    assert "tl::mbarrier_arrive_expect_tx(1, 16384)" in device_source
    assert "__musa_async_wait(1, 0)" in device_source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_tma_copy_store_pipeline_2_stages():
    run_gemm_tma_copy_store(num_stages=2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_tma_copy_store_pipeline_3_stages():
    run_gemm_tma_copy_store(num_stages=3)


if __name__ == "__main__":
    tilelang.testing.main()
