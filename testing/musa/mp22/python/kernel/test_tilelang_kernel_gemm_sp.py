import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tilelang.utils.sparse import get_e_factor, randn_semi_sparse, torch_compress


TARGET = {"kind": "musa", "arch": "mp_22"}


def _sparse_matmul_source():
    M = 16
    N = 8
    K = 32
    block_M = 16
    block_N = 8
    block_K = 32
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    metadata_dtype = T.int16
    e_factor = get_e_factor(in_dtype, metadata_dtype)

    @T.prim_func
    def main(
        A_sparse: T.Tensor((M, K // 2), in_dtype),
        E: T.Tensor((M, K // e_factor), metadata_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=32) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), metadata_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_frag)
            T.copy(A_sparse[by * block_M, 0], A_shared)
            T.copy(E[by * block_M, 0], E_shared)
            T.copy(B[0, bx * block_N], B_shared)
            T.gemm_sp(A_shared, E_shared, B_shared, C_frag, False, False, False)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def _sparse_matmul(
    M,
    N,
    K,
    block_M=16,
    block_N=16,
    block_K=16,
    threads=32,
    stages=0,
    trans_A=False,
    trans_E=False,
    trans_B=False,
):
    dtype = T.float16
    accum_dtype = T.float32
    metadata_dtype = T.int16
    e_factor = get_e_factor(dtype, metadata_dtype)
    a_sparse_shape = (K // 2, M) if trans_A else (M, K // 2)
    e_shape = (K // e_factor, M) if trans_E else (M, K // e_factor)
    b_shape = (N, K) if trans_B else (K, N)
    a_shared_shape = (block_K // 2, block_M) if trans_A else (block_M, block_K // 2)
    e_shared_shape = (block_K // e_factor, block_M) if trans_E else (block_M, block_K // e_factor)
    b_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A_sparse: T.Tensor(a_sparse_shape, dtype),
        E: T.Tensor(e_shape, metadata_dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor((M, N), accum_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(a_shared_shape, dtype)
            E_shared = T.alloc_shared(e_shared_shape, metadata_dtype)
            B_shared = T.alloc_shared(b_shared_shape, dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_frag)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=stages):
                if trans_A:
                    T.copy(A_sparse[k * (block_K // 2), by * block_M], A_shared)
                else:
                    T.copy(A_sparse[by * block_M, k * (block_K // 2)], A_shared)
                if trans_E:
                    T.copy(E[k * (block_K // e_factor), by * block_M], E_shared)
                else:
                    T.copy(E[by * block_M, k * (block_K // e_factor)], E_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm_sp(A_shared, E_shared, B_shared, C_frag, trans_A, trans_E, trans_B)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def _run_case(M, N, K, block_M, block_N, block_K, threads, trans_A, trans_E, trans_B):
    kernel = tilelang.compile(
        _sparse_matmul(
            M,
            N,
            K,
            block_M,
            block_N,
            block_K,
            threads,
            trans_A=trans_A,
            trans_E=trans_E,
            trans_B=trans_B,
        ),
        out_idx=[3],
        target=TARGET,
    )
    source = kernel.get_kernel_source()
    assert f"tl::gemm_sp_ss<{block_M}, {block_N}, {block_K}" in source
    assert "mma.sp.sync.aligned" not in source

    a_dense_cpu = randn_semi_sparse(M, K, dtype=torch.float16, device="cpu").contiguous()
    a_sparse_cpu, metadata_cpu = torch_compress(a_dense_cpu, meta_dtype=torch.int16)
    b_cpu = torch.randn((N, K) if trans_B else (K, N), dtype=torch.float16)

    if trans_A:
        a_sparse_cpu = a_sparse_cpu.t().contiguous()
    if trans_E:
        metadata_cpu = metadata_cpu.t().contiguous()

    result = kernel(
        a_sparse_cpu.to("musa"),
        metadata_cpu.to("musa"),
        b_cpu.to("musa"),
    )
    logical_b = b_cpu.t().contiguous() if trans_B else b_cpu
    reference = a_dense_cpu.float() @ logical_b.float()

    torch.testing.assert_close(result.cpu(), reference, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_sp_lowers_to_musa_sparse_fma_source():
    target = tvm.target.Target(TARGET)
    with target:
        artifact = tilelang.lower(_sparse_matmul_source(), target=target)
    source = artifact.kernel_source

    assert "tl::gemm_sp_ss<16, 8, 32" in source
    assert "mma.sp.sync.aligned" not in source


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,threads,trans_A,trans_E,trans_B",
    [
        (64, 64, 64, 16, 16, 16, 32, False, False, False),
        (64, 64, 64, 16, 16, 16, 32, True, False, False),
        (64, 64, 64, 16, 16, 16, 32, False, False, True),
        (64, 64, 64, 16, 16, 16, 32, True, False, True),
        (64, 64, 64, 16, 16, 16, 32, False, True, False),
        (64, 64, 64, 16, 16, 16, 32, True, True, False),
        (64, 64, 64, 16, 16, 16, 32, False, True, True),
        (64, 64, 64, 16, 16, 16, 32, True, True, True),
        (32, 64, 64, 16, 16, 32, 32, False, False, False),
        (32, 64, 64, 16, 16, 32, 32, False, False, True),
        (32, 64, 64, 16, 16, 32, 32, False, True, False),
        (32, 64, 64, 16, 16, 32, 32, False, True, True),
        (64, 32, 64, 16, 16, 32, 32, True, False, False),
        (64, 32, 64, 16, 16, 32, 32, True, False, True),
        (64, 32, 64, 16, 16, 32, 32, True, True, False),
        (64, 32, 64, 16, 16, 32, 32, True, True, True),
        (32, 32, 64, 16, 16, 32, 64, False, False, False),
        (32, 32, 64, 16, 16, 32, 128, False, False, False),
    ],
)
def test_gemm_sp_qy2_fp16(M, N, K, block_M, block_N, block_K, threads, trans_A, trans_E, trans_B):
    _run_case(M, N, K, block_M, block_N, block_K, threads, trans_A, trans_E, trans_B)
