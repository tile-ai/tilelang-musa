import torch
import pytest
import tilelang
import tilelang.testing
from tilelang.testing import get_tilelang_type, matmul_reference
import tilelang.language as T

tilelang.disable_cache()

WMMA_TOLERANCE_OVERRIDES = {
    torch.float32: (1e-4, 1e-4),
}


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    threads,
    stages,
    trans_A=False,
    trans_B=True,
    dtype="float16",
    accum_dtype="float",
):
    if trans_A and trans_B:

        @T.prim_func
        def gemm(
            A: T.Tensor((K, M), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_fragment = T.alloc_fragment((block_K, block_M), dtype)
                B_fragment = T.alloc_fragment((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=stages):
                    T.copy(A[k * block_K, by * block_M], A_fragment)
                    T.copy(B[bx * block_N, k * block_K], B_fragment)
                    T.gemm(A_fragment, B_fragment, C_local, transpose_A=True, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    if trans_A:

        @T.prim_func
        def gemm(
            A: T.Tensor((K, M), dtype),
            B: T.Tensor((K, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_fragment = T.alloc_fragment((block_K, block_M), dtype)
                B_fragment = T.alloc_fragment((block_K, block_N), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=stages):
                    T.copy(A[k * block_K, by * block_M], A_fragment)
                    T.copy(B[k * block_K, bx * block_N], B_fragment)
                    T.gemm(A_fragment, B_fragment, C_local, transpose_A=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    if trans_B:

        @T.prim_func
        def gemm(
            A: T.Tensor((M, K), dtype),
            B: T.Tensor((N, K), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
                A_fragment = T.alloc_fragment((block_M, block_K), dtype)
                B_fragment = T.alloc_fragment((block_N, block_K), dtype)
                C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=stages):
                    T.copy(A[by * block_M, k * block_K], A_fragment)
                    T.copy(B[bx * block_N, k * block_K], B_fragment)
                    T.gemm(A_fragment, B_fragment, C_local, transpose_B=True)
                T.copy(C_local, C[by * block_M, bx * block_N])

        return gemm

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_fragment = T.alloc_fragment((block_M, block_K), dtype)
            B_fragment = T.alloc_fragment((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=stages):
                T.copy(A[by * block_M, k * block_K], A_fragment)
                T.copy(B[k * block_K, bx * block_N], B_fragment)
                T.gemm(A_fragment, B_fragment, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


size_list = [(256, 256, 256)]
threads = 32
stage_list = [0, 2]
type_block_list = [
    (torch.float16, 16, 16, 32),
    (torch.float16, 16, 16, 16),
    (torch.float16, 16, 8, 16),
    (torch.float16, 16, 8, 8),
    (torch.bfloat16, 16, 16, 32),
    (torch.bfloat16, 16, 16, 16),
    (torch.bfloat16, 16, 8, 16),
    (torch.bfloat16, 16, 8, 8),
    (torch.float32, 16, 16, 16),
    (torch.float32, 16, 8, 8),
    (torch.float32, 16, 8, 4),
    (torch.float8_e4m3fn, 16, 16, 64),
    (torch.float8_e4m3fn, 16, 16, 32),
    (torch.float8_e4m3fn, 16, 16, 16),
    (torch.float8_e4m3fn, 16, 8, 16),
]
test_params = [
    (
        elem_type,
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        threads,
        stages,
        trans_A,
        trans_B,
        disable_ws_tma,
    )
    for (M, N, K) in size_list
    for (elem_type, BLOCK_M, BLOCK_N, BLOCK_K) in type_block_list
    for stages in stage_list
    for (trans_A, trans_B) in [(False, False), (True, True)]
    for disable_ws_tma in [False, True]
    if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
]


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K, threads, stages, trans_A, trans_B, disable_ws_tma",
    test_params,
)
def test_mm_kernel(
    elem_type,
    M,
    N,
    K,
    BLOCK_M,
    BLOCK_N,
    BLOCK_K,
    threads,
    stages,
    trans_A,
    trans_B,
    disable_ws_tma,
):
    device = "musa"
    a_shape = (K, M) if trans_A else (M, K)
    b_shape = (N, K) if trans_B else (K, N)
    A = torch.randn(a_shape, dtype=torch.float16, device=device).to(elem_type)
    B = torch.randn(b_shape, dtype=torch.float16, device=device).to(elem_type)
    program = matmul(
        M,
        N,
        K,
        BLOCK_M,
        BLOCK_N,
        BLOCK_K,
        threads,
        stages,
        trans_A=trans_A,
        trans_B=trans_B,
        dtype=get_tilelang_type(elem_type),
        accum_dtype="float32",
    )
    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_SQMMA: True,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: disable_ws_tma,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: disable_ws_tma,
    }
    kernel = tilelang.compile(
        program,
        out_idx=-1,
        target="musa",
        execution_backend="cython",
        verbose=True,
        pass_configs=pass_configs,
    )
    logical_A = A.T if trans_A else A
    logical_B = B.T if trans_B else B
    rtol, atol = tilelang.testing.get_tolerance(
        elem_type,
        profile="gemm_contract",
        overrides=WMMA_TOLERANCE_OVERRIDES,
    )
    ref_out = matmul_reference(logical_A, logical_B, out_dtype=elem_type)
    C = kernel(A, B)
    torch.testing.assert_close(
        C.to(torch.float32),
        ref_out.to(torch.float32),
        rtol=rtol,
        atol=atol,
    )
    print(
        f"elem_type={elem_type}, M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, "
        f"BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, threads={threads}, "
        f"trans_A={trans_A}, trans_B={trans_B}, "
        f"disable_ws_tma={disable_ws_tma}"
    )
    print("Pass")


def main():
    test_mm_kernel(
        elem_type=torch.float16,
        M=256,
        N=256,
        K=256,
        BLOCK_M=16,
        BLOCK_N=16,
        BLOCK_K=32,
        threads=32,
        stages=0,
        trans_A=False,
        trans_B=False,
        disable_ws_tma=False,
    )


if __name__ == "__main__":
    main()
