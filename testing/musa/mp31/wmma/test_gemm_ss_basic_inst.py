import torch
import pytest
import tilelang
import tilelang.testing
import tilelang.language as T

tilelang.disable_cache()


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

    a_shape = (K, M) if trans_A else (M, K)
    b_shape = (N, K) if trans_B else (K, N)
    a_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    b_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def gemm(
        A: T.Tensor(a_shape, dtype),
        B: T.Tensor(b_shape, dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(a_shared_shape, dtype)
            B_shared = T.alloc_shared(b_shared_shape, dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_A=trans_A, transpose_B=trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def get_tilelang_type(elem_type):
    type_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float32: "float32",
        torch.float8_e4m3fn: "float8_e4m3",
    }
    return type_map.get(elem_type)


size_list = [(4096, 4096, 4096)]
type_block_threads_stages_list = [
    (torch.float16, 16, 16, 32, 32, 0),
    (torch.float16, 16, 16, 16, 32, 0),
    (torch.float16, 16, 8, 16, 32, 0),
    (torch.float16, 16, 8, 8, 32, 0),
    (torch.bfloat16, 16, 16, 32, 32, 0),
    (torch.bfloat16, 16, 16, 16, 32, 0),
    (torch.bfloat16, 16, 8, 16, 32, 0),
    (torch.bfloat16, 16, 8, 8, 32, 0),
    (torch.float32, 16, 16, 16, 32, 0),
    (torch.float32, 16, 8, 8, 32, 0),
    (torch.float32, 16, 8, 4, 32, 0),
    (torch.float8_e4m3fn, 16, 16, 64, 32, 0),
    (torch.float8_e4m3fn, 16, 16, 32, 32, 0),
    (torch.float8_e4m3fn, 16, 16, 16, 32, 0),
    (torch.float8_e4m3fn, 16, 8, 16, 32, 0),
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
    for (elem_type, BLOCK_M, BLOCK_N, BLOCK_K, threads, stages) in type_block_threads_stages_list
    for (trans_A, trans_B) in [(False, False), (False, True), (True, False), (True, True)]
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
    print(kernel.get_kernel_source())
    logical_A = A.T if trans_A else A
    logical_B = B.T if trans_B else B
    ref_out = logical_A @ logical_B
    C = kernel(A, B)
    torch.testing.assert_close(ref_out.to(torch.float16), C.to(torch.float16), rtol=1.25e-1, atol=1.25e-1)
    print(
        f"elem_type={elem_type}, M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, "
        f"BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}, threads={threads}, "
        f"trans_A={trans_A}, trans_B={trans_B}, "
        f"disable_ws_tma={disable_ws_tma}"
    )
    print("Pass")
