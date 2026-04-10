import pytest
import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_SQMMA: True,
}


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def matmul(A, B, block_M, block_N, block_K, dtype="float16", accum_dtype="float32"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local)
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


def get_tilelang_type(elem_type):
    type_map = {
        torch.float16: "float16",
        torch.bfloat16: "bfloat16",
        torch.float8_e4m3fn: "float8_e4m3",
    }
    return type_map[elem_type]


def get_rtol(elem_type):
    rtol_map = {
        torch.float16: 1e-3,
        torch.bfloat16: 7.9e-3,
        torch.float8_e4m3fn: 1.25e-1,
    }
    return rtol_map[elem_type]


def get_atol(elem_type):
    atol_map = {
        torch.float16: 1e-3,
        torch.bfloat16: 1e-3,
        torch.float8_e4m3fn: 1.25e-1,
    }
    return atol_map[elem_type]


def _assert_case(elem_type, M, N, K, block_M, block_N, block_K):
    a = torch.randn((M, K), dtype=torch.float16, device="musa").to(elem_type)
    b = torch.randn((K, N), dtype=torch.float16, device="musa").to(elem_type)
    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=get_tilelang_type(elem_type),
        accum_dtype="float32",
    )

    out = kernel(a, b)
    ref = a @ b
    torch.testing.assert_close(
        ref.to(torch.float32),
        out.to(torch.float32),
        rtol=get_rtol(elem_type),
        atol=get_atol(elem_type),
    )
    return kernel


elem_type_list = [torch.float16, torch.bfloat16, torch.float8_e4m3fn]
size_list = [(1024, 1024, 1024)]
block_size_list = [(128, 128, 64), (32, 32, 32), (16, 16, 16), (8, 8, 8), (4, 4, 4), (2, 2, 2)]
test_params = [
    (elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    for elem_type in elem_type_list
    for (M, N, K) in size_list
    for (BLOCK_M, BLOCK_N, BLOCK_K) in block_size_list
]


@pytest.mark.parametrize("elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K", test_params)
def test_fma_tma(elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    _assert_case(
        elem_type=elem_type,
        M=M,
        N=N,
        K=K,
        block_M=BLOCK_M,
        block_N=BLOCK_N,
        block_K=BLOCK_K,
    )


def main():
    kernel = _assert_case(
        elem_type=torch.float16,
        M=256,
        N=256,
        K=256,
        block_M=128,
        block_N=128,
        block_K=64,
    )
    print(kernel.get_kernel_source())
    print("pass")


if __name__ == "__main__":
    main()
