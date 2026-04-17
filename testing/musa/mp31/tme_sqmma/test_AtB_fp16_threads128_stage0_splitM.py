import torch
import pytest
import tilelang
import tilelang.testing
from tilelang.testing import get_tilelang_type
import tilelang.language as T

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def matmul(A, B, block_M, block_N, block_K, dtype="float16", accum_dtype="float"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[K, M], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        A_shared = T.alloc_shared((block_K, block_M), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
            T.copy(A[k * block_K, by * block_M], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(A_shared, B_shared, C_local, transpose_A=True)
        T.copy(C_local, C[by * block_M, bx * block_N])

    return C


elem_type_list = [torch.float16]
size_list = [(4096, 4096, 4096)]
block_size_list = [
    #  (128,   16,    16,    128,    0), mutlass split assert
    #  (128,   16,    32,    128,    0), mutlass split assert
    #  (128,   16,    64,    128,    0), mutlass split assert
    #  (256,   16,    16,    128,    0), mutlass split assert
    #  (256,   16,    32,    128,    0), mutlass split assert
    #  (256,   16,    64,    128,    0), mutlass split assert
    #  (512,   16,    16,    128,    0), mutlass split assert
    #  (512,   16,    32,    128,    0), mutlass split assert
    #  (512,   16,    64,    128,    0), mutlass split assert
    (256, 32, 16),
    (256, 32, 32),
    (256, 32, 64),
    (512, 32, 16),
    (512, 32, 32),
    (512, 32, 64),
    (256, 64, 16),
    (256, 64, 32),
    (256, 64, 64),
    (512, 64, 16),
    (512, 64, 32),
    (512, 64, 64),
    (256, 128, 16),
    (256, 128, 32),
    (256, 128, 64),
    # (512, 128, 16, 128, 0), mtcc compile error
    # (512, 128, 32, 128, 0), mtcc compile error
    # (512, 128, 64, 128, 0), mtcc compile error
]
test_params = [
    (elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K)
    for elem_type in elem_type_list
    for (M, N, K) in size_list
    for (BLOCK_M, BLOCK_N, BLOCK_K) in block_size_list
    if M % BLOCK_M == 0 and N % BLOCK_N == 0 and K % BLOCK_K == 0
]


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K", test_params)
def test_mm_kernel(elem_type, M, N, K, BLOCK_M, BLOCK_N, BLOCK_K):
    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=BLOCK_M,
        block_N=BLOCK_N,
        block_K=BLOCK_K,
        dtype=get_tilelang_type(elem_type),
        accum_dtype="float32",
    )
    A = torch.randn((K, M), dtype=elem_type, device="musa")
    B = torch.randn((K, N), dtype=elem_type, device="musa")
    ref_out = A.T @ B
    C = kernel(A, B)
    torch.testing.assert_close(ref_out, C, rtol=1.25e-1, atol=1.25e-1)


def main():
    M, N, K = 4096, 4096, 4096
    BLOCK_M, BLOCK_N, BLOCK_K = 256, 32, 16
    elem_type = torch.float16
    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=BLOCK_M,
        block_N=BLOCK_N,
        block_K=BLOCK_K,
        dtype=get_tilelang_type(elem_type),
        accum_dtype="float32",
    )
    print(kernel.get_kernel_source())
    A = torch.randn((K, M), dtype=elem_type, device="musa")
    B = torch.randn((K, N), dtype=elem_type, device="musa")
    ref_out = A.T @ B
    C = kernel(A, B)
    torch.testing.assert_close(ref_out.to(torch.float16), C.to(torch.float16), rtol=1.25e-1, atol=1.25e-1)
    print(f"elem_type={elem_type}, M={M}, N={N}, K={K}, BLOCK_M={BLOCK_M}, BLOCK_N={BLOCK_N}, BLOCK_K={BLOCK_K}")
    print("Pass")


if __name__ == "__main__":
    main()
