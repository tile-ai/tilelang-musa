import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main

# Each sequence returns to its first shape to check reuse after shape changes.
REUSE_SHAPES = {
    "dynamic_m": [(1024, 1024, 768), (128, 1024, 768), (257, 1024, 768), (1024, 1024, 768)],
    "dynamic_mn": [(1024, 1024, 768), (128, 256, 768), (257, 513, 768), (1024, 1024, 768)],
    "dynamic_mnk": [(1024, 1024, 768), (128, 256, 32), (257, 513, 65), (1024, 1024, 768)],
}


def compile_dynamic_matmul(case, target=None):
    m = T.dynamic("m")
    n = T.dynamic("n") if case != "dynamic_m" else 1024
    k = T.dynamic("k") if case == "dynamic_mnk" else 768
    program = matmul(m, n, k, 128, 256, 32, False, False, T.float16, T.float16, T.float32, 2, 128)
    return tilelang.compile(
        program,
        target=target,
        execution_backend="tvm_ffi",
        pass_configs={
            # Preserve the original MP22 dynamic-GEMM MCC workaround.
            tilelang.PassConfigKey.TL_DEVICE_COMPILE_FLAGS: ["-fslp-vectorize"],
        },
    )


@pytest.mark.parametrize("case", list(REUSE_SHAPES))
@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_tvm_ffi_dynamic_shape(case):
    kernel = compile_dynamic_matmul(case)
    # Compile exactly once; every invocation below uses this same FFI executable.
    for m, n, k in REUSE_SHAPES[case]:
        a = torch.randn(m, k, dtype=torch.float16, device="musa")
        b = torch.randn(k, n, dtype=torch.float16, device="musa")
        c = torch.full((m, n), float("nan"), dtype=torch.float16, device="musa")
        kernel(a, b, c)
        expected = (a.float() @ b.float()).to(torch.float16)
        torch.testing.assert_close(c, expected, atol=1e-2, rtol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
