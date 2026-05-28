import re

import pytest
import tilelang
from tilelang import tvm as tvm
import tilelang.testing


@pytest.fixture(autouse=True)
def force_mp22_m16_mma(monkeypatch):
    cache_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    monkeypatch.setenv("TILELANG_MUSA_MP22_FORCE_M16_MMA", "1")
    yield
    if cache_enabled:
        tilelang.enable_cache()


THREADS = [32, 64, 128, 256, 512, 1024]
TRANSPOSE_CASES = [
    (False, False, "nn"),
    (False, True, "nt"),
    (True, False, "tn"),
    (True, True, "tt"),
]


def _assert_uses_m16_gemm(kernel_source, op_name, layout_tag):
    if not re.search(rf"tl::{op_name}<[^;]*,\s*16,\s*16,\s*16>", kernel_source):
        raise AssertionError(
            f"{layout_tag} did not lower to MP22 m16n16k16 {op_name}. Expected trailing inst shape template args: 16, 16, 16."
        )


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

    import tilelang.language as T

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


def run_gemm(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
):
    program = matmul(
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
        dtypeAccum,
        num_stages,
        num_threads,
    )
    kernel = tilelang.compile(program, out_idx=[2])
    kernel_source = kernel.get_kernel_source()
    _assert_uses_m16_gemm(kernel_source, "gemm_ss", f"{'t' if trans_A else 'n'}{'t' if trans_B else 'n'}")
    print(kernel_source)
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if dtypeAccum in ("float16", "bfloat16") or in_dtype == "int8":
            return tilelang.testing.matmul_naive(A, B, getattr(torch, dtypeAccum), getattr(torch, out_dtype))
        if in_dtype == "float32":
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 meas
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f16f16f32(trans_A, trans_B, layout_tag, num_threads):
    run_gemm(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


def matmul_sr(
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

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_local = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_local)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_local)
                T.gemm(A_shared, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_sr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
):

    program = matmul_sr(
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
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2], verbose=True)
    kernel_source = kernel.get_kernel_source()
    _assert_uses_m16_gemm(kernel_source, "gemm_sr", f"{'t' if trans_A else 'n'}{'t' if trans_B else 'n'}")
    print(kernel_source)
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        A = A.to(torch.float)
        B = B.to(torch.float)
        C = torch.matmul(A, B)
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f16f16f32_sr(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_sr(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


def matmul_rr(
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

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_local = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_local)
                else:
                    T.copy(A[by * block_M, k * block_K], A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_local)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_local)
                T.gemm(A_local, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
):
    program = matmul_rr(
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
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    kernel_source = kernel.get_kernel_source()
    _assert_uses_m16_gemm(kernel_source, "gemm_rr", f"{'t' if trans_A else 'n'}{'t' if trans_B else 'n'}")
    print(kernel_source)
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f16f16f32_rr(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_rr(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


def matmul_rs(
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

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_local)
                else:
                    T.copy(A[by * block_M, k * block_K], A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_local, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rs(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
):
    program = matmul_rs(
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
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    kernel_source = kernel.get_kernel_source()
    _assert_uses_m16_gemm(kernel_source, "gemm_rs", f"{'t' if trans_A else 'n'}{'t' if trans_B else 'n'}")
    print(kernel_source)
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f16f16f32_rs(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_rs(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


if __name__ == "__main__":
    tilelang.testing.main()
