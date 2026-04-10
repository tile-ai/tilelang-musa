import itertools

import pytest
import tilelang
import tilelang.language as T
import torch
from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils.tensor import map_torch_type

tilelang.disable_cache()

BLOCK_CASES = [
    # 16x64
    (16, 64, 16),
    (16, 64, 32),
    (16, 64, 64),
    # 32x32
    (32, 32, 16),
    (32, 32, 32),
    (32, 32, 64),
    (32, 32, 128),
    (32, 32, 256),
    # 32x64
    (32, 64, 16),
    (32, 64, 32),
    (32, 64, 64),
    # 32x128
    (32, 128, 16),
    (32, 128, 32),
    (32, 128, 64),
    # 64x16
    (64, 16, 16),
    (64, 16, 32),
    (64, 16, 64),
    # 64x32
    (64, 32, 16),
    (64, 32, 32),
    (64, 32, 64),
    # 64x64
    (64, 64, 16),
    (64, 64, 32),
    (64, 64, 64),
    (64, 64, 128),
    (64, 64, 256),
    # 64x128
    (64, 128, 16),
    (64, 128, 32),
    (64, 128, 64),
    # 128x32
    (128, 32, 16),
    (128, 32, 32),
    (128, 32, 64),
    # 128x64
    (128, 64, 16),
    (128, 64, 32),
    (128, 64, 64),
    # 128x128
    (128, 128, 16),
    (128, 128, 32),
    (128, 128, 64),
    (128, 128, 128),
    # 128x256
    (128, 256, 64),
    # 256x128
    (256, 128, 64),
    # 256x256
    (256, 256, 64),
]

BASE_DTYPE_CASES = [
    ("float16", "float32"),
    ("bfloat16", "float32"),
]

STAGE_CASES = [1, 3]

# Keep policy/warp coverage compact but representative.
POLICY_CASES = [
    (4, "m"),
    (8, "m"),
    (8, "n"),
    (16, "square"),
    (16, "m"),
    (16, "n"),
]

POLICY_BLOCK_CASES = [
    (128, 128, 64),
    (256, 128, 64),
    (128, 256, 64),
    (256, 256, 64),
]

FLOAT8_BLOCK_CASES = [
    (32, 128, 64),
    (64, 64, 64),
    (128, 128, 64),
    (32, 32, 128),
]


def _problem_size(block_M, block_N, block_K):
    # Keep problem sizes small while guaranteeing multiple tiles per axis.
    M = max(block_M * 2, 128)
    N = max(block_N * 2, 128)
    K = max(block_K * 4, 128)
    return M, N, K


def _case_id(M, N, K, block_M, block_N, block_K, dtype, accum_dtype, num_warp, policy, num_stages):
    return f"M{M}-N{N}-K{K}-bm{block_M}-bn{block_N}-bk{block_K}-{dtype}-{accum_dtype}-warp{num_warp}-{policy}-stage{num_stages}"


def _add_case(cases, seen, *, block_M, block_N, block_K, dtype, accum_dtype, num_warp, policy, num_stages):
    M, N, K = _problem_size(block_M, block_N, block_K)
    key = (M, N, K, block_M, block_N, block_K, dtype, accum_dtype, num_warp, policy, num_stages)
    if key in seen:
        return
    seen.add(key)
    cases.append(
        pytest.param(
            M,
            N,
            K,
            block_M,
            block_N,
            block_K,
            dtype,
            accum_dtype,
            num_warp,
            policy,
            num_stages,
            id=_case_id(
                M,
                N,
                K,
                block_M,
                block_N,
                block_K,
                dtype,
                accum_dtype,
                num_warp,
                policy,
                num_stages,
            ),
        )
    )


def _build_test_cases():
    cases = []
    seen = set()

    # Core matrix: all blocks x base dtypes x stage, with default gemm policy.
    for (block_M, block_N, block_K), (dtype, accum_dtype), num_stages in itertools.product(BLOCK_CASES, BASE_DTYPE_CASES, STAGE_CASES):
        _add_case(
            cases,
            seen,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            dtype=dtype,
            accum_dtype=accum_dtype,
            num_warp=4,
            policy="default",
            num_stages=num_stages,
        )

    # Dedicated policy/warp coverage.
    for (block_M, block_N, block_K), (num_warp, policy), num_stages in itertools.product(POLICY_BLOCK_CASES, POLICY_CASES, STAGE_CASES):
        _add_case(
            cases,
            seen,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            dtype="float16",
            accum_dtype="float32",
            num_warp=num_warp,
            policy=policy,
            num_stages=num_stages,
        )

    # Float8 coverage.
    for (block_M, block_N, block_K), num_stages in itertools.product(FLOAT8_BLOCK_CASES, STAGE_CASES):
        _add_case(
            cases,
            seen,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            dtype="float8_e4m3",
            accum_dtype="float32",
            num_warp=4,
            policy="m",
            num_stages=num_stages,
        )

    return cases


TEST_CASES = _build_test_cases()


@tilelang.jit(target="musa")
def matmul(A, B, block_M, block_N, block_K, dtype, accum_dtype, num_warp, policy, num_stages):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)

    threads = num_warp * 32
    if policy == "default":
        warp_policy = None
    elif policy == "square":
        warp_policy = GemmWarpPolicy.Square
    elif policy == "m":
        warp_policy = GemmWarpPolicy.FullRow
    elif policy == "n":
        warp_policy = GemmWarpPolicy.FullCol
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        a_shared = T.alloc_shared((block_M, block_K), dtype)
        b_shared = T.alloc_shared((block_K, block_N), dtype)
        c_local = T.alloc_fragment((block_M, block_N), accum_dtype)

        T.clear(c_local)
        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
            T.copy(A[by * block_M, k * block_K], a_shared)
            T.copy(B[k * block_K, bx * block_N], b_shared)
            if warp_policy is None:
                T.gemm(a_shared, b_shared, c_local)
            else:
                T.gemm(a_shared, b_shared, c_local, policy=warp_policy)
        T.copy(c_local, C[by * block_M, bx * block_N])

    return C


def _make_inputs(M, N, K, dtype):
    torch_dtype = map_torch_type(dtype)
    if torch_dtype is torch.float8_e4m3fn:
        a = torch.randint(-128, 128, (M, K), device="musa", dtype=torch.int8).to(torch_dtype)
        b = torch.randint(-128, 128, (K, N), device="musa", dtype=torch.int8).to(torch_dtype)
    else:
        a = torch.randn(M, K, device="musa", dtype=torch_dtype)
        b = torch.randn(K, N, device="musa", dtype=torch_dtype)
    return a, b


def _assert_gemm_case(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype,
    accum_dtype,
    num_warp,
    policy,
    num_stages,
):
    kernel = matmul.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
        num_warp=num_warp,
        policy=policy,
        num_stages=num_stages,
    )
    a, b = _make_inputs(M, N, K, dtype)
    c = kernel(a, b)
    ref = a @ b

    if a.dtype is torch.float8_e4m3fn:
        torch.testing.assert_close(c.float(), ref.float(), rtol=1e-2, atol=1e-2)
    else:
        torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)

    return kernel


@pytest.mark.parametrize(
    "M,N,K,block_M,block_N,block_K,dtype,accum_dtype,num_warp,policy,num_stages",
    TEST_CASES,
)
def test_gemm(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    dtype,
    accum_dtype,
    num_warp,
    policy,
    num_stages,
):
    _assert_gemm_case(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        dtype=dtype,
        accum_dtype=accum_dtype,
        num_warp=num_warp,
        policy=policy,
        num_stages=num_stages,
    )


def main():
    kernel_stage1 = _assert_gemm_case(
        M=256,
        N=256,
        K=256,
        block_M=64,
        block_N=64,
        block_K=64,
        dtype="float16",
        accum_dtype="float32",
        num_warp=4,
        policy="default",
        num_stages=1,
    )
    print(kernel_stage1.get_kernel_source())

    kernel_stage3 = _assert_gemm_case(
        M=256,
        N=256,
        K=256,
        block_M=64,
        block_N=64,
        block_K=64,
        dtype="float16",
        accum_dtype="float32",
        num_warp=8,
        policy="m",
        num_stages=3,
    )
    print(kernel_stage3.get_kernel_source())
    print("pass")


if __name__ == "__main__":
    main()
