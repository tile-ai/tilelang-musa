import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import torch

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
}


def require_musa():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA is not available")


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def kernel_with_existing_vectorized_exp2_loop(A):
    A: T.Tensor[[32], T.float32]
    B = T.empty((32,), "float32")

    with T.Kernel(1, threads=1):
        for i in T.serial(4):
            for j in T.vectorized(8):
                B[i * 8 + j] = T.exp2(A[i * 8 + j])

    return B


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def kernel_with_local_exp2_recurrence(A):
    A: T.Tensor[[2], T.float32]
    Out = T.empty((1,), "float32")

    with T.Kernel(1, threads=1):
        s_lse = T.alloc_local((2,), "float32")
        max_lse = T.alloc_local((1,), "float32")
        sum_exp = T.alloc_local((1,), "float32")

        s_lse[0] = A[0]
        s_lse[1] = A[1]

        max_lse[0] = T.max(s_lse[0], s_lse[1])
        sum_exp[0] = T.float32(0)

        for i in T.unroll(2):
            sum_exp[0] += T.exp2(s_lse[i] - max_lse[0])

        Out[0] = sum_exp[0]

    return Out


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def kernel_with_fixed_index_exp2_overwrite(A):
    A: T.Tensor[[2], T.float32]
    Out = T.empty((1,), "float32")

    with T.Kernel(1, threads=1):
        s_lse = T.alloc_local((2,), "float32")
        max_lse = T.alloc_local((1,), "float32")
        last_exp = T.alloc_local((1,), "float32")

        s_lse[0] = A[0]
        s_lse[1] = A[1]

        max_lse[0] = T.max(s_lse[0], s_lse[1])
        last_exp[0] = T.float32(0)

        for i in T.unroll(2):
            last_exp[0] = T.exp2(s_lse[i] - max_lse[0])

        Out[0] = last_exp[0]

    return Out


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def kernel_with_broadcast_load_elementwise_store(A):
    A: T.Tensor[[2], T.float32]
    Out = T.empty((2,), "float32")

    with T.Kernel(1, threads=1):
        for i in T.unroll(2):
            Out[i] = T.exp2(A[i] - A[0])

    return Out


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def kernel_with_elementwise_cast(A):
    A: T.Tensor[[2], T.float32]
    Out = T.empty((2,), "bfloat16")

    with T.Kernel(1, threads=1):
        for i in T.unroll(2):
            Out[i] = T.Cast("bfloat16", A[i])

    return Out


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_existing_vectorized_exp2_loop_end_to_end():
    require_musa()

    kernel = kernel_with_existing_vectorized_exp2_loop.compile()
    inp = torch.linspace(-2.0, 2.0, 32, device="musa", dtype=torch.float32)
    out = kernel(inp)

    expected = torch.exp2(inp)
    torch.testing.assert_close(out, expected, rtol=1e-5, atol=1e-6)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_local_exp2_recurrence_is_not_late_vectorized():
    source = kernel_with_local_exp2_recurrence.compile().get_kernel_source()
    normalized = source.replace(" ", "")

    assert "float sum_exp[1];" in source
    assert "tl::vec_exp2_f2" not in source
    assert "make_float2(sum_exp[0],sum_exp[0])" not in normalized


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_fixed_index_exp2_overwrite_is_not_late_vectorized():
    source = kernel_with_fixed_index_exp2_overwrite.compile().get_kernel_source()
    normalized = source.replace(" ", "")

    assert "float last_exp[1];" in source
    assert "tl::vec_exp2_f2" not in source
    assert "make_float2(last_exp[0],last_exp[0])" not in normalized


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_broadcast_load_elementwise_store_is_still_late_vectorized():
    source = kernel_with_broadcast_load_elementwise_store.compile().get_kernel_source()
    normalized = source.replace(" ", "")

    assert "tl::vec_exp2_f2" in source
    assert "make_float2(A[0],A[0])" in normalized


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_elementwise_cast_is_still_late_vectorized():
    source = kernel_with_elementwise_cast.compile().get_kernel_source()
    assert "tl::cvt_float_to_bfloat16_x2" in source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_local_exp2_recurrence_numerical():
    require_musa()

    inp = torch.tensor([1.5, -0.5], device="musa", dtype=torch.float32)
    out = kernel_with_local_exp2_recurrence.compile()(inp)

    expected = torch.sum(torch.exp2(inp - torch.max(inp))).reshape(1)
    torch.testing.assert_close(out, expected, rtol=1e-6, atol=1e-6)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def main():
    require_musa()

    source_0 = kernel_with_existing_vectorized_exp2_loop.compile().get_kernel_source()
    source_1 = kernel_with_local_exp2_recurrence.compile().get_kernel_source()
    print(source_0)
    print(source_1)

    inp = torch.linspace(-2.0, 2.0, 32, device="musa", dtype=torch.float32)
    out = kernel_with_existing_vectorized_exp2_loop.compile()(inp)
    torch.testing.assert_close(out, torch.exp2(inp), rtol=1e-5, atol=1e-6)
    print("pass")


if __name__ == "__main__":
    main()
