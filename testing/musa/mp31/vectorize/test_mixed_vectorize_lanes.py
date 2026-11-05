import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing


tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}

MIXED_LANES = [2, 4, 8, 16, 32]


@tilelang.jit(pass_configs=PASS_CONFIGS)
def fp32_to_fp8_vector_cast(A):
    N = T.const("N")
    A: T.Tensor[[N], T.float32]
    B = T.empty((N,), T.float8_e4m3)

    with T.Kernel(1, threads=1):
        for i in T.vectorized(N):
            B[i] = T.Cast(T.float8_e4m3, A[i] * T.float32(1.25))
    return B


@tilelang.jit(pass_configs=PASS_CONFIGS)
def fp8_to_fp32_vector_cast(A):
    N = T.const("N")
    A: T.Tensor[[N], T.float8_e4m3]
    B = T.empty((N,), T.float32)

    with T.Kernel(1, threads=1):
        for i in T.vectorized(N):
            B[i] = T.Cast(T.float32, A[i]) + T.float32(0.5)
    return B


def _assert_no_unsupported_fp32_vectors(source: str):
    assert "float32x16" not in source
    assert "float32x32" not in source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("lanes", MIXED_LANES)
def test_fp32_to_fp8_vectorized_cast_lanes(lanes: int):
    kernel = fp32_to_fp8_vector_cast.compile(N=lanes)

    source = kernel.get_kernel_source()
    _assert_no_unsupported_fp32_vectors(source)
    assert "local_cast" in source

    inp = torch.randn(lanes, device="musa", dtype=torch.float32)
    out = kernel(inp)
    ref = (inp * 1.25).to(torch.float8_e4m3fn)

    rtol, atol = tilelang.testing.get_tolerance(torch.float8_e4m3fn)
    torch.testing.assert_close(out.float(), ref.float(), rtol=rtol, atol=atol)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("lanes", MIXED_LANES)
def test_fp8_to_fp32_vectorized_cast_lanes(lanes: int):
    kernel = fp8_to_fp32_vector_cast.compile(N=lanes)

    source = kernel.get_kernel_source()
    _assert_no_unsupported_fp32_vectors(source)

    inp = torch.randn(lanes, device="musa", dtype=torch.float32).to(torch.float8_e4m3fn)
    out = kernel(inp)
    ref = inp.float() + 0.5

    rtol, atol = tilelang.testing.get_tolerance(torch.float32)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def main():
    lanes = 32
    kernel = fp8_to_fp32_vector_cast.compile(N=lanes)

    source = kernel.get_kernel_source()
    print(source)

    inp = torch.randn(lanes, device="musa", dtype=torch.float32).to(torch.float8_e4m3fn)
    out = kernel(inp)
    ref = inp.float() + 0.5

    rtol, atol = tilelang.testing.get_tolerance(torch.float32)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    main()
