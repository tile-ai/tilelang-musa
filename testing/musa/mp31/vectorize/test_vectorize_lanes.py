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

FP8_LANES = [2, 4, 8, 16, 32]
FP16_LANES = [2, 4, 8, 16]
FP32_LANES = [2, 4, 8, 16]


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def fp8_vector_add(A):
    N = T.const("N")
    A: T.Tensor[[N], T.float8_e4m3]
    B = T.empty((N,), T.float8_e4m3)

    with T.Kernel(1, threads=1):
        for i in T.vectorized(N):
            B[i] = A[i] + A[i]
    return B


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def fp16_vector_add(A):
    N = T.const("N")
    A: T.Tensor[[N], T.float16]
    B = T.empty((N,), T.float16)

    with T.Kernel(1, threads=1):
        for i in T.vectorized(N):
            B[i] = A[i] + A[i]
    return B


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def fp32_vector_add(A):
    N = T.const("N")
    A: T.Tensor[[N], T.float32]
    B = T.empty((N,), T.float32)

    with T.Kernel(1, threads=1):
        for i in T.vectorized(N):
            B[i] = A[i] + A[i]
    return B


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("lanes", FP8_LANES)
def test_fp8_vectorized_add_lanes(lanes: int):
    kernel = fp8_vector_add.compile(N=lanes)

    inp = torch.randn(lanes, device="musa", dtype=torch.float32).to(torch.float8_e4m3fn)
    out = kernel(inp)
    ref = (inp.float() + inp.float()).to(torch.float8_e4m3fn)

    rtol, atol = tilelang.testing.get_tolerance(torch.float8_e4m3fn)
    torch.testing.assert_close(out.float(), ref.float(), rtol=rtol, atol=atol)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("lanes", FP16_LANES)
def test_fp16_vectorized_add_lanes(lanes: int):
    kernel = fp16_vector_add.compile(N=lanes)

    inp = torch.randn(lanes, device="musa", dtype=torch.float16)
    out = kernel(inp)
    ref = inp + inp

    rtol, atol = tilelang.testing.get_tolerance(torch.float16)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("lanes", FP32_LANES)
def test_fp32_vectorized_add_lanes(lanes: int):
    kernel = fp32_vector_add.compile(N=lanes)

    source = kernel.get_kernel_source()
    assert "float32x16" not in source

    inp = torch.randn(lanes, device="musa", dtype=torch.float32)
    out = kernel(inp)
    ref = inp + inp

    rtol, atol = tilelang.testing.get_tolerance(torch.float32)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


def main():
    lanes = 32
    kernel = fp32_vector_add.compile(N=lanes)

    source = kernel.get_kernel_source()
    print(source)

    inp = torch.randn(lanes, device="musa", dtype=torch.float32)
    out = kernel(inp)
    ref = inp + inp

    rtol, atol = tilelang.testing.get_tolerance(torch.float32)
    torch.testing.assert_close(out, ref, rtol=rtol, atol=atol)


if __name__ == "__main__":
    main()
