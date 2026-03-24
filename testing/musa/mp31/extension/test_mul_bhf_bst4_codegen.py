import pytest
import torch
import tilelang
import tilelang.language as T

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
}


def require_musa():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA is not available")


@tilelang.jit(target="musa", out_idx=[2], pass_configs=PASS_CONFIGS)
def mul_bhf_bst4_broadcast16_kernel():
    """Vectorized-16 case lowered by chunking into four bst4 groups."""

    @T.prim_func
    def main(
        A: T.Tensor((16,), "float16"),
        Scale: T.Tensor((1,), "float32"),
        C: T.Tensor((16,), "bfloat16"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(16):
                C[i] = T.Cast("bfloat16", A[i] * Scale[0])

    return main


@tilelang.jit(target="musa", out_idx=[2], pass_configs=PASS_CONFIGS)
def mul_bhf_bst4_kernel():
    """Minimal PH1 burst case for v4f16 * v4f32 -> v4bf16.

    The scalar loop is intentional: with `TL_ENABLE_MUSA_BURST`, TileLang gets a
    chance to vectorize the 4 contiguous lanes and potentially lower the pattern
    to `__musa_mul_bhf_bst4_vv` instead of emitting separate cast helpers.
    """

    @T.prim_func
    def main(
        A: T.Tensor((4,), "float16"),
        B: T.Tensor((4,), "float32"),
        C: T.Tensor((4,), "bfloat16"),
    ):
        with T.Kernel(1, threads=1):
            for i in T.Parallel(4):
                C[i] = T.Cast("bfloat16", A[i] * B[i])

    return main


def collect_codegen_markers(source: str) -> dict[str, bool]:
    return {
        "__musa_mul_bhf_bst4_vv": "__musa_mul_bhf_bst4_vv" in source,
        "tl::mul_half_float_to_bfloat16_x4": "tl::mul_half_float_to_bfloat16_x4" in source,
        "__musa_mul_fhf_bst4_vv": "__musa_mul_fhf_bst4_vv" in source,
        "tl::cvt_half_to_float_x4": "tl::cvt_half_to_float_x4" in source,
        "tl::cvt_float_to_bfloat16_x4": "tl::cvt_float_to_bfloat16_x4" in source,
    }


def relevant_codegen_lines(source: str) -> list[str]:
    needles = (
        "__musa_mul_",
        "tl::mul_half_float_to_bfloat16_x4",
        "tl::cvt_half_to_float_x4",
        "tl::cvt_float_to_bfloat16_x4",
    )
    return [line for line in source.splitlines() if any(needle in line for needle in needles)]


def test_mul_bhf_bst4_numerical():
    require_musa()

    kernel = mul_bhf_bst4_kernel()

    a = torch.randn(4, dtype=torch.float16, device="musa")
    b = torch.randn(4, dtype=torch.float32, device="musa")
    out = kernel(a, b)
    expected = (a.float() * b).to(torch.bfloat16)

    torch.testing.assert_close(out.float(), expected.float(), rtol=0.0, atol=0.0)


def test_mul_bhf_bst4_codegen_report():
    require_musa()

    source = mul_bhf_bst4_kernel().get_kernel_source()
    markers = collect_codegen_markers(source)

    print(markers)
    for line in relevant_codegen_lines(source):
        print(line)

    assert markers["tl::mul_half_float_to_bfloat16_x4"]


def test_mul_bhf_bst4_broadcast16_codegen_report():
    require_musa()

    source = mul_bhf_bst4_broadcast16_kernel().get_kernel_source()
    markers = collect_codegen_markers(source)

    print(markers)
    for line in relevant_codegen_lines(source):
        print(line)

    assert markers["tl::mul_half_float_to_bfloat16_x4"]


if __name__ == "__main__":
    source = mul_bhf_bst4_kernel().get_kernel_source()
    print(collect_codegen_markers(source))
    print(source)
    source = mul_bhf_bst4_broadcast16_kernel().get_kernel_source()
    print(collect_codegen_markers(source))
    print(source)
