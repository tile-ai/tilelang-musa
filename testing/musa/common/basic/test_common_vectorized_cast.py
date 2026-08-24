import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def _make_vector_cast_kernel(src_dtype, dst_dtype, lanes):
    @T.prim_func
    def kernel(
        src: T.Tensor((lanes,), src_dtype),
        dst: T.Tensor((lanes,), dst_dtype),
    ):
        with T.Kernel(1, threads=1):
            dst[T.Ramp(0, 1, lanes)] = T.cast(src[T.Ramp(0, 1, lanes)], dst_dtype.with_lanes(lanes))

    return kernel


def _make_e8m0_roundtrip_kernel(lanes):
    @T.prim_func
    def kernel(
        src: T.Tensor((lanes,), T.float32),
        dst: T.Tensor((lanes,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            value = T.cast(src[T.Ramp(0, 1, lanes)], T.float8_e8m0fnu.with_lanes(lanes))
            dst[T.Ramp(0, 1, lanes)] = T.cast(value, T.float32.with_lanes(lanes))

    return kernel


def _cvt_type_name(dtype):
    return {
        T.float16: "half",
        T.bfloat16: "bfloat16",
        T.float32: "float",
        T.float8_e4m3fn: "fp8e4m3",
        T.float8_e5m2: "fp8e5m2",
        T.float8_e8m0fnu: "fp8e8m0",
    }[dtype]


def _torch_dtype(dtype):
    return {
        T.float16: torch.float16,
        T.bfloat16: torch.bfloat16,
        T.float32: torch.float32,
        T.float8_e4m3fn: torch.float8_e4m3fn,
        T.float8_e5m2: torch.float8_e5m2,
        T.float8_e8m0fnu: torch.float8_e8m0fnu,
    }[dtype]


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (T.float16, T.float32),
        (T.float32, T.float16),
        (T.bfloat16, T.float32),
        (T.float32, T.bfloat16),
        (T.float32, T.float8_e4m3fn),
        (T.float8_e4m3fn, T.float32),
        (T.float16, T.float8_e4m3fn),
        (T.float8_e4m3fn, T.float16),
        (T.float32, T.float8_e5m2),
        (T.float8_e5m2, T.float32),
        (T.float16, T.float8_e5m2),
        (T.float8_e5m2, T.float16),
        (T.float32, T.float8_e8m0fnu),
        (T.float8_e8m0fnu, T.float32),
    ],
)
@pytest.mark.parametrize("lanes", [2, 4])
@tilelang.testing.requires_musa
def test_musa_common_vectorized_cast_codegen(src_dtype, dst_dtype, lanes):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _make_vector_cast_kernel(src_dtype, dst_dtype, lanes),
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert source is not None
    assert "#include <tl_templates/musa/common/cvt.h>" in source
    helper = f"tl::cvt_{_cvt_type_name(src_dtype)}_to_{_cvt_type_name(dst_dtype)}_x{lanes}"
    assert helper in source

    values = torch.tensor([0.25, 0.5, 1.0, 2.0], dtype=torch.float32)[:lanes]
    if T.float8_e8m0fnu in (src_dtype, dst_dtype):
        kernel = tilelang.compile(
            _make_e8m0_roundtrip_kernel(lanes),
            out_idx=[1],
            target={"kind": "musa"},
            execution_backend="tvm_ffi",
        )
        out = kernel(values.to("musa"))
        torch.musa.synchronize()
        torch.testing.assert_close(out.cpu(), values)
        return

    kernel = tilelang.compile(
        _make_vector_cast_kernel(src_dtype, dst_dtype, lanes),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    src = values.to(_torch_dtype(src_dtype))
    out = kernel(src.to("musa"))
    torch.musa.synchronize()

    expected = src.to(_torch_dtype(dst_dtype)).float()
    torch.testing.assert_close(out.float().cpu(), expected)


@pytest.mark.parametrize(
    "src_dtype,dst_dtype",
    [
        (T.float16, T.float32),
        (T.float32, T.float16),
        (T.bfloat16, T.float32),
        (T.float32, T.bfloat16),
    ],
)
@pytest.mark.parametrize("lanes", [6, 8])
@tilelang.testing.requires_musa
def test_musa_common_wide_float_cast(src_dtype, dst_dtype, lanes):
    kernel = tilelang.compile(
        _make_vector_cast_kernel(src_dtype, dst_dtype, lanes),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    source = kernel.get_kernel_source()
    helper = f"tl::cvt_{_cvt_type_name(src_dtype)}_to_{_cvt_type_name(dst_dtype)}_x{lanes}("
    assert source.count(helper) == 1
    # Distinct lanes include halfway cases for both FP16 and BF16 rounding.
    batches = [
        [0.0, -0.0, 1.00048828125, 1.00146484375, 1.00390625, 1.01171875, -2.125, 123.75],
        [float("inf"), -float("inf"), float("nan"), 65504.0, 2**-24, -(2**-24), 2**-14, 2**-126],
    ]
    for values in batches:
        src = torch.tensor(values[:lanes], dtype=torch.float32).to(_torch_dtype(src_dtype))
        out = kernel(src.to("musa"))
        torch.musa.synchronize()
        expected = src.to(_torch_dtype(dst_dtype)).float()
        actual = out.float().cpu()
        torch.testing.assert_close(actual, expected, atol=0, rtol=0, equal_nan=True)
        zeros = expected == 0
        torch.testing.assert_close(torch.signbit(actual[zeros]), torch.signbit(expected[zeros]))


@pytest.mark.parametrize("src_dtype,dst_dtype", [(T.float32, T.float8_e4m3fn), (T.float8_e4m3fn, T.float32)])
@tilelang.testing.requires_musa
def test_musa_common_wide_fp8_cast_rejected(src_dtype, dst_dtype):
    with pytest.raises(tvm.error.InternalError, match="MUSA vectorized cast .*supports up to 4 lanes"):
        tilelang.lower(
            _make_vector_cast_kernel(src_dtype, dst_dtype, 8),
            target={"kind": "musa"},
            enable_device_compile=False,
        )


if __name__ == "__main__":
    tilelang.testing.main()
