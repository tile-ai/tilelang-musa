import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.engine.lower import lower
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()

M = 128

_DTYPES = [
    ("float32", T.float32, torch.float32, "tl_f4"),
    ("float16", T.float16, torch.float16, "tl_h4"),
    ("bfloat16", T.bfloat16, torch.bfloat16, "tl_bf4"),
]

_AUTO_VEC_OPS = {
    "add": (lambda a, b: a + b, "add4"),
    "sub": (lambda a, b: a - b, "sub4"),
    "mul": (lambda a, b: a * b, "mul4"),
    "max": (T.max, "max4"),
    "min": (T.min, "min4"),
}

_TORCH_REFS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": lambda a, b: torch.maximum(a, b),
    "min": lambda a, b: torch.minimum(a, b),
    "fma": lambda a, b, c: a * b + c,
}


def _assert_uses_musa_native_x4(src, op_name, dtype_name, native_type):
    assert f"tl::{op_name}" in src
    assert native_type in src
    assert "tl_f2" not in src
    assert "tl_h2" not in src
    assert "tl_bf2" not in src
    assert "uint1" not in src
    assert "uint2" not in src
    assert "float2" not in src
    assert f"{dtype_name}x4" not in src
    assert "__half2" not in src
    assert "__mt_bfloat162" not in src


def _make_auto_vec_binary_kernel(py_op, dtype_tl):
    @T.prim_func
    def main(
        A: T.Tensor((M, 4), dtype=dtype_tl),
        B: T.Tensor((M, 4), dtype=dtype_tl),
        C: T.Tensor((M, 4), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, 4):
                C[i, v] = py_op(A[i, v], B[i, v])

    return main


def _make_auto_vec_fma_kernel(dtype_tl):
    @T.prim_func
    def main(
        A: T.Tensor((M, 4), dtype=dtype_tl),
        B: T.Tensor((M, 4), dtype=dtype_tl),
        C: T.Tensor((M, 4), dtype=dtype_tl),
        D: T.Tensor((M, 4), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, 4):
                D[i, v] = A[i, v] * B[i, v] + C[i, v]

    return main


def _make_auto_vec_fms_kernel(a_scalar=False, b_scalar=False, c_scalar=False):
    @T.prim_func
    def main(
        A: T.Tensor((M, 1 if a_scalar else 4), dtype=T.float32),
        B: T.Tensor((M, 1 if b_scalar else 4), dtype=T.float32),
        C: T.Tensor((M, 1 if c_scalar else 4), dtype=T.float32),
        D: T.Tensor((M, 4), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, 4):
                D[i, v] = (
                    A[i, 0 if a_scalar else v] * B[i, 0 if b_scalar else v]
                    - C[i, 0 if c_scalar else v]
                )

    return main


def _lower_to_musa_source(func, enable_fma_contraction=None):
    pass_configs = {}
    if enable_fma_contraction is not None:
        pass_configs[tilelang.PassConfigKey.TL_ENABLE_FMA_CONTRACTION] = enable_fma_contraction
    with tvm.transform.PassContext(config=pass_configs), tvm.target.Target("musa"):
        artifact = lower(func, target="musa", enable_device_compile=False)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
@pytest.mark.parametrize("op_key", list(_AUTO_VEC_OPS.keys()))
def test_musa_codegen_auto_vec_uses_tilelang_x4_interface(op_key, dtype_name, dtype_tl, torch_dtype, native_type):
    del torch_dtype
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    src = _lower_to_musa_source(_make_auto_vec_binary_kernel(py_op, dtype_tl))
    _assert_uses_musa_native_x4(src, tl_func, dtype_name, native_type)


@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
def test_musa_codegen_auto_vec_fma_contraction_disabled_by_default(dtype_name, dtype_tl, torch_dtype, native_type):
    del torch_dtype
    src = _lower_to_musa_source(_make_auto_vec_fma_kernel(dtype_tl))
    _assert_uses_musa_native_x4(src, "mul4", dtype_name, native_type)
    _assert_uses_musa_native_x4(src, "add4", dtype_name, native_type)
    assert "tl::fma4" not in src


@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
def test_musa_codegen_auto_vec_fma_contraction_enabled(dtype_name, dtype_tl, torch_dtype, native_type):
    del torch_dtype
    src = _lower_to_musa_source(_make_auto_vec_fma_kernel(dtype_tl), enable_fma_contraction=True)
    _assert_uses_musa_native_x4(src, "fma4", dtype_name, native_type)


def test_musa_codegen_auto_vec_f32x4_fms_contraction_disabled_by_default():
    src = _lower_to_musa_source(_make_auto_vec_fms_kernel())
    _assert_uses_musa_native_x4(src, "mul4", "float32", "tl_f4")
    _assert_uses_musa_native_x4(src, "sub4", "float32", "tl_f4")
    assert "tl::fma4" not in src


def test_musa_codegen_auto_vec_f32x4_fms_contraction_enabled():
    src = _lower_to_musa_source(_make_auto_vec_fms_kernel(), enable_fma_contraction=True)
    _assert_uses_musa_native_x4(src, "fma4", "float32", "tl_f4")
    assert "tl::mul4" not in src
    assert "tl::sub4" not in src


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
@pytest.mark.parametrize("op_key", list(_AUTO_VEC_OPS.keys()))
def test_musa_correctness_auto_vec_x4(op_key, dtype_name, dtype_tl, torch_dtype, native_type):
    del dtype_name, native_type
    py_op, _ = _AUTO_VEC_OPS[op_key]
    func = _make_auto_vec_binary_kernel(py_op, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[2], target="musa")

    a = torch.randn((M, 4), device="musa", dtype=torch_dtype)
    b = torch.randn((M, 4), device="musa", dtype=torch_dtype)
    c = kernel(a, b)
    ref = _TORCH_REFS[op_key](a, b)
    torch.testing.assert_close(c, ref)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
def test_musa_correctness_auto_vec_fma_x4(dtype_name, dtype_tl, torch_dtype, native_type):
    del dtype_name, native_type
    func = _make_auto_vec_fma_kernel(dtype_tl)
    kernel = tilelang.compile(func, out_idx=[3], target="musa")

    a = torch.randn((M, 4), device="musa", dtype=torch_dtype)
    b = torch.randn((M, 4), device="musa", dtype=torch_dtype)
    c = torch.randn((M, 4), device="musa", dtype=torch_dtype)
    d = kernel(a, b, c)
    ref = _TORCH_REFS["fma"](a, b, c)
    if torch_dtype == torch.float32:
        torch.testing.assert_close(d, ref)
    else:
        torch.testing.assert_close(d, ref, atol=1e-2, rtol=1e-1)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_musa_correctness_auto_vec_f32x4_fms():
    kernel = tilelang.compile(_make_auto_vec_fms_kernel(), out_idx=[3], target="musa")

    a = torch.randn((M, 4), device="musa", dtype=torch.float32)
    b = torch.randn((M, 4), device="musa", dtype=torch.float32)
    c = torch.randn((M, 4), device="musa", dtype=torch.float32)
    d = kernel(a, b, c)
    ref = a * b - c
    torch.testing.assert_close(d, ref)


if __name__ == "__main__":
    tilelang.testing.main()
