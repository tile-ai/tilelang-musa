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
    ("float32", T.float32, torch.float32, "tl_f2"),
    ("float16", T.float16, torch.float16, "tl_h2"),
    ("bfloat16", T.bfloat16, torch.bfloat16, "tl_bf2"),
]

_BINARY_OPS = [
    ("add2", T.add2),
    ("sub2", T.sub2),
    ("mul2", T.mul2),
    ("max2", T.max2),
    ("min2", T.min2),
]

_AUTO_VEC_OPS = {
    "add": (lambda a, b: a + b, "add2"),
    "sub": (lambda a, b: a - b, "sub2"),
    "mul": (lambda a, b: a * b, "mul2"),
    "max": (T.max, "max2"),
    "min": (T.min, "min2"),
}

_TORCH_REFS = {
    "add2": lambda a, b: a + b,
    "sub2": lambda a, b: a - b,
    "mul2": lambda a, b: a * b,
    "max2": lambda a, b: torch.maximum(a, b),
    "min2": lambda a, b: torch.minimum(a, b),
    "fma2": lambda a, b, c: a * b + c,
    "abs2": lambda a: torch.abs(a),
}

def _assert_uses_musa_native_x2(src, op_name, dtype_name, native_type):
    assert f"tl::{op_name}" in src
    assert native_type in src
    assert "uint1" not in src
    assert "uint2" not in src
    assert "float2" not in src
    assert f"{dtype_name}x2" not in src
    if dtype_name == "float16":
        assert "tl_h2" in src
    elif dtype_name == "bfloat16":
        assert "tl_bf2" in src
    else:
        assert "tl_f2" in src
    assert "__half2" not in src
    assert "__mt_bfloat162" not in src


def _make_binary_kernel(op_func, dtype_tl):
    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=dtype_tl),
        B: T.Tensor((M * 2,), dtype=dtype_tl),
        C: T.Tensor((M * 2,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            C[idx] = op_func(A[idx], B[idx])

    return main


def _make_ternary_kernel(op_func, dtype_tl):
    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=dtype_tl),
        B: T.Tensor((M * 2,), dtype=dtype_tl),
        C: T.Tensor((M * 2,), dtype=dtype_tl),
        D: T.Tensor((M * 2,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            D[idx] = op_func(A[idx], B[idx], C[idx])

    return main


def _make_unary_kernel(op_func, dtype_tl):
    @T.prim_func
    def main(
        A: T.Tensor((M * 2,), dtype=dtype_tl),
        C: T.Tensor((M * 2,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * 2, 1, 2)
            C[idx] = op_func(A[idx])

    return main


def _make_auto_vec_binary_kernel(py_op, dtype_tl):
    @T.prim_func
    def main(
        A: T.Tensor((M, 2), dtype=dtype_tl),
        B: T.Tensor((M, 2), dtype=dtype_tl),
        C: T.Tensor((M, 2), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, 2):
                C[i, v] = py_op(A[i, v], B[i, v])

    return main


def _make_auto_vec_fma_kernel(dtype_tl):
    @T.prim_func
    def main(
        A: T.Tensor((M, 2), dtype=dtype_tl),
        B: T.Tensor((M, 2), dtype=dtype_tl),
        C: T.Tensor((M, 2), dtype=dtype_tl),
        D: T.Tensor((M, 2), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            for i, v in T.Parallel(M, 2):
                D[i, v] = A[i, v] * B[i, v] + C[i, v]

    return main


def _lower_to_musa_source(func):
    with tvm.transform.PassContext(), tvm.target.Target("musa"):
        artifact = lower(func, target="musa", enable_device_compile=False)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
@pytest.mark.parametrize("op_name,op_func", _BINARY_OPS, ids=[name for name, _ in _BINARY_OPS])
def test_musa_codegen_binary_uses_native_vector_type(
    op_name, op_func, dtype_name, dtype_tl, torch_dtype, native_type
):
    del torch_dtype
    src = _lower_to_musa_source(_make_binary_kernel(op_func, dtype_tl))
    _assert_uses_musa_native_x2(src, op_name, dtype_name, native_type)


@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
def test_musa_codegen_fma2_uses_native_vector_type(dtype_name, dtype_tl, torch_dtype, native_type):
    del torch_dtype
    src = _lower_to_musa_source(_make_ternary_kernel(T.fma2, dtype_tl))
    _assert_uses_musa_native_x2(src, "fma2", dtype_name, native_type)


@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
def test_musa_codegen_abs2_uses_native_vector_type(dtype_name, dtype_tl, torch_dtype, native_type):
    del torch_dtype
    src = _lower_to_musa_source(_make_unary_kernel(T.abs2, dtype_tl))
    _assert_uses_musa_native_x2(src, "abs2", dtype_name, native_type)


@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
@pytest.mark.parametrize("op_key", list(_AUTO_VEC_OPS.keys()))
def test_musa_codegen_auto_vec_uses_tilelang_interface(
    op_key, dtype_name, dtype_tl, torch_dtype, native_type
):
    del torch_dtype
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    src = _lower_to_musa_source(_make_auto_vec_binary_kernel(py_op, dtype_tl))
    _assert_uses_musa_native_x2(src, tl_func, dtype_name, native_type)


@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
def test_musa_codegen_auto_vec_fma_uses_tilelang_interface(
    dtype_name, dtype_tl, torch_dtype, native_type
):
    del torch_dtype
    src = _lower_to_musa_source(_make_auto_vec_fma_kernel(dtype_tl))
    _assert_uses_musa_native_x2(src, "fma2", dtype_name, native_type)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
@pytest.mark.parametrize("op_name,op_func", _BINARY_OPS, ids=[name for name, _ in _BINARY_OPS])
def test_musa_correctness_binary(op_name, op_func, dtype_name, dtype_tl, torch_dtype, native_type):
    del dtype_name, native_type
    func = _make_binary_kernel(op_func, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[2], target="musa")

    a = torch.randn(M * 2, device="musa", dtype=torch_dtype)
    b = torch.randn(M * 2, device="musa", dtype=torch_dtype)
    c = kernel(a, b)
    ref = _TORCH_REFS[op_name](a, b)
    torch.testing.assert_close(c, ref)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
def test_musa_correctness_auto_vec_fma(dtype_name, dtype_tl, torch_dtype, native_type):
    del dtype_name, native_type
    func = _make_auto_vec_fma_kernel(dtype_tl)
    kernel = tilelang.compile(func, out_idx=[3], target="musa")

    a = torch.randn((M, 2), device="musa", dtype=torch_dtype)
    b = torch.randn((M, 2), device="musa", dtype=torch_dtype)
    c = torch.randn((M, 2), device="musa", dtype=torch_dtype)
    d = kernel(a, b, c)
    ref = a * b + c
    if torch_dtype == torch.float32:
        torch.testing.assert_close(d, ref)
    else:
        torch.testing.assert_close(d, ref, atol=1e-2, rtol=1e-1)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
def test_musa_correctness_fma2(dtype_name, dtype_tl, torch_dtype, native_type):
    del dtype_name, native_type
    func = _make_ternary_kernel(T.fma2, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[3], target="musa")

    a = torch.randn(M * 2, device="musa", dtype=torch_dtype)
    b = torch.randn(M * 2, device="musa", dtype=torch_dtype)
    c = torch.randn(M * 2, device="musa", dtype=torch_dtype)
    d = kernel(a, b, c)
    ref = _TORCH_REFS["fma2"](a, b, c)
    if torch_dtype == torch.float32:
        torch.testing.assert_close(d, ref)
    else:
        torch.testing.assert_close(d, ref, atol=1e-2, rtol=1e-1)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
def test_musa_correctness_abs2(dtype_name, dtype_tl, torch_dtype, native_type):
    del dtype_name, native_type
    func = _make_unary_kernel(T.abs2, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[1], target="musa")

    a = torch.randn(M * 2, device="musa", dtype=torch_dtype)
    c = kernel(a)
    ref = _TORCH_REFS["abs2"](a)
    torch.testing.assert_close(c, ref)


if __name__ == "__main__":
    tilelang.testing.main()
