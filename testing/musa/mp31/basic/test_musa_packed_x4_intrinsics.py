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


def _lower_to_musa_source(func):
    with tvm.transform.PassContext(), tvm.target.Target("musa"):
        artifact = lower(func, target="musa", enable_device_compile=False)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
@pytest.mark.parametrize("op_key", list(_AUTO_VEC_OPS.keys()))
def test_musa_codegen_auto_vec_uses_tilelang_x4_interface(
    op_key, dtype_name, dtype_tl, torch_dtype, native_type
):
    del torch_dtype
    py_op, tl_func = _AUTO_VEC_OPS[op_key]
    src = _lower_to_musa_source(_make_auto_vec_binary_kernel(py_op, dtype_tl))
    _assert_uses_musa_native_x4(src, tl_func, dtype_name, native_type)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype_name,dtype_tl,torch_dtype,native_type", _DTYPES)
@pytest.mark.parametrize("op_key", list(_AUTO_VEC_OPS.keys()))
def test_musa_correctness_auto_vec_x4(
    op_key, dtype_name, dtype_tl, torch_dtype, native_type
):
    del dtype_name, native_type
    py_op, _ = _AUTO_VEC_OPS[op_key]
    func = _make_auto_vec_binary_kernel(py_op, dtype_tl)
    kernel = tilelang.compile(func, out_idx=[2], target="musa")

    a = torch.randn((M, 4), device="musa", dtype=torch_dtype)
    b = torch.randn((M, 4), device="musa", dtype=torch_dtype)
    c = kernel(a, b)
    ref = _TORCH_REFS[op_key](a, b)
    torch.testing.assert_close(c, ref)


if __name__ == "__main__":
    tilelang.testing.main()
