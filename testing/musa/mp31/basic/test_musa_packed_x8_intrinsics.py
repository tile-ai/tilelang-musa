import pytest
import torch

import tilelang
from tilelang import tvm
from tilelang.engine.lower import lower
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()

M = 128

_AUTO_VEC_OPS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": T.max,
    "min": T.min,
}

_TORCH_REFS = {
    "add": lambda a, b: a + b,
    "sub": lambda a, b: a - b,
    "mul": lambda a, b: a * b,
    "max": lambda a, b: torch.maximum(a, b),
    "min": lambda a, b: torch.minimum(a, b),
}

_CODEGEN_CASES = [
    (T.float32, 6, "tl_f2", "2", 3),
    (T.float32, 8, "tl_f4", "4", 2),
    (T.float16, 8, "tl_h4", "4", 2),
    (T.float16, 12, "tl_h4", "4", 3),
    (T.float16, 16, "tl_h4", "4", 4),
    (T.bfloat16, 8, "tl_bf4", "4", 2),
    (T.bfloat16, 12, "tl_bf4", "4", 3),
    (T.bfloat16, 16, "tl_bf4", "4", 4),
]


def _make_auto_vec_binary_kernel(py_op, dtype_tl=T.float32, width: int = 8):
    @T.prim_func
    def main(
        A: T.Tensor((M * width,), dtype=dtype_tl),
        B: T.Tensor((M * width,), dtype=dtype_tl),
        C: T.Tensor((M * width,), dtype=dtype_tl),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            tid = T.get_thread_binding()
            idx = T.Ramp(tid * width, 1, width)
            C[idx] = py_op(A[idx], B[idx])

    return main


def _lower_to_musa_source(func):
    with tvm.transform.PassContext(), tvm.target.Target("musa"):
        artifact = lower(func, target="musa", enable_device_compile=False)
    assert artifact.kernel_source is not None
    return artifact.kernel_source


@pytest.mark.parametrize("op_key", list(_AUTO_VEC_OPS.keys()))
@pytest.mark.parametrize("dtype_tl,width,chunk_type,op_width,num_chunks", _CODEGEN_CASES)
def test_musa_codegen_auto_vec_wide_uses_tilelang_chunks(
    op_key, dtype_tl, width, chunk_type, op_width, num_chunks
):
    py_op = _AUTO_VEC_OPS[op_key]
    src = _lower_to_musa_source(_make_auto_vec_binary_kernel(py_op, dtype_tl, width))
    assert "\x00" not in src, "Generated MUSA source should not contain embedded NUL bytes"
    assert f"tl::{op_key}{op_width}" in src
    assert src.count(f"tl::{op_key}{op_width}") == num_chunks
    assert chunk_type in src
    assert "tl_f8" not in src


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("op_key", list(_AUTO_VEC_OPS.keys()))
def test_musa_correctness_auto_vec_f32_width8(op_key):
    py_op = _AUTO_VEC_OPS[op_key]
    func = _make_auto_vec_binary_kernel(py_op)
    kernel = tilelang.compile(func, out_idx=[2], target="musa")

    a = torch.randn((M * 8,), device="musa", dtype=torch.float32)
    b = torch.randn((M * 8,), device="musa", dtype=torch.float32)
    c = kernel(a, b)
    ref = _TORCH_REFS[op_key](a, b)
    torch.testing.assert_close(c, ref)


if __name__ == "__main__":
    tilelang.testing.main()
