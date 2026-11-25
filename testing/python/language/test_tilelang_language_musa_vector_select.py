"""MUSA vector predicates require elementwise selection, not a C++ ternary."""

import os
from contextlib import nullcontext
from pathlib import Path
import subprocess

import pytest

from tilelang import language as T, tvm
from tilelang.engine import lower


def make_vector_select(dtype, lanes):
    @T.prim_func
    def kernel(
        predicate: T.Tensor((128, lanes), "int32"),
        lhs: T.Tensor((128, lanes), dtype),
        rhs: T.Tensor((128, lanes), dtype),
        output: T.Tensor((128, lanes), dtype),
    ):
        with T.Kernel(1, threads=128):
            t = T.get_thread_binding()
            for i in T.vectorized(lanes):
                output[t, i] = T.Select(predicate[t, i] != 0, lhs[t, i], rhs[t, i])

    return kernel


def lower_source(kernel, arch="mp_31", with_scope=True):
    target = tvm.target.Target({"kind": "musa", "arch": arch})
    with target if with_scope else nullcontext(), tvm.transform.PassContext(opt_level=3):
        return lower(kernel, target=target, enable_host_codegen=False, enable_device_compile=False).kernel_source


@pytest.mark.parametrize("arch", ["mp_31", "mp_22"])
@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32", "uint32", "int8", "uint8"])
@pytest.mark.parametrize("lanes", [2, 4])
def test_vector_select_source_and_mcc(tmp_path, dtype, lanes, arch):
    source = lower_source(make_vector_select(dtype, lanes), arch=arch)
    assert "select_vec" in source
    assert source.count(" ? ") >= lanes
    if dtype in ("float16", "bfloat16"):
        # Selecting native storage lanes must not round-trip through float.
        selected_lines = [line for line in source.splitlines() if "select_vec" in line and " ? " in line]
        assert len(selected_lines) == lanes
        assert all("make_bfloat16" not in line and "make_half" not in line for line in selected_lines)
    mcc = os.environ.get("TILELANG_TEST_MCC")
    if mcc:
        root = Path(__file__).resolve().parents[3]
        source_path = tmp_path / "kernel.mu"
        source_path.write_text(source)
        result = subprocess.run(
            [
                mcc,
                "--cuda-device-only",
                "--cuda-gpu-arch=" + arch,
                "--musa-path=" + os.environ.get("MUSA_HOME", "/usr/local/musa"),
                "-B" + str(Path(mcc).absolute().parent),
                "-I" + str(root / "src"),
                "-I" + str(root / "3rdparty/mutlass/include"),
                "-std=c++17",
                "-Wno-error=c++11-narrowing",
                "-Wno-error=address-of-temporary",
                "-O3",
                "-c",
                str(source_path),
                "-o",
                str(tmp_path / "kernel.o"),
            ],
            capture_output=True,
            text=True,
            timeout=60,
        )
        assert result.returncode == 0, result.stdout + result.stderr


def test_scalar_select_preserves_nested_lazy_expression():
    @T.prim_func
    def kernel(source: T.Tensor((128,), "float32"), output: T.Tensor((128,), "float32")):
        with T.Kernel(1, threads=128):
            t = T.get_thread_binding()
            output[t] = T.Select(source[t] > 0, T.if_then_else(source[t] < 2, source[t], 2.0), -1.0)

    source = lower_source(kernel)
    assert "select_vec" not in source
    assert " ? " in source
