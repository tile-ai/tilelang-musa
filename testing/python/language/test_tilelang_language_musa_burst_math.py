"""Regression tests for MUSA burst vector math lowering."""

import os
from pathlib import Path
import subprocess

import pytest

from tilelang import language as T, tvm
from tilelang.engine import lower
from tilelang.transform import PassConfigKey


LOG2E = 1.4426950408889634


def _expression(value, op):
    if op == "exp2_negate":
        return T.exp2(-value * LOG2E)
    if op == "exp2_coefficient":
        return T.exp2(value * -LOG2E)
    if op == "native_mix":
        return T.exp2(value * -LOG2E) + T.call_pure_extern(
            value.dtype, "__tanhf", value
        )
    raise ValueError(op)


def _make_kernel(op, explicit_vector):
    @T.prim_func
    def kernel(x: T.Tensor((1024,), T.float32), y: T.Tensor((1024,), T.float32)):
        with T.Kernel(1, threads=128):
            base = T.get_thread_binding() * 8
            values = T.alloc_local((8,), T.float32)
            result = T.alloc_local((8,), T.float32)
            for i in T.vectorized(8):
                values[i] = x[base + i]
            if explicit_vector:
                for chunk in T.unroll(2):
                    lanes = T.Ramp(chunk * 4, 1, 4)
                    result[lanes] = _expression(values[lanes], op)
            else:
                for i in T.vectorized(8):
                    result[i] = _expression(values[i], op)
            for i in T.vectorized(8):
                y[base + i] = result[i]

    return kernel


def _lower_source(kernel, arch):
    target = tvm.target.Target({"kind": "musa", "arch": arch})
    config = {
        PassConfigKey.TL_ENABLE_MUSA_BURST.value: True,
        PassConfigKey.TL_ENABLE_FAST_MATH.value: False,
    }
    with target, tvm.transform.PassContext(opt_level=3, config=config):
        return lower(
            kernel,
            target=target,
            enable_host_codegen=False,
            enable_device_compile=False,
        ).kernel_source


def _compile_with_mcc(source, tmp_path, name, arch):
    mcc = os.environ.get("TILELANG_TEST_MCC")
    if not mcc:
        return
    root = Path(__file__).resolve().parents[3]
    source_path = tmp_path / f"{name}.mu"
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
            str(tmp_path / f"{name}.o"),
        ],
        capture_output=True,
        text=True,
        timeout=60,
    )
    assert result.returncode == 0, result.stdout + result.stderr


@pytest.mark.parametrize("arch", ["mp_31", "mp_22"])
def test_pure_extern_preserves_vectorizable_math(tmp_path, arch):
    source = _lower_source(_make_kernel("native_mix", explicit_vector=False), arch)
    assert "tl::vec_exp2_f4" in source
    assert "__tanhf" in source
    assert "for (int i_1 = 0; i_1 < 8; ++i_1)" not in source
    _compile_with_mcc(source, tmp_path, "native_mix_loop", arch)
