import pytest
import torch

import tilelang
import tilelang.language as T
from tilelang import PassConfigKey
from tilelang.contrib import mcc


@pytest.fixture
def inputs(monkeypatch):
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")
    # Recompile with the active toolchain when comparing MTCC versions.
    monkeypatch.setenv("TILELANG_DISABLE_CACHE", "1")
    a = torch.randn((128, 32), device="musa", dtype=torch.float32)
    scale = torch.randn((128, 2), device="musa", dtype=torch.float32)
    expected = a * scale.repeat_interleave(16, dim=1)
    return a, scale, expected


@tilelang.jit
def _scale_fragment(A, Scale, kind: str):
    A: T.Tensor((128, 32), T.float32)
    Scale: T.Tensor((128, 2), T.float32)
    Out = T.empty((128, 32), T.float32)
    with T.Kernel(1, threads=128):
        acc = T.alloc_fragment((128, 32), T.float32)
        T.annotate_layout({acc: T.Fragment((128, 32), lambda i, j: (i, j))})
        T.copy(A, acc)
        if kind == "fragment":
            scale = T.alloc_fragment((128, 2), T.float32)
            T.annotate_layout({scale: T.Fragment((128, 2), lambda i, j: (i, j))})
            T.copy(Scale, scale)
            for i, j in T.Parallel(128, 32):
                acc[i, j] *= scale[i, j // 16]
        elif kind == "local":
            scale = T.alloc_local((2,), T.float32)
            tx = T.get_thread_binding()
            for j in T.serial(2):
                scale[j] = Scale[tx, j]
            for i, j in T.Parallel(128, 32):
                acc[i, j] *= scale[j // 16]
        else:
            for i, j in T.Parallel(128, 32):
                acc[i, j] *= Scale[i, j // 16]
        T.copy(acc, Out)
    return Out


def _compile_and_check(kind, burst, inputs, disable=None):
    func = _scale_fragment.get_tir(T.Tensor((128, 32), T.float32), T.Tensor((128, 2), T.float32), kind)
    config = {
        PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        PassConfigKey.TL_ENABLE_MUSA_BURST: burst,
    }
    if disable is not None:
        config[PassConfigKey.TL_DISABLE_FRAGMENT_VECTORIZE] = disable
    # Infer the architecture from the current device; do not cross-compile.
    kernel = tilelang.compile(func, target="musa", pass_configs=config)
    a, scale, expected = inputs
    actual = kernel(a, scale)
    torch.musa.synchronize()
    torch.testing.assert_close(actual, expected, rtol=0.0, atol=0.0)
    return kernel.get_kernel_source()


@pytest.mark.parametrize("burst", [False, True])
def test_fragment_only_vectorization_can_be_disabled(burst, inputs):
    default = _compile_and_check("fragment", burst, inputs)
    enabled = _compile_and_check("fragment", burst, inputs, disable=False)
    disabled = _compile_and_check("fragment", burst, inputs, disable=True)
    assert default == (disabled if mcc.get_musa_version()[:2] < (5, 3) else enabled)
    assert "tl::mul4(" in enabled
    if burst:
        # The independent burst pass may still vectorize this multiplication.
        assert "tl::mul4(" in disabled
    else:
        assert "tl::mul4(" not in disabled
        assert "tl::mul2(" not in disabled
    # Global/fragment copies must remain vectorized in the fallback kernel.
    assert "float4" in disabled or "tl_f4" in disabled


@pytest.mark.parametrize("burst", [False, True])
@pytest.mark.parametrize("kind", ["local", "global"])
def test_mixed_fragment_loops_are_unchanged(burst, kind, inputs):
    enabled = _compile_and_check(kind, burst, inputs, disable=False)
    disabled = _compile_and_check(kind, burst, inputs, disable=True)
    assert enabled == disabled
