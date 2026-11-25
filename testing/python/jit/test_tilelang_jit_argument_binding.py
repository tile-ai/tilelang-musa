from dataclasses import dataclass

import pytest
import torch

import tilelang
import tilelang.language as T


@tilelang.jit
def _lazy_kernel_factory(hidden: int, token_stride: int, sf_only: bool = False, cast_only: bool = False):
    @T.prim_func
    def kernel():
        T.evaluate(0)

    return kernel


def test_lazy_jit_cache_key_canonicalizes_defaults_and_call_forms():
    positional_defaults_key, _ = _lazy_kernel_factory.func.parse_args(512, 512)
    positional_explicit_key, _ = _lazy_kernel_factory.func.parse_args(512, 512, False, False)
    kwargs_key, _ = _lazy_kernel_factory.func.parse_args(
        hidden=512,
        token_stride=512,
        sf_only=False,
        cast_only=False,
    )
    reordered_kwargs_key, _ = _lazy_kernel_factory.func.parse_args(
        cast_only=False,
        sf_only=False,
        token_stride=512,
        hidden=512,
    )

    expected = ((512, 512, False, False), None)
    assert positional_defaults_key == expected
    assert positional_explicit_key == expected
    assert kwargs_key == expected
    assert reordered_kwargs_key == expected


def test_lazy_jit_call_form_cache_skips_rebinding(monkeypatch):
    sentinel_kernel = object()
    canonical_key, _ = _lazy_kernel_factory.func.parse_args(
        hidden=512,
        token_stride=512,
    )
    _lazy_kernel_factory.mode = "lazy"
    _lazy_kernel_factory.func.set_mode("lazy")
    _lazy_kernel_factory._kernel_cache[canonical_key] = sentinel_kernel
    _lazy_kernel_factory._call_form_cache.clear()

    assert _lazy_kernel_factory(hidden=512, token_stride=512) is sentinel_kernel

    def fail_parse_args(*args, **kwargs):
        raise AssertionError("call-form cache hit should skip parse_args")

    monkeypatch.setattr(_lazy_kernel_factory.func, "parse_args", fail_parse_args)
    assert _lazy_kernel_factory(hidden=512, token_stride=512) is sentinel_kernel


def test_lazy_jit_call_form_cache_requires_hashable_arguments():
    _lazy_kernel_factory.mode = "lazy"
    _lazy_kernel_factory.func.set_mode("lazy")
    _lazy_kernel_factory._call_form_cache.clear()

    with pytest.raises(TypeError, match="unhashable"):
        _lazy_kernel_factory(hidden=512, token_stride=512, opts=[1, 2])


def test_jit_argument_binding_reports_python_call_errors():
    with pytest.raises(TypeError, match="multiple values"):
        _lazy_kernel_factory.func.parse_args(512, hidden=512, token_stride=512)

    with pytest.raises(TypeError, match="missing a required argument"):
        _lazy_kernel_factory.func.parse_args(hidden=512)

    extra_const_key, _ = _lazy_kernel_factory.func.parse_args(
        hidden=512,
        token_stride=512,
        M=1024,
    )
    assert extra_const_key == ((512, 512, False, False, (("M", 1024),)), None)


@tilelang.jit
def _lazy_var_keyword_factory(M: int, **kwargs):
    @T.prim_func
    def kernel():
        T.evaluate(0)

    return kernel


def test_lazy_jit_var_keyword_kwargs_are_flattened_and_hashable():
    key, kernel_args = _lazy_var_keyword_factory.func.parse_args(
        128,
        N=64,
        opts={"axis": [1, 2]},
    )
    hash(key)
    assert kernel_args == {}

    p1_key, tensor_args, compile_kwargs = _lazy_var_keyword_factory.func._parse_phase1_key(
        128,
        N=64,
        opts={"axis": [1, 2]},
    )
    hash(p1_key)
    assert tensor_args == {}
    assert compile_kwargs == {"M": 128, "N": 64, "opts": {"axis": [1, 2]}}
    assert "kwargs" not in compile_kwargs


_UNHASHABLE_DEFAULT_OPTS = [1, 2]
_UNHASHABLE_DEFAULT_CONFIG = {"axis": [0]}


@dataclass
class _UnhashableConfig:
    axis: int


@tilelang.jit
def _lazy_unhashable_default_factory(
    M: int,
    opts=_UNHASHABLE_DEFAULT_OPTS,
    config=_UNHASHABLE_DEFAULT_CONFIG,
):
    @T.prim_func
    def kernel():
        T.evaluate(0)

    return kernel


def test_lazy_jit_unhashable_defaults_are_normalized_in_cache_key():
    omitted_key, kernel_args = _lazy_unhashable_default_factory.func.parse_args(128)
    explicit_key, _ = _lazy_unhashable_default_factory.func.parse_args(
        128,
        opts=[1, 2],
        config={"axis": [0]},
    )

    hash(omitted_key)
    assert kernel_args == {}
    assert omitted_key == explicit_key


def test_lazy_jit_unsupported_unhashable_compile_time_value_errors():
    with pytest.raises(TypeError, match="Unsupported unhashable JIT compile-time cache key value"):
        _lazy_unhashable_default_factory.func.parse_args(
            128,
            config=_UnhashableConfig(axis=0),
        )


@tilelang.jit
def _copy_kernel(A, B, block: int = 64, dtype: T.dtype = T.float32):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], dtype]
    B: T.Tensor[[M, N], dtype]

    with T.Kernel(T.ceildiv(M, block), T.ceildiv(N, block), threads=128) as (pid_m, pid_n):
        T.copy(A[pid_m * block, pid_n * block], B[pid_m * block, pid_n * block])


def test_eager_jit_phase1_key_excludes_tensor_args_and_applies_defaults():
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    p1_key, tensor_args, compile_kwargs = _copy_kernel.func._parse_phase1_key(a, b)

    assert p1_key == (64, T.float32)
    assert tensor_args == {"A": a, "B": b}
    assert compile_kwargs == {"block": 64, "dtype": T.float32}


def test_eager_jit_phase2_matchers_cache_buffer_and_packed_dtype_metadata():
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    old_mode = _copy_kernel.mode
    old_func_mode = _copy_kernel.func.mode
    _copy_kernel.mode = "eager"
    _copy_kernel.func.set_mode("eager")
    try:
        _copy_kernel.func.parse_args(a, b)
        template = next(iter(_copy_kernel.func.p1_cache.values()))

        assert template.matcher
        buffers = tuple(template.prim_func.buffer_map.values())
        for _, _, _, _, buffer, is_packed in template.matcher.values():
            assert any(buffer.same_as(candidate) for candidate in buffers)
            assert is_packed is False

        _, kernel_args = _copy_kernel.func.parse_args(a, b)
        assert kernel_args == {"A": a, "B": b}
    finally:
        _copy_kernel.mode = old_mode
        _copy_kernel.func.set_mode(old_func_mode)


def test_eager_jit_hot_call_reuses_kernel_with_current_tensors(monkeypatch):
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)
    a_next = torch.empty_like(a)
    b_next = torch.empty_like(b)
    calls = []

    def sentinel_kernel(*args):
        calls.append(args)
        return "sentinel"

    old_mode = _copy_kernel.mode
    old_func_mode = _copy_kernel.func.mode
    old_kernel_cache = _copy_kernel._kernel_cache
    old_eager_cache = _copy_kernel._eager_call_cache
    _copy_kernel.mode = "eager"
    _copy_kernel.func.set_mode("eager")
    _copy_kernel._kernel_cache = {}
    _copy_kernel._eager_call_cache = old_eager_cache.__class__()
    try:
        key, _ = _copy_kernel.func.parse_args(a, b, 64, T.float32)
        _copy_kernel._kernel_cache[key] = sentinel_kernel
        assert _copy_kernel(a, b, 64, T.float32) == "sentinel"

        def fail_parse_args(*args, **kwargs):
            raise AssertionError("eager hot-call cache hit should skip parse_args")

        monkeypatch.setattr(_copy_kernel.func, "parse_args", fail_parse_args)
        assert _copy_kernel(a_next, b_next, 64, T.float32) == "sentinel"
        assert calls[-1] == (a_next, b_next)
    finally:
        _copy_kernel.mode = old_mode
        _copy_kernel.func.set_mode(old_func_mode)
        _copy_kernel._kernel_cache = old_kernel_cache
        _copy_kernel._eager_call_cache = old_eager_cache


def test_eager_jit_hot_call_rechecks_metadata_after_in_place_resize(monkeypatch):
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    def sentinel_kernel(*args):
        return args

    old_mode = _copy_kernel.mode
    old_func_mode = _copy_kernel.func.mode
    old_kernel_cache = _copy_kernel._kernel_cache
    old_eager_cache = _copy_kernel._eager_call_cache
    _copy_kernel.mode = "eager"
    _copy_kernel.func.set_mode("eager")
    _copy_kernel._kernel_cache = {}
    _copy_kernel._eager_call_cache = old_eager_cache.__class__()
    try:
        key, _ = _copy_kernel.func.parse_args(a, b, 64, T.float32)
        _copy_kernel._kernel_cache[key] = sentinel_kernel
        _copy_kernel(a, b, 64, T.float32)
        a.resize_(4, 16)
        b.resize_(4, 16)

        def mark_parse_args(*args, **kwargs):
            raise RuntimeError("normal argument binding reached")

        monkeypatch.setattr(_copy_kernel.func, "parse_args", mark_parse_args)
        with pytest.raises(RuntimeError, match="normal argument binding reached"):
            _copy_kernel(a, b, 64, T.float32)
    finally:
        _copy_kernel.mode = old_mode
        _copy_kernel.func.set_mode(old_func_mode)
        _copy_kernel._kernel_cache = old_kernel_cache
        _copy_kernel._eager_call_cache = old_eager_cache


@pytest.mark.parametrize(
    "make_args",
    [
        lambda a, b: (torch.empty(4, 16), torch.empty(4, 16), 64, T.float32),
        lambda a, b: (torch.empty(8, 16)[:, ::2], torch.empty(8, 16)[:, ::2], 64, T.float32),
        lambda a, b: (torch.empty(8, 8, dtype=torch.float64), torch.empty(8, 8, dtype=torch.float64), 64, T.float32),
        lambda a, b: (a, b, 32, T.float32),
    ],
    ids=["shape", "stride", "dtype", "compile-argument"],
)
def test_eager_jit_hot_call_rebinds_when_metadata_changes(monkeypatch, make_args):
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    def sentinel_kernel(*args):
        return args

    old_mode = _copy_kernel.mode
    old_func_mode = _copy_kernel.func.mode
    old_kernel_cache = _copy_kernel._kernel_cache
    old_eager_cache = _copy_kernel._eager_call_cache
    _copy_kernel.mode = "eager"
    _copy_kernel.func.set_mode("eager")
    _copy_kernel._kernel_cache = {}
    _copy_kernel._eager_call_cache = old_eager_cache.__class__()
    try:
        key, _ = _copy_kernel.func.parse_args(a, b, 64, T.float32)
        _copy_kernel._kernel_cache[key] = sentinel_kernel
        _copy_kernel(a, b, 64, T.float32)

        def mark_parse_args(*args, **kwargs):
            raise RuntimeError("normal argument binding reached")

        monkeypatch.setattr(_copy_kernel.func, "parse_args", mark_parse_args)
        with pytest.raises(RuntimeError, match="normal argument binding reached"):
            _copy_kernel(*make_args(a, b))
    finally:
        _copy_kernel.mode = old_mode
        _copy_kernel.func.set_mode(old_func_mode)
        _copy_kernel._kernel_cache = old_kernel_cache
        _copy_kernel._eager_call_cache = old_eager_cache


@tilelang.jit
def _copy_kernel_with_var_keywords(A, B, block: int = 64, **metadata):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], T.float32]
    B: T.Tensor[[M, N], T.float32]

    with T.Kernel(T.ceildiv(M, block), T.ceildiv(N, block), threads=128) as (pid_m, pid_n):
        T.copy(A[pid_m * block, pid_n * block], B[pid_m * block, pid_n * block])


def test_eager_jit_var_keyword_compile_kwargs_are_flattened():
    a = torch.empty(8, 8)
    b = torch.empty(8, 8)

    p1_key, tensor_args, compile_kwargs = _copy_kernel_with_var_keywords.func._parse_phase1_key(
        a,
        b,
        opts={"axis": [1, 2]},
    )

    hash(p1_key)
    assert tensor_args == {"A": a, "B": b}
    assert compile_kwargs == {"block": 64, "opts": {"axis": [1, 2]}}
    assert "metadata" not in compile_kwargs
