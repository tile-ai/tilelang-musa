import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

CACHE_HINT_CODEGEN_CASES = [
    pytest.param({}, ("CACHE_NORMAL", "CACHE_NORMAL"), id="default"),
    pytest.param(
        {"eviction_policy": "evict_first"},
        ("CACHE_ONCE", "CACHE_ONCE"),
        id="evict_first",
    ),
    pytest.param(
        {"eviction_policy": "evict_last"},
        ("CACHE_PERSIST", "CACHE_PERSIST"),
        id="evict_last",
    ),
    pytest.param(
        {
            "inner_cache_policy": "cache_none",
            "outer_cache_policy": "cache_persist",
        },
        ("CACHE_NONE", "CACHE_PERSIST"),
        id="inner_none_outer_persist",
    ),
    pytest.param(
        {"inner_cache_policy": "cache_none"},
        ("CACHE_NONE", "CACHE_NORMAL"),
        id="inner_none_outer_default",
    ),
]


def _make_tme_load_kernel(
    *,
    eviction_policy=None,
    inner_cache_policy=None,
    outer_cache_policy=None,
):
    @tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
    def kernel(A, dtype="float32"):
        M, N = T.const("M N")
        A: T.Tensor[[M, N], dtype]
        C = T.empty((M, N), dtype)

        with T.Kernel(1, threads=128) as _:
            tile = T.alloc_shared((M, N), dtype)
            mbar = T.alloc_barrier(128)
            T.tma_copy(
                A[0:M, 0:N],
                tile,
                barrier=mbar,
                eviction_policy=eviction_policy,
                inner_cache_policy=inner_cache_policy,
                outer_cache_policy=outer_cache_policy,
            )
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)
            T.copy(tile, C[0:M, 0:N], disable_tma=True)

        return C

    return kernel


def _make_tme_store_kernel(
    *,
    eviction_policy=None,
    inner_cache_policy=None,
    outer_cache_policy=None,
):
    @tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
    def kernel(A, dtype="float32"):
        M, N = T.const("M N")
        A: T.Tensor[[M, N], dtype]
        C = T.empty((M, N), dtype)

        with T.Kernel(1, threads=128) as _:
            tile = T.alloc_shared((M, N), dtype)
            T.copy(A[0:M, 0:N], tile, disable_tma=True)
            T.copy(
                tile,
                C[0:M, 0:N],
                eviction_policy=eviction_policy,
                inner_cache_policy=inner_cache_policy,
                outer_cache_policy=outer_cache_policy,
            )

        return C

    return kernel


def _compile_load_source(**kwargs):
    compiled = _make_tme_load_kernel(**kwargs).compile(
        M=16,
        N=128,
        dtype="float32",
    )
    return compiled, compiled.get_kernel_source()


def _compile_store_source(**kwargs):
    compiled = _make_tme_store_kernel(**kwargs).compile(
        M=16,
        N=128,
        dtype="float32",
    )
    return compiled, compiled.get_kernel_source()


def _assert_musa_tme_cache_hints(source, callee, inner_hint, outer_hint):
    flat_source = " ".join(source.split())
    pattern = (
        rf"tl::{callee}<[^>]*"
        rf"CacheHint::{inner_hint},\s*CacheHint::{outer_hint}[^>]*>"
    )
    assert re.search(pattern, flat_source), source
    assert "CacheHintSm90" not in source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "kwargs, expected",
    CACHE_HINT_CODEGEN_CASES,
)
def test_musa_tme_load_cache_hint_codegen(kwargs, expected):
    # compile() invokes the MUSA device toolchain, so this checks both TileLang
    # source emission and downstream device/ASM code generation.
    _, source = _compile_load_source(**kwargs)
    _assert_musa_tme_cache_hints(source, "tma_load", *expected)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "kwargs, expected",
    CACHE_HINT_CODEGEN_CASES,
)
def test_musa_tme_store_cache_hint_codegen(kwargs, expected):
    _, source = _compile_store_source(**kwargs)
    _assert_musa_tme_cache_hints(source, "tma_store", *expected)
    assert "tl::tma_store_arrive()" in source
    assert "tl::tma_store_wait<0>()" in source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_musa_tme_cache_none_end_to_end():
    compiled, source = _compile_load_source(
        inner_cache_policy="cache_none",
        outer_cache_policy="cache_normal",
    )
    _assert_musa_tme_cache_hints(source, "tma_load", "CACHE_NONE", "CACHE_NORMAL")

    a = torch.randn((16, 128), device="musa", dtype=torch.float32)
    c = compiled(a)
    torch.musa.synchronize()
    torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_musa_tme_store_cache_none_end_to_end():
    compiled, source = _compile_store_source(
        inner_cache_policy="cache_none",
        outer_cache_policy="cache_normal",
    )
    _assert_musa_tme_cache_hints(source, "tma_store", "CACHE_NONE", "CACHE_NORMAL")

    a = torch.randn((16, 128), device="musa", dtype=torch.float32)
    c = compiled(a)
    torch.musa.synchronize()
    torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_eviction_policy_conflicts_with_musa_cache_policy():
    with pytest.raises(ValueError, match="eviction_policy cannot be combined"):
        _make_tme_load_kernel(
            eviction_policy="evict_first",
            inner_cache_policy="cache_none",
        ).compile(M=16, N=128, dtype="float32")


if __name__ == "__main__":
    tilelang.testing.main()
