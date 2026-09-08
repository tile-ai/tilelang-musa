import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import torch

from tilelang.musa.language.memory import (
    _INNER_PERSISTENCE,
    normalize_lsu_hint,
)


@T.prim_func
def _cache_hint_copy(
    source: T.Tensor((128,), T.float32),
    destination: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        thread = T.get_thread_binding()
        destination[thread] = T.lsu_ld_cache_hint(
            source[thread],
            inner_persistence="cache_persist",
            outer_persistence="cache_once",
            chrnt="l1",
            l2="new_alloc",
        )


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_lsu_cache_hint_codegen_and_correctness():
    kernel = tilelang.compile(
        _cache_hint_copy,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    source = torch.randn(128, dtype=torch.float32, device="musa")
    result = kernel(source)
    torch.musa.synchronize()
    assert "tl::lsu_ld_cache_hint<3, 1, 0, 0, false>" in kernel.get_kernel_source()
    torch.testing.assert_close(result, source, atol=0, rtol=0)


def test_lsu_cache_hint_rejects_invalid_policies():
    with pytest.raises(ValueError, match="Unsupported inner_persistence"):
        normalize_lsu_hint("invalid", "inner_persistence", _INNER_PERSISTENCE, 5)
    with pytest.raises(ValueError, match=r"must be in \[0, 5\]"):
        normalize_lsu_hint(6, "inner_persistence", _INNER_PERSISTENCE, 5)


if __name__ == "__main__":
    tilelang.testing.main()
