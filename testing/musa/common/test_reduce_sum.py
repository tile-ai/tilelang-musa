import re

import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
import torch
from tilelang import tvm

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def reduce_sum_cross_warp(A, threads=128):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], T.float32]
    B = T.empty((M,), "float32")

    with T.Kernel(1, threads=threads) as _:
        a_local = T.alloc_fragment((M, N), "float32")
        b_local = T.alloc_fragment((M,), "float32")

        T.copy(A, a_local)
        T.reduce_sum(a_local, b_local, dim=1)
        T.copy(b_local, B)

    return B


def _extract_named_barrier_base(source):
    match = re.search(r"tl::NamedBarrier<([0-9]+)>", source)
    return int(match.group(1)) if match is not None else None


def _extract_allreduce_threads(source):
    match = re.search(r"AllReduce<[^,]+,\s*([0-9]+),", source)
    assert match is not None, source
    return int(match.group(1))


def _assert_named_barrier_init_if_used(source):
    base_id = _extract_named_barrier_base(source)
    if base_id is None:
        return

    init0 = re.search(rf"__musa_async_init_arrival\(\s*{base_id}\s*,", source)
    init1 = re.search(rf"__musa_async_init_arrival\(\s*{base_id + 1}\s*,", source)
    assert init0 is not None, source
    assert init1 is not None, source


def _assert_barrier_arrival_count(source, barrier_id, reducing_threads):
    match = re.search(
        rf"__musa_async_init_arrival\(\s*{barrier_id}\s*,\s*([^,]+),\s*0\s*\)",
        source,
    )
    assert match is not None, source
    arrival_count = re.sub(r"\s+", "", match.group(1))
    expected_warps = (reducing_threads + 31) // 32
    assert arrival_count in {
        str(expected_warps),
        f"(({reducing_threads}+31)/32)",
    }, source


def _assert_reduce_barrier_matches_participants(source):
    base_id = _extract_named_barrier_base(source)
    assert base_id is not None, source
    reducing_threads = _extract_allreduce_threads(source)
    _assert_barrier_arrival_count(source, base_id, reducing_threads)
    _assert_barrier_arrival_count(source, base_id + 1, reducing_threads)


def ref_program(x):
    return torch.sum(x, dim=1)


def reduce_sum_with_thread_y(thread_y_extent=4, batch=1):
    M = 32
    N = 512

    @T.prim_func
    def main(
        A: T.Tensor((M, N), T.float32),
        B: T.Tensor((M,), T.float32),
    ):
        with T.Kernel(1, threads=(128, thread_y_extent)) as _:
            a_local = T.alloc_fragment((M, N), T.float32)
            b_local = T.alloc_fragment((M,), T.float32)

            T.copy(A, a_local)
            if batch > 1:
                T.reduce_sum(a_local, b_local, dim=1, batch=batch)
            else:
                T.reduce_sum(a_local, b_local, dim=1)
            T.copy(b_local, B)

    return main


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_ph1_thread_y_reduce_uses_original_allreduce():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        artifact = tilelang.lower(reduce_sum_with_thread_y(), target=target)
    source = artifact.kernel_source

    assert "tl::AllReduce<" in source, source
    assert "AllReduceSubgroup<" not in source, source


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_ph1_scalar_reduce_initializes_named_barriers():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        artifact = tilelang.lower(
            reduce_sum_with_thread_y(thread_y_extent=1),
            target=target,
        )
    source = artifact.kernel_source

    assert "tl::AllReduce<" in source, source
    assert "::run_batch" not in source, source
    _assert_reduce_barrier_matches_participants(source)


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_ph1_batch_reduce_initializes_named_barriers():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        artifact = tilelang.lower(
            reduce_sum_with_thread_y(thread_y_extent=1, batch=4),
            target=target,
        )
    source = artifact.kernel_source

    assert "tl::AllReduce<" in source, source
    assert "::run_batch" in source, source
    _assert_reduce_barrier_matches_participants(source)


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_qy2_thread_y_reduce_uses_subgroup_allreduce():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_22"})
    with target:
        artifact = tilelang.lower(reduce_sum_with_thread_y(), target=target)
    source = artifact.kernel_source

    assert "tl::AllReduceSubgroup<" in source, source
    assert "tl::AllReduce<" not in source, source


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_qy2_single_thread_y_reduce_uses_original_allreduce():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_22"})
    with target:
        artifact = tilelang.lower(
            reduce_sum_with_thread_y(thread_y_extent=1),
            target=target,
        )
    source = artifact.kernel_source

    assert "tl::AllReduce<" in source, source
    assert "AllReduceSubgroup<" not in source, source


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_qy2_thread_y_batch_reduce_uses_subgroup_allreduce():
    target = tvm.target.Target({"kind": "musa", "arch": "mp_22"})
    with target:
        artifact = tilelang.lower(
            reduce_sum_with_thread_y(batch=4),
            target=target,
        )
    source = artifact.kernel_source

    assert "tl::AllReduceSubgroup<" in source, source
    assert "::run_batch" in source, source
    assert "tl::AllReduce<" not in source, source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_reduce_sum_cross_warp():
    M = 32
    N = 512

    kernel = reduce_sum_cross_warp.compile(M=M, N=N)
    source = kernel.get_kernel_source()

    assert "tl::AllReduce<" in source, source
    allreduce_threads = _extract_allreduce_threads(source)
    assert allreduce_threads > 32, source
    _assert_named_barrier_init_if_used(source)

    a = torch.randn(M, N, dtype=torch.float32, device="musa")
    b = kernel(a)
    ref = ref_program(a)
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


@pytest.mark.parametrize("batch", [1, 4])
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_reduce_sum_cross_warp_workspace_reuse(batch):
    kernel = tilelang.compile(
        reduce_sum_with_thread_y(thread_y_extent=1, batch=batch),
        target="musa",
        out_idx=[1],
    )
    source = kernel.get_kernel_source()
    _assert_reduce_barrier_matches_participants(source)
    assert _extract_allreduce_threads(source) >= 128, source

    for _ in range(32):
        a = torch.randint(-4, 5, (32, 512), device="musa").float()
        b = kernel(a)
        # Small integers make every intermediate sum exactly representable.
        torch.testing.assert_close(b, a.sum(dim=1), rtol=0, atol=0)


def main():
    M = 32
    N = 512
    kernel = reduce_sum_cross_warp.compile(M=M, N=N)
    source = kernel.get_kernel_source()
    print(source)

    a = torch.randn(M, N, dtype=torch.float32, device="musa")
    b = kernel(a)
    ref = ref_program(a)
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)
    print("pass")


if __name__ == "__main__":
    main()
