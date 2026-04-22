import re

import torch

import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
}


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def copy_with_manual_tma_barrier(A, block_n, dtype="float32"):
    N = T.const("N")
    A: T.Tensor[[N], dtype]
    C = T.empty((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
        tile = T.alloc_shared((block_n,), dtype)
        barrier = T.alloc_barrier(128)
        T.copy(A[bx * block_n], tile, barrier=barrier)
        T.barrier_arrive(barrier)
        T.barrier_wait(barrier, 0)
        T.copy(tile, C[bx * block_n])

    return C


@tilelang.jit(target="musa", pass_configs=PASS_CONFIGS)
def copy_with_manual_tma_barrier_dynamic_index(A, block_n, dtype="float32"):
    N = T.const("N")
    A: T.Tensor[[N], dtype]
    C = T.empty((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
        tile = T.alloc_shared((block_n,), dtype)
        barriers = T.alloc_barrier([128, 128])
        T.copy(A[bx * block_n], tile, barrier=barriers[bx % 2])
        T.barrier_arrive(barriers[bx % 2])
        T.barrier_wait(barriers[bx % 2], 0)
        T.copy(tile, C[bx * block_n])

    return C


def _assert_tma_barrier_source(source, expected_bytes, require_non_constant=False):
    flat_source = " ".join(source.split())
    expect_tx = re.search(rf"__musa_async_add_trans\(([^,]+),\s*{expected_bytes}\)", flat_source)
    assert expect_tx, source
    barrier_expr = expect_tx.group(1).strip()
    if require_non_constant:
        assert not re.fullmatch(r"\d+", barrier_expr), source

    barrier_id = re.escape(barrier_expr)
    tma_load_pattern = rf"tl::tma_load\([^;]*,\s*{barrier_id}\s*,\s*{expected_bytes}\)"
    assert re.search(tma_load_pattern, flat_source), source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_copy_with_manual_tma_barrier_end_to_end():
    n = 1024
    block_n = 128
    expected_bytes = block_n * 4

    compiled_kernel = copy_with_manual_tma_barrier.compile(
        N=n,
        block_n=block_n,
        dtype="float32",
    )

    source = compiled_kernel.get_kernel_source()
    _assert_tma_barrier_source(source, expected_bytes)

    a = torch.randn((n,), device="musa", dtype=torch.float32)
    c = compiled_kernel(a)

    torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_copy_with_manual_tma_barrier_dynamic_index_end_to_end():
    n = 1024
    block_n = 128
    expected_bytes = block_n * 4

    compiled_kernel = copy_with_manual_tma_barrier_dynamic_index.compile(
        N=n,
        block_n=block_n,
        dtype="float32",
    )

    source = compiled_kernel.get_kernel_source()
    _assert_tma_barrier_source(source, expected_bytes, require_non_constant=True)

    a = torch.randn((n,), device="musa", dtype=torch.float32)
    c = compiled_kernel(a)

    torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    n = 1024
    block_n = 128
    expected_bytes = block_n * 4

    compiled_kernel = copy_with_manual_tma_barrier_dynamic_index.compile(
        N=n,
        block_n=block_n,
        dtype="float32",
    )

    print(compiled_kernel.get_kernel_source())

    a = torch.randn((n,), device="musa", dtype=torch.float32)
    c = compiled_kernel(a)

    torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)
