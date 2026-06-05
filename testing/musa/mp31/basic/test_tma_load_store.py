import gc
import re

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang.contrib import mcc

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def tma_add_one_1d(A, dim0, dtype="float32"):
    D0 = T.const("D0")
    A: T.Tensor[[D0], dtype]
    C = T.empty((D0,), dtype)

    with T.Kernel(1, threads=128) as _:
        tile = T.alloc_shared((dim0,), dtype)
        mbar = T.alloc_barrier(128)
        T.tma_copy(A[0], tile, barrier=mbar)
        T.barrier_arrive(mbar)
        T.mbarrier_wait_parity(mbar, 0)
        for i in T.grid(dim0):
            tile[i] = tile[i] + 1
        T.copy(tile, C[0])

    return C


@tilelang.jit(target="musa", verbose=True)
def tma_add_one_2d(A, dim0, dim1, dtype="float32"):
    D0, D1 = T.const("D0 D1")
    A: T.Tensor[[D0, D1], dtype]
    C = T.empty((D0, D1), dtype)

    with T.Kernel(1, threads=128) as _:
        tile = T.alloc_shared((dim0, dim1), dtype)
        mbar = T.alloc_barrier(128)
        T.tma_copy(A[0, 0], tile, barrier=mbar)
        T.barrier_arrive(mbar)
        T.mbarrier_wait_parity(mbar, 0)
        for i, j in T.grid(dim0, dim1):
            tile[i, j] = tile[i, j] + 1
        T.copy(tile, C[0, 0])

    return C


@tilelang.jit(target="musa", verbose=True)
def tma_add_one_3d(A, dim0, dim1, dim2, dtype="float32"):
    D0, D1, D2 = T.const("D0 D1 D2")
    A: T.Tensor[[D0, D1, D2], dtype]
    C = T.empty((D0, D1, D2), dtype)

    with T.Kernel(dim0, threads=128) as i:
        tile = T.alloc_shared((dim1, dim2), dtype)
        mbar = T.alloc_barrier(128)
        T.tma_copy(A[i, 0, 0], tile, barrier=mbar)
        T.barrier_arrive(mbar)
        T.mbarrier_wait_parity(mbar, 0)
        for j, k in T.grid(dim1, dim2):
            tile[j, k] = tile[j, k] + 1
        T.copy(tile, C[i, 0, 0])

    return C


@tilelang.jit(target="musa", verbose=True)
def tma_add_one_4d(A, dim0, dim1, dim2, dim3, dtype="float32"):
    D0, D1, D2, D3 = T.const("D0 D1 D2 D3")
    A: T.Tensor[[D0, D1, D2, D3], dtype]
    C = T.empty((D0, D1, D2, D3), dtype)

    with T.Kernel(dim0, dim1, threads=128) as (i, j):
        tile = T.alloc_shared((dim2, dim3), dtype)
        mbar = T.alloc_barrier(128)
        T.tma_copy(A[i, j, 0, 0], tile, barrier=mbar)
        T.barrier_arrive(mbar)
        T.mbarrier_wait_parity(mbar, 0)
        for k, l in T.grid(dim2, dim3):
            tile[k, l] = tile[k, l] + 1
        T.copy(tile, C[i, j, 0, 0])

    return C


@tilelang.jit(target="musa", verbose=True)
def tma_add_one_5d(A, dim0, dim1, dim2, dim3, dim4, dtype="float32"):
    D0, D1, D2, D3, D4 = T.const("D0 D1 D2 D3 D4")
    A: T.Tensor[[D0, D1, D2, D3, D4], dtype]
    C = T.empty((D0, D1, D2, D3, D4), dtype)

    with T.Kernel(dim0, dim1, dim2, threads=128) as (i, j, k):
        tile = T.alloc_shared((dim3, dim4), dtype)
        mbar = T.alloc_barrier(128)
        T.tma_copy(A[i, j, k, 0, 0], tile, barrier=mbar)
        T.barrier_arrive(mbar)
        T.mbarrier_wait_parity(mbar, 0)
        for l, m in T.grid(dim3, dim4):
            tile[l, m] = tile[l, m] + 1
        T.copy(tile, C[i, j, k, 0, 0])

    return C


TEST_CASES = [
    ("1d", tma_add_one_1d, (1024,)),
    ("2d", tma_add_one_2d, (32, 64)),
    ("3d", tma_add_one_3d, (8, 16, 32)),
    ("4d", tma_add_one_4d, (4, 8, 8, 8)),
    ("5d", tma_add_one_5d, (2, 4, 8, 4, 4)),
]


def ref_program(x):
    return x + 1


def _compile_kernel(kernel_fn, shape):
    compile_kwargs = {f"D{axis}": extent for axis, extent in enumerate(shape)}
    compile_kwargs.update({f"dim{axis}": extent for axis, extent in enumerate(shape)})
    compile_kwargs["dtype"] = "float32"
    return kernel_fn.compile(**compile_kwargs)


def _synchronize_device():
    if hasattr(torch, "musa") and torch.musa.is_available():
        torch.musa.synchronize()


def _cleanup_kernel(kernel):
    _synchronize_device()
    adapter = getattr(kernel, "adapter", None)
    if adapter is not None:
        adapter.func = None
    kernel.torch_function = None
    kernel.adapter = None
    gc.collect()


def _run_case(kernel_fn, shape):
    kernel = _compile_kernel(kernel_fn, shape)
    try:
        code = kernel.get_kernel_source()
        assert re.search(r"tl::tma_load", code), "tl::tma_load not found in generated code"
        assert re.search(r"tl::tma_store", code), "tl::tma_store not found in generated code"
        assert re.search(r"tl::tma_store_arrive\(\)", code), "tl::tma_store_arrive() not found after tl::tma_store in generated code"
        assert re.search(r"tl::tma_store_wait<0>\(\)", code), "tl::tma_store_wait<0>() not found after tl::tma_store in generated code"

        a = torch.randn(shape, device="musa", dtype=torch.float32)
        c = kernel(a)
        _synchronize_device()
        torch.testing.assert_close(c, ref_program(a), rtol=1e-6, atol=1e-6)
    finally:
        _cleanup_kernel(kernel)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "rank_name, kernel_fn, shape",
    [pytest.param(rank_name, kernel_fn, shape, id=rank_name) for rank_name, kernel_fn, shape in TEST_CASES],
)
def test_tma_load_store_nd(rank_name, kernel_fn, shape):
    del rank_name
    _run_case(kernel_fn, shape)


def _check_runtime_requirements():
    try:
        compute_version = mcc.parse_musa_compute_version(mcc.get_musa_compute_version())
    except ValueError:
        compute_version = (0, 0)
    return compute_version >= (3, 1), compute_version


def main():
    ok, compute_version = _check_runtime_requirements()
    if not ok:
        version_str = ".".join(str(v) for v in compute_version)
        print(f"skipped: Requires MUSA compute ge 3.1, but have {version_str}")
        return 0

    for rank_name, kernel_fn, shape in TEST_CASES:
        print(f"running {rank_name}: shape={shape}")
        _run_case(kernel_fn, shape)
    print("5 passed")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
