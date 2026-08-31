import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang.musa.layout import make_mp31_sqmma_shared_ab


@pytest.mark.parametrize("k_major", [True, False])
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_tme_swizzle_load(k_major):
    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float16"),
        C: T.Tensor((64, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 128), "float16")
            mbar = T.alloc_barrier(128)
            T.annotate_layout({A_shared: make_mp31_sqmma_shared_ab(A_shared, k_major=k_major)})
            T.tma_copy(A, A_shared, barrier=mbar)
            T.barrier_arrive(mbar)
            T.barrier_wait(mbar, 0)
            T.copy(A_shared, C)

    kernel = tilelang.compile(
        main,
        out_idx=-1,
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    A = torch.randn((64, 128), dtype=torch.float16, device="musa")
    C = kernel(A)
    torch.testing.assert_close(C, A)


@pytest.mark.parametrize("k_major", [True, False])
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_tme_swizzle_store(k_major):
    @T.prim_func
    def main(
        A: T.Tensor((64, 128), "float16"),
        C: T.Tensor((64, 128), "float16"),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 128), "float16")
            T.annotate_layout({A_shared: make_mp31_sqmma_shared_ab(A_shared, k_major=k_major)})
            T.copy(A, A_shared)
            T.tma_copy(A_shared, C)
            T.tma_store_arrive()
            T.tma_store_wait(0)

    kernel = tilelang.compile(
        main,
        out_idx=-1,
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    A = torch.randn((64, 128), dtype=torch.float16, device="musa")
    C = kernel(A)
    torch.testing.assert_close(C, A)
