"""Tests for the automatic MP31 ``T.gemm`` interface."""

import torch

import tilelang
import tilelang.language as T
import tilelang.testing


_TARGET = {"kind": "musa", "arch": "mp_31"}


def _assert_matches_torch(kernel):
    A = torch.randn((64, 16), dtype=torch.float16, device="musa")
    B = torch.randn((16, 64), dtype=torch.float16, device="musa")
    expected = A.cpu().float() @ B.cpu().float()
    actual = kernel(A, B).cpu()
    torch.testing.assert_close(actual, expected, rtol=1e-3, atol=1e-3)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_gemm_shared_operands_selects_sqmma():
    """Shared/shared T.gemm lowers through the existing SQMMA implementation."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        C: T.Tensor((64, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_shared = T.alloc_shared((64, 16), T.float16)
            B_shared = T.alloc_shared((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.gemm(A_shared, B_shared, C_local, clear_accum=True)
            T.copy(C_local, C)

    kernel = tilelang.compile(main, target=_TARGET, execution_backend="tvm_ffi", out_idx=[2])
    assert "tl::sqmma_ss<" in kernel.get_kernel_source()
    _assert_matches_torch(kernel)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_gemm_fragment_operands_selects_wmma():
    """Fragment/fragment T.gemm lowers through the existing WMMA implementation."""

    @T.prim_func
    def main(
        A: T.Tensor((64, 16), T.float16),
        B: T.Tensor((16, 64), T.float16),
        C: T.Tensor((64, 64), T.float32),
    ):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_fragment((64, 16), T.float16)
            B_local = T.alloc_fragment((16, 64), T.float16)
            C_local = T.alloc_fragment((64, 64), T.float32)
            T.copy(A, A_local)
            T.copy(B, B_local)
            T.gemm(A_local, B_local, C_local, clear_accum=True)
            T.copy(C_local, C)

    kernel = tilelang.compile(main, target=_TARGET, execution_backend="tvm_ffi", out_idx=[2])
    assert "tl::wmma_rr<" in kernel.get_kernel_source()
    _assert_matches_torch(kernel)
