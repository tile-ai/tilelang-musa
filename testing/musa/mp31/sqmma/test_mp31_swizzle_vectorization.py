"""Regression tests for element-unit MP31 swizzle layout vectorization."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@pytest.mark.parametrize(
    ("bm", "bn", "bk", "threads"),
    [
        (128, 128, 64, 256),
        (256, 128, 64, 512),
    ],
)
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_swizzle_layout_enables_16b_async_copy(bm, bn, bk, threads):
    M = N = K = 512

    @tilelang.jit(
        target="musa",
        pass_configs={"tl.disable_warp_specialized": True},
    )
    def gemm():
        @T.prim_func
        def main(
            A: T.Tensor((M, K), "float16"),
            B: T.Tensor((N, K), "float16"),
            C: T.Tensor((M, N), "float16"),
        ):
            with T.Kernel(T.ceildiv(N, bn), T.ceildiv(M, bm), threads=threads) as (bx, by):
                A_shared = T.alloc_shared((bm, bk), "float16")
                B_shared = T.alloc_shared((bn, bk), "float16")
                C_local = T.alloc_fragment((bm, bn), "float32")
                T.clear(C_local)
                for k in T.Pipelined(T.ceildiv(K, bk), num_stages=2):
                    T.copy(A[by * bm, k * bk], A_shared)
                    T.copy(B[bx * bn, k * bk], B_shared)
                    T.gemm(A_shared, B_shared, C_local, transpose_B=True)
                T.copy(C_local, C[by * bm, bx * bn])

        return main

    kernel = gemm()
    src = kernel.get_kernel_source()
    assert "tl::cp_async_gs<16>" in src, (
        "Expected 16-byte vectorized cp.async copy into swizzled SQMMA shared buffers"
    )

    A = torch.randn((M, K), device="musa", dtype=torch.float16) * 0.1
    B = torch.randn((N, K), device="musa", dtype=torch.float16) * 0.1
    C = torch.empty((M, N), device="musa", dtype=torch.float16)
    kernel(A, B, C)
    torch.musa.synchronize()
    ref = torch.matmul(A, B.T)
    torch.testing.assert_close(C, ref, rtol=1e-2, atol=1e-2)


if __name__ == "__main__":
    tilelang.testing.main()
