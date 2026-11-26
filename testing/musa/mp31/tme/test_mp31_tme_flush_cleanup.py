"""Tests for MP31 TME exit drain cleanup (tl::tme_flush)."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_load_emits_exit_flush():
    @tilelang.jit(out_idx=[1], target="musa")
    def build():
        @T.prim_func
        def main(A: T.Tensor((128, 64), "float16"), B: T.Tensor((128, 64), "float16")):
            with T.Kernel(1, threads=128):
                A_shared = T.alloc_shared((128, 64), "float16")
                mbar = T.alloc_barrier(128)
                T.tma_copy(A[0, 0], A_shared, barrier=mbar)
                T.barrier_arrive(mbar)
                T.barrier_wait(mbar, 0)
                T.copy(A_shared, B[0, 0])

        return main

    kernel = build()
    src = kernel.get_kernel_source()
    assert "tl::tme_load(" in src
    assert "tl::tme_flush();" in src

    x = torch.randn((128, 64), dtype=torch.float16, device="musa")
    y = kernel(x)
    torch.musa.synchronize()
    torch.testing.assert_close(y, x)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_store_emits_exit_flush():
    @tilelang.jit(out_idx=[1], target="musa")
    def build():
        @T.prim_func
        def main(A: T.Tensor((128, 64), "float16"), B: T.Tensor((128, 64), "float16")):
            with T.Kernel(1, threads=128):
                A_shared = T.alloc_shared((128, 64), "float16")
                T.copy(A[0, 0], A_shared)
                T.sync_threads()
                T.tma_copy(A_shared, B[0, 0])
                T.tma_store_wait(0)

        return main

    kernel = build()
    src = kernel.get_kernel_source()
    assert "tl::tme_store(" in src
    assert "tl::tme_flush();" in src

    x = torch.randn((128, 64), dtype=torch.float16, device="musa")
    y = kernel(x)
    torch.musa.synchronize()
    torch.testing.assert_close(y, x)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_non_tme_kernel_omits_flush():
    @tilelang.jit(out_idx=[1], target="musa")
    def build():
        @T.prim_func
        def main(A: T.Tensor((256,), "float32"), B: T.Tensor((256,), "float32")):
            with T.Kernel(1, threads=128):
                for i in T.Parallel(256):
                    B[i] = A[i] + 1.0

        return main

    kernel = build()
    src = kernel.get_kernel_source()
    assert "tl::tme_flush" not in src

    x = torch.randn((256,), dtype=torch.float32, device="musa")
    y = kernel(x)
    torch.musa.synchronize()
    torch.testing.assert_close(y, x + 1.0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_symbolic_shape_emits_device_flush_only():
    M = T.dynamic("M")
    N, K = 256, 128
    BM, BN, BK = 128, 128, 64

    @T.prim_func
    def sym_gemm(
        A: T.Tensor((M, K), "float16"),
        B: T.Tensor((N, K), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=256) as (bx, by):
            As = T.alloc_shared((BM, BK), "float16")
            Bs = T.alloc_shared((BN, BK), "float16")
            Cl = T.alloc_fragment((BM, BN), "float32")
            T.clear(Cl)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                T.copy(A[by * BM, k * BK], As)
                T.copy(B[bx * BN, k * BK], Bs)
                T.gemm(As, Bs, Cl, transpose_B=True)
            T.copy(Cl, C[by * BM, bx * BN])

    kernel = tilelang.compile(sym_gemm, out_idx=[2], target={"kind": "musa", "arch": "mp_31"})
    src = kernel.get_kernel_source()
    assert "tl::tme_load(" in src
    assert "tl::tme_flush();" in src

    for m_val in (200, 256):
        a = torch.randn((m_val, K), dtype=torch.float16, device="musa") * 0.1
        b = torch.randn((N, K), dtype=torch.float16, device="musa") * 0.1
        out = kernel(a, b)
        torch.musa.synchronize()
        ref = torch.matmul(a.float().cpu(), b.float().cpu().T).to(torch.float16)
        torch.testing.assert_close(out.cpu(), ref, rtol=2e-2, atol=2e-2)


if __name__ == "__main__":
    tilelang.testing.main()
