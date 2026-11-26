"""Tests for MUSA __launch_bounds__ emission and T.annotate_min_blocks_per_sm."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


def test_musa_default_launch_bounds_high_threads():
    @tilelang.jit(target="musa")
    def kernel(n: int = 1024):
        @T.prim_func
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            with T.Kernel(T.ceildiv(n, 512), threads=512) as bx:
                for i in T.Parallel(512):
                    idx = bx * 512 + i
                    if idx < n:
                        B[idx] = A[idx] + 1.0

        return main

    compiled = kernel()
    src = compiled.get_kernel_source()
    assert "__launch_bounds__(512)" in src

    a = torch.randn(1024, device="musa", dtype=torch.float32)
    b = torch.empty_like(a)
    compiled(a, b)
    torch.musa.synchronize()
    torch.testing.assert_close(b, a + 1.0)


def test_musa_default_launch_bounds_256_threads():
    @tilelang.jit(target="musa")
    def kernel(n: int = 1024):
        @T.prim_func
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            with T.Kernel(T.ceildiv(n, 256), threads=256) as bx:
                for i in T.Parallel(256):
                    idx = bx * 256 + i
                    if idx < n:
                        B[idx] = A[idx] + 1.0

        return main

    compiled = kernel()
    src = compiled.get_kernel_source()
    assert "__launch_bounds__(256)" in src

    a = torch.randn(1024, device="musa", dtype=torch.float32)
    b = torch.empty_like(a)
    compiled(a, b)
    torch.musa.synchronize()
    torch.testing.assert_close(b, a + 1.0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_musa_launch_bounds_omitted_for_low_thread_cp_async_sqmma():
    @tilelang.jit(target={"kind": "musa", "arch": "mp_31"}, pass_configs={"tl.disable_warp_specialized": True})
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor((64, 64), "float16"),
            B: T.Tensor((64, 64), "float16"),
            C: T.Tensor((64, 64), "float32"),
        ):
            with T.Kernel(1, 1, threads=128):
                As = T.alloc_shared((64, 64), "float16")
                Bs = T.alloc_shared((64, 64), "float16")
                Cf = T.alloc_fragment((64, 64), "float32")
                T.clear(Cf)
                T.async_copy(A[0:64, 0:64], As)
                T.async_copy(B[0:64, 0:64], Bs)
                T.ptx_wait_group(0)
                T.sync_threads()
                T.gemm(As, Bs, Cf, transpose_B=True)
                T.copy(Cf, C)

        return main

    compiled = kernel()
    src = compiled.get_kernel_source()
    assert "__launch_bounds__" not in src

    a = torch.randn((64, 64), device="musa", dtype=torch.float16) * 0.1
    b = torch.randn((64, 64), device="musa", dtype=torch.float16) * 0.1
    c = torch.empty((64, 64), device="musa", dtype=torch.float32)
    compiled(a, b, c)
    torch.musa.synchronize()
    torch.testing.assert_close(c.cpu(), (a.cpu().float() @ b.cpu().float().T), rtol=1e-2, atol=1e-2)


def test_musa_annotate_min_blocks_per_sm():
    @tilelang.jit(target="musa")
    def kernel(n: int = 1024):
        @T.prim_func
        def main(A: T.Tensor((n,), "float32"), B: T.Tensor((n,), "float32")):
            with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
                T.annotate_min_blocks_per_sm(2)
                for i in T.Parallel(128):
                    idx = bx * 128 + i
                    if idx < n:
                        B[idx] = A[idx] * 2.0

        return main

    compiled = kernel()
    src = compiled.get_kernel_source()
    assert "__launch_bounds__(128, 2)" in src

    a = torch.randn(1024, device="musa", dtype=torch.float32)
    b = torch.empty_like(a)
    compiled(a, b)
    torch.musa.synchronize()
    torch.testing.assert_close(b, a * 2.0)


if __name__ == "__main__":
    tilelang.testing.main()
