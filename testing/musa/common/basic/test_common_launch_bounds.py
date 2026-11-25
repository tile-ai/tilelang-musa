"""Tests for MUSA __launch_bounds__ emission and T.annotate_min_blocks_per_sm."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


def test_musa_default_launch_bounds():
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
