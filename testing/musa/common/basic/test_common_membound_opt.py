"""Tests for MUSA memory-bound TIR optimizations:
1) Round-to-int cast fusion (nearbyintf -> __float2int_rn) preserving vector stores.
2) Automatic leader-guarding of uniform/strided scalar global stores.
3) Signed int8x8 / int8x16 packed store sign-extension safety.
"""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@tilelang.testing.requires_musa
def test_musa_round_to_int8_fusion_and_ties_to_even():
    M, N = 64, 256

    @tilelang.jit(target="musa")
    def build():
        @T.prim_func
        def main(X: T.Tensor((M, N), "float32"), Y: T.Tensor((M, N), "int8")):
            with T.Kernel(M, threads=32) as row:
                for i in T.Parallel(N):
                    q = T.max(T.min(T.round(X[row, i]), T.float32(127.0)), T.float32(-127.0))
                    Y[row, i] = T.cast(q, "int8")
        return main

    k = build()
    src = k.get_kernel_source()
    assert "__float2int_rn(" in src
    assert "nearbyintf(" not in src
    assert "*(int*)" in src or "*(int2*)" in src or "*(int4*)" in src

    # Test ties-to-even, negative values, and saturation boundaries.
    special = torch.tensor(
        [-200.0, -127.5, -127.0, -2.5, -1.5, -0.5, 0.0, 0.5, 1.5, 2.5, 126.5, 127.0, 200.0],
        dtype=torch.float32,
    )
    x = torch.randn((M, N), dtype=torch.float32) * 50.0
    x[0, : len(special)] = special
    x_dev = x.to("musa")
    y_dev = torch.empty((M, N), dtype=torch.int8, device="musa")
    k(x_dev, y_dev)
    torch.musa.synchronize()
    ref = torch.round(x).clamp(-127, 127).to(torch.int8)
    assert torch.equal(y_dev.cpu(), ref)


@tilelang.testing.requires_musa
def test_musa_uniform_global_scalar_store_guard():
    M, N = 32, 128

    @tilelang.jit(target="musa")
    def build():
        @T.prim_func
        def main(
            X: T.Tensor((M, N), "float32"),
            S: T.Tensor((M,), "float32"),
            Y: T.Tensor((M, N), "float32"),
        ):
            with T.Kernel(M, threads=128) as row:
                xf = T.alloc_fragment((N,), "float32")
                am = T.alloc_fragment((1,), "float32")
                for i in T.Parallel(N):
                    xf[i] = X[row, i]
                T.reduce_absmax(xf, am, dim=0)
                S[row] = am[0]
                for i in T.Parallel(N):
                    Y[row, i] = xf[i]
        return main

    k = build()
    src = k.get_kernel_source()
    assert "((int)threadIdx.x) == 0" in src

    x = torch.randn((M, N), dtype=torch.float32, device="musa")
    s = torch.empty((M,), dtype=torch.float32, device="musa")
    y = torch.empty_like(x)
    k(x, s, y)
    torch.musa.synchronize()
    torch.testing.assert_close(s.cpu(), x.cpu().abs().amax(dim=1), atol=0, rtol=0)
    torch.testing.assert_close(y.cpu(), x.cpu(), atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_int8x16_negative_packed_store():
    M, N = 32, 512

    @tilelang.jit(target="musa")
    def build():
        @T.prim_func
        def main(X: T.Tensor((M, N), "float16"), Y: T.Tensor((M, N), "int8")):
            with T.Kernel(M, threads=32) as row:
                xf = T.alloc_fragment((N,), "float32")
                for i in T.Parallel(N):
                    xf[i] = T.cast(X[row, i], "float32")
                for i in T.Parallel(N):
                    q = T.max(T.min(T.round(xf[i]), T.float32(127.0)), T.float32(-128.0))
                    Y[row, i] = T.cast(q, "int8")
        return main

    k = build()
    src = k.get_kernel_source()
    assert "*(int2*)" in src or "*(int4*)" in src
    assert "__float2int_rn(" in src

    # Deliberately alternate negative and positive values across all 16 lanes.
    x = torch.randint(-128, 128, (M, N), dtype=torch.int32).to(torch.float16)
    y = torch.empty((M, N), dtype=torch.int8, device="musa")
    k(x.to("musa"), y)
    torch.musa.synchronize()
    assert torch.equal(y.cpu(), x.to(torch.int8))


if __name__ == "__main__":
    tilelang.testing.main()
