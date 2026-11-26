"""Tests for MUSA RNG DSL APIs: T.rng_init, T.rng_rand, T.rng_rand_float (issue #8)."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@pytest.mark.parametrize(
    "generator",
    [
        "curandStatePhilox4_32_10_t",
        "murandStatePhilox4_32_10_t",
        "curandStateXORWOW_t",
    ],
)
def test_musa_rng_rand_uint32_and_reproducibility(generator):
    @tilelang.jit(out_idx=[0], target="musa")
    def build(n: int = 1024, seed: int = 42):
        @T.prim_func
        def main(out: T.Tensor((n,), "uint32")):
            with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
                T.rng_init(seed, generator=generator)
                for i in T.Parallel(128):
                    idx = bx * 128 + i
                    if idx < n:
                        out[idx] = T.rng_rand()

        return main

    k_a = build(1024, 42)
    k_b = build(1024, 42)
    k_c = build(1024, 99)

    src = k_a.get_kernel_source()
    assert "#include <murand_kernel.h>" in src
    assert "murand_init(" in src
    assert "murand(&" in src

    out_a = k_a().cpu()
    out_b = k_b().cpu()
    out_c = k_c().cpu()

    assert torch.equal(out_a, out_b), "Same seed must produce identical uint32 outputs"
    assert not torch.equal(out_a, out_c), "Different seeds must produce distinct outputs"
    assert torch.unique(out_a.to(torch.int64)).numel() > 900


def test_musa_rng_rand_float_uniform_and_normal():
    @tilelang.jit(out_idx=[0, 1], target="musa")
    def build(n: int = 4096, seed: int = 17):
        @T.prim_func
        def main(
            out_u: T.Tensor((n,), "float32"),
            out_n: T.Tensor((n,), "float32"),
        ):
            with T.Kernel(T.ceildiv(n, 128), threads=128) as bx:
                T.rng_init(seed)
                # Ensure function-scope state survives a block barrier.
                T.sync_threads()
                for i in T.Parallel(128):
                    idx = bx * 128 + i
                    if idx < n:
                        out_u[idx] = T.rng_rand_float(32, "uniform")
                        out_n[idx] = T.rng_rand_float(32, "normal")

        return main

    k = build()
    src = k.get_kernel_source()
    assert "murand_uniform(&" in src
    assert "murand_normal(&" in src

    u, norm = k()
    torch.musa.synchronize()

    assert torch.isfinite(u).all() and torch.isfinite(norm).all()
    assert float(u.min()) > 0.0 and float(u.max()) <= 1.0
    assert 0.42 < float(u.mean()) < 0.58
    assert -0.15 < float(norm.mean()) < 0.15
    assert 0.85 < float(norm.std()) < 1.15


def test_musa_rng_inverted_dropout_issue8():
    @tilelang.jit(out_idx=[1], target="musa")
    def build_dropout(M: int = 16, N: int = 64, p: float = 0.5, seed: int = 17):
        scale = 1.0 / (1.0 - p)

        @T.prim_func
        def main(x: T.Tensor((M, N), "float32"), y: T.Tensor((M, N), "float32")):
            with T.Kernel(M, threads=N) as bx:
                T.rng_init(seed)
                for j in T.Parallel(N):
                    u = T.rng_rand_float()
                    y[bx, j] = T.if_then_else(u >= p, x[bx, j] * scale, 0.0)

        return main

    k = build_dropout()
    x = torch.ones((16, 64), device="musa", dtype=torch.float32)
    y1 = k(x)
    y2 = k(x)
    torch.musa.synchronize()

    assert torch.equal(y1, y2)
    zeros = (y1 == 0.0).sum().item()
    scaled = (y1 == 2.0).sum().item()
    assert zeros + scaled == 16 * 64
    assert 350 < zeros < 674


if __name__ == "__main__":
    tilelang.testing.main()
