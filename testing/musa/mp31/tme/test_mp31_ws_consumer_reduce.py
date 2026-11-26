"""Tests for multi-warp T.reduce_* inside MP31 warp-specialized consumer branches."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_ws_consumer_multi_warp_reduce_partial_barrier():
    M, N, K_TILES = 16, 1024, 4
    TILE_N = N // K_TILES

    @T.prim_func
    def ws_pipelined_row_sum(
        X: T.Tensor((M, N), "float32"),
        Y: T.Tensor((M,), "float32"),
    ):
        with T.Kernel(M, threads=256) as bx:
            Xs = T.alloc_shared((1, TILE_N), "float32")
            Xf = T.alloc_fragment((1, TILE_N), "float32")
            Acc = T.alloc_fragment((1,), "float32")
            T.clear(Acc)
            for k in T.Pipelined(K_TILES, num_stages=2):
                T.copy(X[bx : bx + 1, k * TILE_N : (k + 1) * TILE_N], Xs)
                T.copy(Xs, Xf)
                T.reduce_sum(Xf, Acc, dim=1, clear=False)
            T.copy(Acc, Y[bx : bx + 1])

    kernel = tilelang.compile(
        ws_pipelined_row_sum,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
    )
    src = kernel.get_kernel_source()
    assert "tl::mp31_allreduce_" in src
    x = torch.randn((M, N), dtype=torch.float32, device="musa")
    y = kernel(x)
    torch.musa.synchronize()
    torch.testing.assert_close(y.cpu(), x.cpu().sum(dim=1), rtol=1e-4, atol=1e-4)


if __name__ == "__main__":
    tilelang.testing.main()
