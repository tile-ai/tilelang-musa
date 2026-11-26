"""Tests for MP31 packed UINT8 TME loads of 4-bit (int4) tensors."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_packed_int4_load_and_store():
    M, K = 128, 128
    BM, BK = 64, 64

    @T.prim_func
    def copy_int4(
        Src: T.Tensor((M, K), "int4"),
        Dst: T.Tensor((M, K), "int4"),
    ):
        with T.Kernel(T.ceildiv(M, BM), threads=128) as bx:
            S = T.alloc_shared((BM, BK), "int4")
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                T.copy(Src[bx * BM, k * BK], S)
                T.copy(S, Dst[bx * BM, k * BK])

    kernel = tilelang.compile(
        copy_int4,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
    )
    src = kernel.get_kernel_source()
    assert "Src_desc" in src
    assert "tl::tme_load(" in src

    x = torch.randint(0, 256, (M, K // 2), dtype=torch.uint8, device="musa")
    y = kernel(x)
    torch.musa.synchronize()
    torch.testing.assert_close(y.view(torch.uint8).cpu(), x.cpu(), rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
