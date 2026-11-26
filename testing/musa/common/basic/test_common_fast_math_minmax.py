"""Tests for MUSA tl.enable_fast_math and fminf/fmaxf codegen."""

import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@tilelang.testing.requires_musa
def test_musa_fminf_fmaxf_and_fast_math():
    @T.prim_func
    def kernel(
        A: T.Tensor((256,), "float32"),
        B: T.Tensor((256,), "float32"),
        OutMax: T.Tensor((256,), "float32"),
        OutMin: T.Tensor((256,), "float32"),
    ):
        with T.Kernel(1, threads=64):
            for i in T.Parallel(256):
                OutMax[i] = T.max(A[i], B[i])
                OutMin[i] = T.min(A[i], B[i])

    compiled = tilelang.compile(
        kernel,
        out_idx=[2, 3],
        target={"kind": "musa"},
        pass_configs={"tl.enable_fast_math": True},
    )
    src = compiled.get_kernel_source()
    assert "fmaxf(" in src
    assert "fminf(" in src

    a = torch.randn((256,), device="musa", dtype=torch.float32)
    b = torch.randn((256,), device="musa", dtype=torch.float32)
    out_max, out_min = compiled(a, b)
    torch.musa.synchronize()
    torch.testing.assert_close(out_max.cpu(), torch.maximum(a.cpu(), b.cpu()), rtol=0, atol=0)
    torch.testing.assert_close(out_min.cpu(), torch.minimum(a.cpu(), b.cpu()), rtol=0, atol=0)
