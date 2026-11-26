"""Tests for block-parallel T.sort, T.argsort, and T.topk on MUSA."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()

_TORCH_DTYPE = {
    "float32": torch.float32,
    "float16": torch.float16,
    "bfloat16": torch.bfloat16,
    "int32": torch.int32,
    "int64": torch.int64,
}


@pytest.mark.parametrize("dtype", ["float32", "float16", "bfloat16", "int32"])
@pytest.mark.parametrize("n,descending,use_frag", [
    (64, False, False),
    (256, True, False),
    (100, True, False),
    (128, True, True),
])
@tilelang.testing.requires_musa
def test_musa_sort_1d(dtype, n, descending, use_frag):
    @T.prim_func
    def kernel(X: T.Tensor((n,), dtype), Y: T.Tensor((n,), dtype)):
        with T.Kernel(1, threads=128):
            if use_frag:
                buf = T.alloc_fragment((n,), dtype)
                out = T.alloc_fragment((n,), dtype)
                T.copy(X, buf)
                T.sort(buf, out, dim=0, descending=descending)
                T.copy(out, Y)
            else:
                buf = T.alloc_shared((n,), dtype)
                T.copy(X, buf)
                T.sort(buf, dim=0, descending=descending)
                T.copy(buf, Y)

    compiled = tilelang.compile(kernel, out_idx=[1], target={"kind": "musa"})
    assert "#include <tl_templates/musa/common/sort.h>" in compiled.get_kernel_source()

    tdt = _TORCH_DTYPE[dtype]
    if dtype.startswith("int"):
        x = torch.randint(-50, 50, (n,), device="musa", dtype=tdt)
    else:
        x = torch.randn((n,), device="musa", dtype=tdt)
    y = compiled(x)
    torch.musa.synchronize()
    ref, _ = torch.sort(x.cpu().float() if dtype in ("float16", "bfloat16") else x.cpu(), descending=descending)
    torch.testing.assert_close(y.cpu().float() if dtype in ("float16", "bfloat16") else y.cpu(), ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("dim,descending", [(1, True), (1, False), (0, True)])
@tilelang.testing.requires_musa
def test_musa_sort_2d(dim, descending):
    H, W = 8, 64

    @T.prim_func
    def kernel(X: T.Tensor((H, W), "float32"), Y: T.Tensor((H, W), "float32")):
        with T.Kernel(1, threads=128):
            buf = T.alloc_shared((H, W), "float32")
            T.copy(X, buf)
            T.sort(buf, dim=dim, descending=descending)
            T.copy(buf, Y)

    compiled = tilelang.compile(kernel, out_idx=[1], target={"kind": "musa"})
    x = torch.randn((H, W), device="musa", dtype=torch.float32)
    y = compiled(x)
    torch.musa.synchronize()
    ref, _ = torch.sort(x.cpu(), dim=dim, descending=descending)
    torch.testing.assert_close(y.cpu(), ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("descending", [False, True])
@tilelang.testing.requires_musa
def test_musa_argsort_stable_ties(descending):
    M, N = 4, 100

    @T.prim_func
    def kernel(X: T.Tensor((M, N), "int32"), Idx: T.Tensor((M, N), "int32")):
        with T.Kernel(M, threads=128) as row:
            xs = T.alloc_shared((N,), "int32")
            is_ = T.alloc_shared((N,), "int32")
            T.copy(X[row, :], xs)
            T.argsort(xs, is_, dim=0, descending=descending)
            T.copy(is_, Idx[row, :])

    compiled = tilelang.compile(kernel, out_idx=[1], target={"kind": "musa"})
    # Small value range [-3, 3] forces many duplicate keys to stress tie stability.
    x = torch.randint(-3, 4, (M, N), device="musa", dtype=torch.int32)
    idx = compiled(x)
    torch.musa.synchronize()
    ref = torch.argsort(x.cpu(), dim=1, descending=descending, stable=True).to(torch.int32)
    torch.testing.assert_close(idx.cpu(), ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("N,K,largest", [
    (8, 2, True),       # MoE router top-2 of 8
    (512, 64, True),    # NSA / sampling top-64 of 512
    (100, 16, False),   # Non-pow2 smallest-16 of 100
])
@tilelang.testing.requires_musa
def test_musa_topk_1d_batched(N, K, largest):
    B = 8

    @T.prim_func
    def kernel(
        X: T.Tensor((B, N), "float32"),
        Vals: T.Tensor((B, K), "float32"),
        Idxs: T.Tensor((B, K), "int32"),
    ):
        with T.Kernel(B, threads=128) as row:
            xs = T.alloc_shared((N,), "float32")
            vs = T.alloc_shared((K,), "float32")
            is_ = T.alloc_shared((K,), "int32")
            T.copy(X[row, :], xs)
            T.topk(xs, vs, is_, k=K, dim=0, largest=largest)
            T.copy(vs, Vals[row, :])
            T.copy(is_, Idxs[row, :])

    compiled = tilelang.compile(kernel, out_idx=[1, 2], target={"kind": "musa"})
    # Use distinct floats so top-k indices are uniquely defined.
    x = torch.arange(B * N, device="musa", dtype=torch.float32)
    perm = torch.randperm(B * N, device="musa")
    x = x[perm].view(B, N)
    vals, idxs = compiled(x)
    torch.musa.synchronize()
    ref_v, ref_i = torch.topk(x.cpu(), k=K, dim=1, largest=largest, sorted=True)
    torch.testing.assert_close(vals.cpu(), ref_v, rtol=0.0, atol=0.0)
    torch.testing.assert_close(idxs.cpu(), ref_i.to(torch.int32), rtol=0.0, atol=0.0)


@tilelang.testing.requires_musa
def test_musa_topk_2d_and_values_only():
    H, W, K = 4, 64, 8

    @T.prim_func
    def kernel(X: T.Tensor((H, W), "float16"), Vals: T.Tensor((H, K), "float16")):
        with T.Kernel(1, threads=128):
            xs = T.alloc_shared((H, W), "float16")
            vs = T.alloc_shared((H, K), "float16")
            T.copy(X, xs)
            T.topk(xs, dst_values=vs, k=K, dim=1, largest=True)
            T.copy(vs, Vals)

    compiled = tilelang.compile(kernel, out_idx=[1], target={"kind": "musa"})
    x = torch.randn((H, W), device="musa", dtype=torch.float16)
    vals = compiled(x)
    torch.musa.synchronize()
    ref_v, _ = torch.topk(x.cpu().float(), k=K, dim=1, largest=True, sorted=True)
    torch.testing.assert_close(vals.cpu().float(), ref_v, rtol=0.0, atol=0.0)


if __name__ == "__main__":
    tilelang.testing.main()
