"""Shared epilogues must keep wide copies around strided SQMMA registers."""

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@tilelang.jit(target="musa")
def shared_fragment_update(K, H, V, G, columns=128, dtype="float16", mode="v2"):
    K: T.Tensor((64, 128), dtype)
    H: T.Tensor((128, columns), dtype)
    V: T.Tensor((64, columns), dtype)
    G: T.Tensor((64,), T.float32)
    output = T.empty((64, columns), dtype)
    with T.Kernel(1, threads=128):
        k_shared = T.alloc_shared((64, 128), dtype)
        h_shared = T.alloc_shared((128, columns), dtype)
        v_shared = T.alloc_shared((64, columns), dtype)
        g_shared = T.alloc_shared((64,), T.float32)
        uv_fragment = T.alloc_fragment((64, columns), T.float32)
        T.copy(K, k_shared)
        T.copy(H, h_shared)
        T.copy(V, v_shared)
        T.copy(G, g_shared)
        T.gemm(k_shared, h_shared, uv_fragment, clear_accum=True)
        if mode == "v1":
            for i, j in T.Parallel(64, columns):
                v_shared[i, j % (columns // 8) * 8 + j // (columns // 8)] -= g_shared[i] * uv_fragment[i, j]
        else:
            for i, j in T.Parallel(64, columns):
                v_shared[i, j] -= g_shared[i] * uv_fragment[i, j % 8 * (columns // 8) + j // 8]
        T.copy(v_shared, output)
    return output


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
@pytest.mark.parametrize("mode", ["v1", "v2"])
@pytest.mark.parametrize("columns", [128, 256])
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_shared_fragment_update(mode, columns, dtype):
    kernel = shared_fragment_update.compile(columns=columns, dtype=dtype, mode=mode)
    source = kernel.get_kernel_source()
    assert "g_shared_local_cast" not in source, "Row broadcasts should remain scalar loads"
    prefix = "tl_h" if dtype == "float16" else "tl_bf"
    vector_types = [f"{prefix}{width}" for width in ([8] if columns == 128 else [4, 8])]
    shared_lines = [line for line in source.splitlines() if "*)v_shared)" in line]
    assert any(f"= *({vector_type}*)" in line for line in shared_lines for vector_type in vector_types), source
    assert any(line.lstrip().startswith(f"*({vector_type}*)") for line in shared_lines for vector_type in vector_types), source
    generator = torch.Generator().manual_seed(20261008)
    torch_dtype = getattr(torch, dtype)
    k = (torch.randn((64, 128), generator=generator) * 0.125).to(torch_dtype)
    h = (torch.randn((128, columns), generator=generator) * 0.125).to(torch_dtype)
    v = torch.randn((64, columns), generator=generator).to(torch_dtype)
    g = torch.randn((64,), generator=generator) * 0.25
    permutation = torch.arange(columns) % 8 * (columns // 8) + torch.arange(columns) // 8
    reference = (v.float() - g[:, None] * (k.float() @ h.float())[:, permutation]).to(torch_dtype)
    actual = kernel(*(value.to("musa") for value in (k, h, v, g))).cpu()
    tolerance = 1e-3 if dtype == "float16" else 1e-2
    torch.testing.assert_close(actual, reference, rtol=tolerance, atol=tolerance)
