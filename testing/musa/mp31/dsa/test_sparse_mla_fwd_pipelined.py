import torch
import tilelang
import pytest
import tilelang.testing
from . import sparse_mla_decode_fwd_pipelined_v2 as decode_v2
from . import sparse_mla_decode_fwd_scheduled_v2 as decode_scheduled_v2
from . import sparse_mla_fwd_pipelined_v2 as prefill_v2
from .compare import check_is_allclose
from . import quant
import random

torch.random.manual_seed(42)


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("sq", [1, 2, 4])
@pytest.mark.parametrize("skv", [1024, 4096, 65536])
@pytest.mark.parametrize(
    "heads",
    [
        128,
        64,
    ],
)
@pytest.mark.parametrize(
    "hkv",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dqk",
    [
        576,
    ],
)
@pytest.mark.parametrize(
    "dv",
    [
        512,
    ],
)
@pytest.mark.parametrize(
    "topk",
    [
        2048,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
def test_dsa_decode(batch, sq, skv, heads, hkv, dqk, dv, topk, dtype):
    device = get_test_device()
    total_q = sq * batch
    q = torch.randn((total_q, heads, dqk), dtype=dtype, device=device) / 10 + (random.random() - 0.5) / 10
    page_size = 64
    page_num = (skv + page_size - 1) // page_size
    kv = torch.randn((page_num, page_size, hkv, dqk), dtype=dtype, device=device) / 10 + (random.random() - 0.5) / 10
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)

    indices = torch.full((total_q, hkv, topk), -1, dtype=torch.int32, device=device)
    for t in range(total_q):
        if random.random() < 0.8:
            for h in range(hkv):
                i_i = torch.randperm(skv, device=device)[:topk]
                indices[t, h, : len(i_i)] = i_i
    kcache = quant.quantize_k_cache(kv, quant.FP8KVCacheLayout.V32_FP8Sparse).contiguous()
    kv_dequant = quant.dequantize_k_cache(kcache, quant.FP8KVCacheLayout.V32_FP8Sparse).contiguous()

    tl_out, _ = decode_v2.tilelang_sparse_mla_fwd_interface(q, kcache.view(page_num * page_size, hkv, -1), indices)
    torch.musa.synchronize()
    tl_out_2, _ = decode_v2.tilelang_sparse_mla_fwd_interface(q, kcache.view(page_num * page_size, hkv, -1), indices)
    assert torch.equal(tl_out, tl_out_2)

    ref_out, _ = decode_v2.ref_sparse_mla_fwd_interface(q, kv_dequant.view(page_num * page_size, hkv, -1), indices)
    is_out_correct = check_is_allclose("output", tl_out, ref_out, abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6)
    assert is_out_correct


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("batch", [1, 4, 8])
@pytest.mark.parametrize("sq", [1, 2, 4])
@pytest.mark.parametrize("skv", [1024, 4096, 65536])
@pytest.mark.parametrize(
    "heads",
    [
        128,
        64,
    ],
)
@pytest.mark.parametrize(
    "hkv",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dqk",
    [
        576,
    ],
)
@pytest.mark.parametrize(
    "dv",
    [
        512,
    ],
)
@pytest.mark.parametrize(
    "topk",
    [
        2048,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
def test_dsa_decode_scheduled(batch, sq, skv, heads, hkv, dqk, dv, topk, dtype):
    device = get_test_device()
    total_q = sq * batch
    q = torch.randn((batch, sq, heads, dqk), dtype=dtype, device=device) / 10 + (random.random() - 0.5) / 10
    page_size = 64
    page_num = (skv + page_size - 1) // page_size
    kv = torch.randn((page_num, page_size, hkv, dqk), dtype=dtype, device=device) / 10 + (random.random() - 0.5) / 10
    cache_seqlens = torch.tensor([skv for i in range(batch)], dtype=torch.int32, device=device)
    indices = torch.full((batch, sq, hkv, topk), -1, dtype=torch.int32, device=device)
    q.clamp_(-10, 10)
    kv.clamp_(-10, 10)
    for b in range(batch):
        for t in range(sq):
            if random.random() < 0.8:
                for h in range(hkv):
                    i_i = torch.randperm(skv, device=device)[:topk]
                    indices[b, t, h, : len(i_i)] = i_i
    kcache = quant.quantize_k_cache(kv, quant.FP8KVCacheLayout.V32_FP8Sparse).contiguous()
    kv_dequant = quant.dequantize_k_cache(kcache, quant.FP8KVCacheLayout.V32_FP8Sparse).contiguous()
    tile_scheduler_metadata, num_splits = decode_scheduled_v2.get_mla_metadata_pytorch(
        cache_seqlens, num_q_tokens_per_head_k=sq * heads // 1, num_heads_k=1, num_heads_q=heads, topk=topk, mp_count=56, TILE_M=64
    )
    tl_out = decode_scheduled_v2.tilelang_flashmla_interface(
        q, kcache.view(page_num * page_size, hkv, -1), indices, tile_scheduler_metadata, num_splits
    )
    torch.musa.synchronize()
    tl_out_2 = decode_scheduled_v2.tilelang_flashmla_interface(
        q, kcache.view(page_num * page_size, hkv, -1), indices, tile_scheduler_metadata, num_splits
    )
    assert torch.equal(tl_out, tl_out_2)
    ref_out, _ = decode_scheduled_v2.ref_sparse_mla_fwd_interface(
        q.view(total_q, heads, dqk), kv_dequant.view(page_num * page_size, hkv, -1), indices.view(total_q, hkv, topk)
    )
    is_out_correct = check_is_allclose(
        "output", tl_out.view(-1), ref_out.to(device).view(-1), abs_tol=1e-3, rel_tol=2.01 / 128, cos_diff_tol=5e-6
    )
    assert is_out_correct


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("total_q", [32, 64, 128])
@pytest.mark.parametrize("skv", [1024, 4096])
@pytest.mark.parametrize("heads", [128, 64])
@pytest.mark.parametrize(
    "hkv",
    [
        1,
    ],
)
@pytest.mark.parametrize(
    "dqk",
    [
        576,
    ],
)
@pytest.mark.parametrize(
    "dv",
    [
        512,
    ],
)
@pytest.mark.parametrize(
    "topk",
    [
        2048,
    ],
)
@pytest.mark.parametrize(
    "dtype",
    [
        torch.bfloat16,
    ],
)
def test_dsa_prefill(total_q, skv, heads, hkv, dqk, dv, topk, dtype):
    device = get_test_device()
    q = torch.randn((total_q, heads, dqk), dtype=dtype, device=device)
    kv = torch.randn((skv, hkv, dqk), dtype=dtype, device=device)

    indices = torch.full((total_q, hkv, topk), -1, dtype=torch.int32, device=device)
    for t in range(total_q):
        for h in range(hkv):
            i_i = torch.randperm(max(1, t), device=device)[:topk]
            indices[t, h, : len(i_i)] = i_i
    tl_out, _ = prefill_v2.sparse_mla_fwd_interface(q, kv, indices)
    tl_out_2, _ = prefill_v2.sparse_mla_fwd_interface(q, kv, indices)
    torch.testing.assert_close(tl_out_2, tl_out, rtol=1e-7, atol=1e-7)
    ref_out = prefill_v2.ref_sparse_mla_fwd_interface(q, kv, indices)
    torch.testing.assert_close(tl_out, ref_out.to(device), rtol=1e-2, atol=1e-2)
