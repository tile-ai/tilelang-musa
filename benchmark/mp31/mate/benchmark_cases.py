from __future__ import annotations

import argparse
import copy
import json
import math
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .benchmark_common import (
    BASELINE_FILENAME,
    DEFAULT_THRESHOLD,
    TermStyle,
    benchmark_root,
    benchmark_timer,
    check_regression,
    count_bytes,
    detect_tilelang_musa_build_type,
    ensure_release_build,
    get_test_device,
    load_baselines,
    print_banner,
    print_json_record,
    print_perf,
    print_summary,
    style,
)

# sweep parameters
SPARSE_MLA_DV = 512
GDN_DECODE_DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
GDN_DECODE_DEFAULT_HEAD_CONFIGS = ((8, 16), (16, 32), (16, 64))
GDN_DECODE_DEFAULT_HEAD_SIZE = 128
GDN_MTP_DEFAULT_BATCH_SIZES = (1, 2, 4, 8, 16, 32, 64, 128, 256, 512)
GDN_MTP_DEFAULT_SEQ_LENS = (2, 3, 4, 8)
GDN_MTP_DEFAULT_HEAD_CONFIGS = ((8, 16), (16, 32), (16, 48), (16, 64))
GDN_MTP_DEFAULT_HEAD_SIZE = 128
GDN_PREFILL_HEAD_CONFIGS = (
    (2, 8, 128, "397b_122b_tp8"),
    (4, 16, 128, "397b_122b_tp4"),
    (8, 32, 128, "397b_122b_tp2"),
    (16, 64, 128, "397b_122b_tp1"),
    (16, 32, 128, "35b_9b_4b_tp1"),
    (16, 48, 128, "27b_tp1"),
    (16, 16, 128, "2b_0p8b_tp1"),
    (32, 32, 128, "sym_h32"),
)
GDN_PREFILL_SEQ_CONFIGS = (
    ((8192,), "1x8192"),
    ((4096,), "1x4096"),
    ((2048,), "1x2048"),
    ((1024 * 6, 8192), "6144_2048"),
    ((1024 * 4, 8192), "4096_4096"),
    ((1024 * 2, 8192), "2048_6144"),
    ((1024, 8192), "1024_7168"),
    ((2048, 2048 * 2, 2048 * 3, 8192), "2048x4"),
    (tuple(1024 * (i + 1) for i in range(8)), "1024x8"),
)


@dataclass(frozen=True)
class CaseSpec:
    name: str
    runner: str
    args: tuple[Any, ...]


@dataclass(frozen=True)
class BenchmarkRunResult:
    records: list[dict[str, Any]]
    regression_stats: dict[str, Any] | None
    exit_code: int


def build_gdn_decode_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            f"gdn_decode_b{batch_size}_h{num_q_heads}_{num_v_heads}_d{GDN_DECODE_DEFAULT_HEAD_SIZE}_bf16",
            "gdn_decode",
            (batch_size, num_q_heads, num_v_heads, GDN_DECODE_DEFAULT_HEAD_SIZE),
        )
        for num_q_heads, num_v_heads in GDN_DECODE_DEFAULT_HEAD_CONFIGS
        for batch_size in GDN_DECODE_DEFAULT_BATCH_SIZES
    ]


def _gdn_mtp_case_name(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
) -> str:
    return f"gdn_mtp_b{batch_size}_t{seq_len}_h{num_q_heads}_{num_v_heads}_d{head_size}_bf16"


def build_gdn_mtp_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            _gdn_mtp_case_name(batch_size, seq_len, num_q_heads, num_v_heads, GDN_MTP_DEFAULT_HEAD_SIZE),
            "gdn_mtp",
            (
                batch_size,
                seq_len,
                num_q_heads,
                num_v_heads,
                GDN_MTP_DEFAULT_HEAD_SIZE,
            ),
        )
        for num_q_heads, num_v_heads in GDN_MTP_DEFAULT_HEAD_CONFIGS
        for seq_len in GDN_MTP_DEFAULT_SEQ_LENS
        for batch_size in GDN_MTP_DEFAULT_BATCH_SIZES
    ]


def gdn_mtp_case_names() -> list[str]:
    return [case.name for case in build_gdn_mtp_cases()]


def _gdn_prefill_case_name(head_label: str, seq_label: str, dtype_label: str) -> str:
    return f"gdn_prefill_{head_label}_{seq_label}_{dtype_label}"


def build_gdn_prefill_cases() -> list[CaseSpec]:
    return [
        CaseSpec(
            _gdn_prefill_case_name(head_label, seq_label, "bf16"),
            "gdn_prefill",
            (endpoints, num_q_heads, num_v_heads, head_size),
        )
        for num_q_heads, num_v_heads, head_size, head_label in GDN_PREFILL_HEAD_CONFIGS
        for endpoints, seq_label in GDN_PREFILL_SEQ_CONFIGS
    ]


def gdn_prefill_case_names() -> list[str]:
    return [case.name for case in build_gdn_prefill_cases()]


def _dtype_size(dtype: torch.dtype) -> int:
    return torch.empty((), dtype=dtype).element_size()


def _gdn_mtp_flops(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
) -> int:
    num_o_heads = max(num_q_heads, num_v_heads)
    return 6 * batch_size * seq_len * num_o_heads * head_size * head_size


def _gdn_mtp_bytes(
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
    *,
    disable_state_update: bool,
    cache_intermediate_states: bool,
) -> int:
    num_o_heads = max(num_q_heads, num_v_heads)
    elem_size = _dtype_size(input_dtype)
    state_bytes = batch_size * num_v_heads * head_size * head_size * 4
    intermediate_bytes = batch_size * seq_len * num_v_heads * head_size * head_size * 4 if cache_intermediate_states else 0
    return (
        batch_size * seq_len * num_q_heads * head_size * elem_size
        + batch_size * seq_len * num_q_heads * head_size * elem_size
        + batch_size * seq_len * num_v_heads * head_size * elem_size
        + batch_size * seq_len * num_o_heads * head_size * _dtype_size(output_dtype)
        + state_bytes
        + (0 if disable_state_update else state_bytes)
        + intermediate_bytes
        + num_v_heads * 4
        + num_v_heads * elem_size
        + batch_size * seq_len * num_v_heads * elem_size
        + batch_size * seq_len * num_v_heads * elem_size
    )


def _gdn_decode_flops(batch_size: int, num_q_heads: int, num_v_heads: int, head_size: int) -> int:
    num_o_heads = max(num_q_heads, num_v_heads)
    return 6 * batch_size * num_o_heads * head_size * head_size


def _gdn_decode_bytes(
    batch_size: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    input_dtype: torch.dtype,
    output_dtype: torch.dtype,
) -> int:
    num_o_heads = max(num_q_heads, num_v_heads)
    elem_size = _dtype_size(input_dtype)
    return (
        batch_size * num_q_heads * head_size * elem_size
        + batch_size * num_q_heads * head_size * elem_size
        + batch_size * num_v_heads * head_size * elem_size
        + batch_size * num_o_heads * head_size * _dtype_size(output_dtype)
        + 2 * batch_size * num_v_heads * head_size * head_size * 4
        + num_v_heads * 4
        + num_v_heads * 4
        + batch_size * num_v_heads * elem_size
        + batch_size * num_v_heads * elem_size
    )


def benchmark_gdn_decode_case(
    *,
    batch_size: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    from .kernels import gdn_decode

    torch.manual_seed(20260522 + batch_size + num_q_heads + num_v_heads)
    q = torch.randn((batch_size, 1, num_q_heads, head_size), dtype=dtype, device=device)
    k = torch.randn((batch_size, 1, num_q_heads, head_size), dtype=dtype, device=device)
    v = torch.randn((batch_size, 1, num_v_heads, head_size), dtype=dtype, device=device)
    state = torch.randn(
        (batch_size, num_v_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )
    A_log = torch.randn((num_v_heads,), dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn((num_v_heads,), dtype=torch.float32, device=device) * 0.1
    a = torch.randn((batch_size, 1, num_v_heads), dtype=dtype, device=device) * 0.1
    b = torch.randn((batch_size, 1, num_v_heads), dtype=dtype, device=device)
    output = torch.empty((batch_size, 1, num_v_heads, head_size), dtype=dtype, device=device)

    def fn() -> None:
        gdn_decode.run_gated_delta_rule_decode_vk_fp32(
            q=q,
            k=k,
            v=v,
            state=state,
            A_log=A_log,
            a=a,
            dt_bias=dt_bias,
            b=b,
            output=output,
            scale=head_size**-0.5,
            use_qk_l2norm=True,
        )

    fn()
    time_us = benchmark_timer(fn)
    flops = _gdn_decode_flops(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
    )
    bytes_rw = _gdn_decode_bytes(
        batch_size=batch_size,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        input_dtype=dtype,
        output_dtype=dtype,
    )
    kernel_config = gdn_decode._resolve_autotuned_kernel_config(batch_size)
    return {
        "kernel": "mate/gdn_decode",
        "operation": "gated_deltanet_decode_fp32_vk",
        "params": {
            "B": batch_size,
            "Hq": num_q_heads,
            "Hv": num_v_heads,
            "D": head_size,
            "dtype": str(dtype).split(".")[-1],
            **kernel_config,
        },
        "time_us": time_us,
        "bandwidth_gbs": bytes_rw / time_us / 1e3,
        "extras": {
            "bytes_rw": bytes_rw,
            "flops": flops,
            "tflops": flops / time_us / 1e6,
        },
    }


def _make_sparse_indices(rows: int, topk: int, seq_len_kv: int, device: str) -> torch.Tensor:
    indices = torch.full((rows, 1, topk), -1, dtype=torch.int32, device=device)
    valid = min(topk, seq_len_kv)
    base = torch.arange(valid, dtype=torch.int32, device=device)
    for row in range(rows):
        offset = (row * 131) % max(seq_len_kv, 1)
        indices[row, 0, :valid] = (base + offset) % seq_len_kv
    return indices


def _make_sparse_temp_prefill_indices(
    seq_len_q: int,
    topk: int,
    seq_len_kv: int,
    device: str,
) -> torch.Tensor:
    indices = torch.full((seq_len_q, 1, topk), -1, dtype=torch.int32, device=device)
    for row in range(seq_len_q):
        valid = torch.randperm(max(1, min(row, seq_len_kv)), device=device)[:topk]
        indices[row, 0, : valid.numel()] = valid
    return indices


def _make_sparse_temp_decode_indices(
    batch_size: int,
    seq_len_q: int,
    topk: int,
    cache_seqlens: torch.Tensor,
    cu_seqlens: torch.Tensor,
    device: str,
) -> torch.Tensor:
    indices = torch.full((batch_size, seq_len_q, 1, topk), -1, dtype=torch.int32, device=device)
    for batch in range(batch_size):
        cur_len = int(cache_seqlens[batch].item())
        base = int(cu_seqlens[batch].item())
        for row in range(seq_len_q):
            valid = torch.randperm(cur_len, device=device)[:topk] + base
            indices[batch, row, 0, : valid.numel()] = valid
    return indices


def _make_sparse_decode_metadata(
    *,
    seqlens_k: torch.Tensor,
    num_q_tokens_per_head_k: int,
    num_heads_k: int,
    topk: int,
    mp_count: int = 56,
    block_size_n: int = 64,
    fixed_overhead_num_blocks: int = 5,
    tile_m: int = 64,
) -> tuple[torch.Tensor, torch.Tensor]:
    """Local equivalent of MATE's temp get_mla_metadata_pytorch helper."""
    device = seqlens_k.device
    batch_size = int(seqlens_k.shape[0])
    q_tiles = (num_q_tokens_per_head_k + tile_m - 1) // tile_m
    num_mp_parts = max(mp_count // num_heads_k // q_tiles, 1)
    effective_seqlens = torch.full((batch_size,), topk, dtype=torch.int32, device=device)

    num_blocks_list: list[int] = []
    first_block_idx_list: list[int] = []
    last_block_idx_list: list[int] = []
    for batch in range(batch_size):
        cur_s_k = int(effective_seqlens[batch].item())
        first_block_idx = 0
        last_block_idx = max(cur_s_k - 1, 0) // block_size_n
        num_blocks_list.append(last_block_idx - first_block_idx + 1)
        first_block_idx_list.append(first_block_idx)
        last_block_idx_list.append(last_block_idx)

    total_num_blocks = sum(n + fixed_overhead_num_blocks for n in num_blocks_list)
    payload = math.ceil(total_num_blocks / num_mp_parts) + fixed_overhead_num_blocks
    metadata = torch.zeros((num_mp_parts, 8), dtype=torch.int32, device=device)
    num_splits = torch.zeros((batch_size + 1,), dtype=torch.int32, device=device)

    now_idx = 0
    now_block = 0
    now_n_split_idx = 0
    cum_num_splits = 0
    for part in range(num_mp_parts):
        metadata[part, 0] = now_idx
        metadata[part, 1] = now_block + first_block_idx_list[now_idx] if now_idx < batch_size else 0
        metadata[part, 4] = now_n_split_idx

        remain_payload = payload
        while now_idx < batch_size:
            num_blocks = num_blocks_list[now_idx]
            now_remain_blocks = num_blocks - now_block
            if remain_payload >= now_remain_blocks + fixed_overhead_num_blocks:
                cum_num_splits += now_n_split_idx + 1
                num_splits[now_idx + 1] = cum_num_splits
                remain_payload -= now_remain_blocks + fixed_overhead_num_blocks
                now_idx += 1
                now_block = 0
                now_n_split_idx = 0
            else:
                if remain_payload - fixed_overhead_num_blocks > 0:
                    now_block += remain_payload - fixed_overhead_num_blocks
                    now_n_split_idx += 1
                break

        if now_block > 0:
            metadata[part, 2] = now_idx
            metadata[part, 3] = now_block + first_block_idx_list[now_idx]
        else:
            metadata[part, 2] = now_idx - 1
            metadata[part, 3] = last_block_idx_list[now_idx - 1] + 1 if now_idx > 0 else 0

    return metadata, num_splits


def _sparse_mla_prefill_stats(
    *,
    seq_len_q: int,
    seq_len_kv: int,
    d_qk: int,
    num_heads: int,
    topk: int,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
    extra_seq_len_kv: int = 0,
    extra_topk: int = 0,
) -> tuple[float, float]:
    if topk_length is None:
        total_topk = seq_len_q * topk
    else:
        total_topk = int(topk_length.sum().item())
    valid_mask = (indices >= 0) & (indices < seq_len_kv)
    if topk_length is not None:
        pos = torch.arange(topk, device=indices.device).view(1, 1, topk)
        valid_mask &= pos < topk_length.view(seq_len_q, 1, 1)
    num_valid_indices = int(valid_mask.sum().item())
    if extra_topk > 0:
        total_topk += seq_len_q * extra_topk
        num_valid_indices += seq_len_q * min(extra_topk, extra_seq_len_kv)
    flops = 2 * total_topk * num_heads * (d_qk + SPARSE_MLA_DV)
    mem_bytes = num_valid_indices * d_qk * 2 + seq_len_q * num_heads * (d_qk + SPARSE_MLA_DV) * 2
    return float(flops), float(mem_bytes)


def _sparse_mla_decode_stats(
    *,
    batch_size: int,
    seq_len_q: int,
    d_qk: int,
    num_heads: int,
    topk: int,
    indices: torch.Tensor,
    topk_length: torch.Tensor | None,
) -> tuple[float, float]:
    if topk_length is None:
        num_attended = batch_size * seq_len_q * topk
        masked = indices
    else:
        num_attended = int(topk_length.sum().item()) * seq_len_q
        pos = torch.arange(topk, device=indices.device).view(1, 1, 1, topk)
        masked = torch.where(
            pos < topk_length.view(batch_size, 1, 1, 1),
            indices,
            indices.new_full((), -1),
        )
    num_retrieved = int(masked.unique().numel())
    flops = 2 * num_heads * num_attended * (d_qk + SPARSE_MLA_DV)
    kv_token_size = 656 if d_qk == 576 else 576
    mem_bytes = (
        2 * batch_size * seq_len_q * num_heads * d_qk
        + num_retrieved * kv_token_size
        + 2 * batch_size * seq_len_q * num_heads * SPARSE_MLA_DV
    )
    return float(flops), float(mem_bytes)


def benchmark_sparse_mla_decode_v32_case(
    *,
    batch_size: int,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    topk: int,
    device: str,
) -> dict[str, Any]:
    from .kernels.sparse_mla_v32_decode_fwd_scheduled import (
        tilelang_flashmla_interface,
    )

    d_qk = 576
    torch.manual_seed(20260524 + batch_size + seq_len_q + seq_len_kv)
    q = torch.randn(
        (batch_size, seq_len_q, num_heads, d_qk),
        dtype=torch.bfloat16,
        device=device,
    )
    cache_seqlens = torch.tensor(
        [seq_len_kv - 4 * i for i in range(batch_size)],
        dtype=torch.int32,
        device=device,
    )
    cu_seqlens = torch.tensor(
        [0] + [seq_len_kv - 4 * i for i in range(batch_size)],
        dtype=torch.int32,
        device=device,
    ).cumsum(dim=0, dtype=torch.int32)
    total_seqlens = int(cache_seqlens.sum().item())
    kv = torch.randn((total_seqlens, 1, d_qk), dtype=torch.bfloat16, device=device)
    indices = _make_sparse_temp_decode_indices(batch_size, seq_len_q, topk, cache_seqlens, cu_seqlens, device)
    quant_scales = torch.tensor([0.6, 0.7, 0.8, 0.9], dtype=torch.float32, device=device).view(1, 1, 4)
    quant_scales = quant_scales.repeat_interleave(total_seqlens, dim=0)
    k_latent_fp8 = kv[..., :SPARSE_MLA_DV].to(torch.float8_e4m3fn).contiguous()
    k_pe = kv[..., SPARSE_MLA_DV:].to(torch.bfloat16).contiguous()
    k_cache = torch.cat(
        [
            k_latent_fp8.view(torch.uint8),
            quant_scales.view(torch.uint8),
            k_pe.view(torch.uint8),
        ],
        dim=-1,
    ).contiguous()
    sched_meta, num_splits = _make_sparse_decode_metadata(
        seqlens_k=cache_seqlens,
        num_q_tokens_per_head_k=seq_len_q * num_heads,
        num_heads_k=1,
        topk=topk,
        mp_count=56,
        tile_m=64,
    )
    topk_length = torch.full((batch_size,), topk, dtype=torch.int32, device=device)

    def fn():
        return tilelang_flashmla_interface(
            q,
            k_cache,
            indices,
            sched_meta,
            num_splits,
            sm_scale=None,
            topk_length=topk_length,
            attn_sink=None,
            d_v=SPARSE_MLA_DV,
            threads=512,
        )

    out, lse = fn()
    time_us = benchmark_timer(fn, warmup=5, rep=20)
    flops, mem_bytes = _sparse_mla_decode_stats(
        batch_size=batch_size,
        seq_len_q=seq_len_q,
        d_qk=d_qk,
        num_heads=num_heads,
        topk=topk,
        indices=indices,
        topk_length=topk_length,
    )
    bytes_rw = max(
        int(mem_bytes),
        count_bytes(q, k_cache, indices, sched_meta, num_splits, topk_length, out, lse),
    )
    return {
        "kernel": "mate/sparse_mla_decode_v32",
        "operation": "direct_tilelang",
        "params": {
            "B": batch_size,
            "seq_len_q": seq_len_q,
            "seq_len_kv": seq_len_kv,
            "heads": num_heads,
            "d_qk": d_qk,
            "d_v": SPARSE_MLA_DV,
            "topk": topk,
            "mp_parts": int(sched_meta.shape[0]),
            "dtype": "bfloat16",
        },
        "time_us": time_us,
        "bandwidth_gbs": bytes_rw / time_us / 1e3,
        "extras": {
            "bytes_rw": bytes_rw,
            "flops": flops,
            "tflops": flops / time_us / 1e6,
        },
    }


def benchmark_sparse_mla_prefill_v32_case(
    *,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    topk: int,
    device: str,
) -> dict[str, Any]:
    from .kernels.sparse_mla_v32_fwd_pipelined import (
        tilelang_sparse_mla_prefill_fwd_interface,
    )

    d_qk = 576
    torch.manual_seed(20260523 + seq_len_q + seq_len_kv + d_qk)
    q = torch.randn((seq_len_q, num_heads, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.randn((seq_len_kv, 1, d_qk), dtype=torch.bfloat16, device=device)
    indices = _make_sparse_temp_prefill_indices(seq_len_q, topk, seq_len_kv, device)

    def fn():
        return tilelang_sparse_mla_prefill_fwd_interface(
            q,
            kv,
            indices,
            sm_scale=None,
            topk_length=None,
            attn_sink=None,
            d_v=SPARSE_MLA_DV,
            return_max_logits=True,
        )

    out, max_logits, lse = fn()
    time_us = benchmark_timer(fn, warmup=5, rep=20)
    flops, mem_bytes = _sparse_mla_prefill_stats(
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        d_qk=d_qk,
        num_heads=num_heads,
        topk=topk,
        indices=indices,
        topk_length=None,
    )
    bytes_rw = max(int(mem_bytes), count_bytes(q, kv, indices, out, max_logits, lse))
    return {
        "kernel": "mate/sparse_mla_prefill_v32",
        "operation": "direct_tilelang",
        "params": {
            "seq_len_q": seq_len_q,
            "seq_len_kv": seq_len_kv,
            "heads": num_heads,
            "d_qk": d_qk,
            "d_v": SPARSE_MLA_DV,
            "topk": topk,
            "dtype": "bfloat16",
        },
        "time_us": time_us,
        "bandwidth_gbs": bytes_rw / time_us / 1e3,
        "extras": {
            "bytes_rw": bytes_rw,
            "flops": flops,
            "tflops": flops / time_us / 1e6,
        },
    }


def benchmark_sparse_mla_prefill_model1_case(
    *,
    seq_len_q: int,
    seq_len_kv: int,
    num_heads: int,
    topk: int,
    extra_seq_len_kv: int,
    extra_topk: int,
    device: str,
) -> dict[str, Any]:
    from .kernels.sparse_mla_model1_fwd_pipelined import (
        sparse_mla_fwd_interface_model1,
    )

    d_qk = 512
    torch.manual_seed(20260523 + seq_len_q + seq_len_kv + d_qk)
    q = torch.randn((seq_len_q, num_heads, d_qk), dtype=torch.bfloat16, device=device)
    kv = torch.randn((seq_len_kv, 1, d_qk), dtype=torch.bfloat16, device=device)
    indices = _make_sparse_indices(seq_len_q, topk, seq_len_kv, device)
    extra_kv = torch.randn((extra_seq_len_kv, 1, d_qk), dtype=torch.bfloat16, device=device)
    extra_indices = _make_sparse_indices(seq_len_q, extra_topk, extra_seq_len_kv, device)

    def fn():
        return sparse_mla_fwd_interface_model1(
            q=q,
            kv=kv,
            indices=indices,
            extra_kv=extra_kv,
            extra_indices=extra_indices,
            topk_length=None,
            extra_topk_length=None,
            sm_scale=d_qk**-0.5,
            attn_sink=None,
            d_v=SPARSE_MLA_DV,
            return_max_logits=True,
        )

    out, max_logits, lse = fn()
    time_us = benchmark_timer(fn, warmup=5, rep=20)
    flops, mem_bytes = _sparse_mla_prefill_stats(
        seq_len_q=seq_len_q,
        seq_len_kv=seq_len_kv,
        d_qk=d_qk,
        num_heads=num_heads,
        topk=topk,
        indices=indices,
        topk_length=None,
        extra_seq_len_kv=extra_seq_len_kv,
        extra_topk=extra_topk,
    )
    bytes_rw = max(
        int(mem_bytes),
        count_bytes(q, kv, indices, extra_kv, extra_indices, out, max_logits, lse),
    )
    return {
        "kernel": "mate/sparse_mla_prefill_model1",
        "operation": "direct_tilelang",
        "params": {
            "seq_len_q": seq_len_q,
            "seq_len_kv": seq_len_kv,
            "heads": num_heads,
            "d_qk": d_qk,
            "d_v": SPARSE_MLA_DV,
            "topk": topk,
            "extra_seq_len_kv": extra_seq_len_kv,
            "extra_topk": extra_topk,
            "dtype": "bfloat16",
        },
        "time_us": time_us,
        "bandwidth_gbs": bytes_rw / time_us / 1e3,
        "extras": {
            "bytes_rw": bytes_rw,
            "flops": flops,
            "tflops": flops / time_us / 1e6,
        },
    }


def _gdn_prefill_tflops(total_tokens: int, num_heads: int, head_size: int, time_us: float) -> float:
    # 2 GEMMs (kv outer product + q@state), MAC counted as 2 FLOPs.
    flops = 2 * 2 * total_tokens * num_heads * head_size * head_size
    return flops / time_us / 1e6


def benchmark_gdn_prefill_case(
    *,
    endpoints: tuple[int, ...],
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    device: str,
) -> dict[str, Any]:
    from .kernels.gdn_chunk_local_cumsum import chunk_local_cumsum
    from .kernels.gdn_kkt_solve import kkt_solve
    from .kernels.gdn_prefill import fused_chunk_gdn_prefill

    torch.manual_seed(20260521 + endpoints[-1] + num_q_heads + num_v_heads)
    num_seqs = len(endpoints)
    total_tokens = endpoints[-1]
    num_heads = max(num_q_heads, num_v_heads)
    cu_seqlens = torch.tensor([0, *endpoints], dtype=torch.int32, device=device)

    q = torch.randn((1, total_tokens, num_q_heads, head_size), dtype=dtype, device=device)
    k = torch.nn.functional.normalize(
        torch.randn(
            (1, total_tokens, num_q_heads, head_size),
            dtype=torch.float32,
            device=device,
        ),
        p=2,
        dim=-1,
    ).to(dtype)
    v = torch.randn((1, total_tokens, num_v_heads, head_size), dtype=dtype, device=device)
    g = torch.exp(-torch.rand(1, total_tokens, num_heads, dtype=torch.float32, device=device))
    beta = torch.sigmoid(torch.randn(1, total_tokens, num_heads, dtype=torch.float32, device=device))
    initial_state = torch.randn(
        (num_seqs, num_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )

    def fn():
        g_cumsum = chunk_local_cumsum(g, chunk_size=64, cu_seqlens=cu_seqlens)
        a = kkt_solve(k=k, b=beta, cu_seqlens=cu_seqlens)
        output, _, final_state = fused_chunk_gdn_prefill(
            q=q,
            k=k,
            v=v,
            a=a,
            g=g_cumsum,
            b=beta,
            scale=head_size**-0.5,
            initial_state=initial_state,
            output_final_state=True,
            output_h=False,
            output_o=True,
            cu_seqlens=cu_seqlens,
        )
        return output, final_state

    output, final_state = fn()
    time_us = benchmark_timer(fn, warmup=10, rep=50)
    bytes_rw = count_bytes(q, k, v, g, beta, initial_state, output, final_state)
    return {
        "kernel": "mate/gdn_prefill",
        "operation": "chunk_gated_delta_rule_pipeline",
        "params": {
            "endpoints": "+".join(str(x) for x in endpoints),
            "tokens": total_tokens,
            "seqs": num_seqs,
            "Hq": num_q_heads,
            "Hv": num_v_heads,
            "H": num_heads,
            "D": head_size,
            "dtype": str(dtype).split(".")[-1],
            "chunk_size": 64,
        },
        "time_us": time_us,
        "bandwidth_gbs": bytes_rw / time_us / 1e3,
        "extras": {
            "bytes_rw": bytes_rw,
            "tflops": _gdn_prefill_tflops(total_tokens, num_heads, head_size, time_us),
        },
    }


def benchmark_gdn_mtp_case(
    *,
    batch_size: int,
    seq_len: int,
    num_q_heads: int,
    num_v_heads: int,
    head_size: int,
    dtype: torch.dtype,
    cache_intermediate_states: bool,
    disable_state_update: bool,
    device: str,
) -> dict[str, Any]:
    from .kernels import gdn_mtp

    torch.manual_seed(20260520 + batch_size + seq_len + num_q_heads + num_v_heads)
    q = torch.randn((batch_size, seq_len, num_q_heads, head_size), dtype=dtype, device=device)
    k = torch.randn((batch_size, seq_len, num_q_heads, head_size), dtype=dtype, device=device)
    v = torch.randn((batch_size, seq_len, num_v_heads, head_size), dtype=dtype, device=device)
    state = torch.randn(
        (batch_size, num_v_heads, head_size, head_size),
        dtype=torch.float32,
        device=device,
    )
    state_indices = torch.arange(batch_size, dtype=torch.int32, device=device)
    A_log = torch.randn((num_v_heads,), dtype=torch.float32, device=device) * 0.1
    dt_bias = torch.randn((num_v_heads,), dtype=dtype, device=device) * 0.1
    a = torch.randn((batch_size, seq_len, num_v_heads), dtype=dtype, device=device) * 0.1
    b = torch.randn((batch_size, seq_len, num_v_heads), dtype=dtype, device=device) * 0.1
    output = torch.empty((batch_size, seq_len, num_v_heads, head_size), dtype=dtype, device=device)
    intermediate = (
        torch.empty(
            (batch_size, seq_len, num_v_heads, head_size, head_size),
            dtype=torch.float32,
            device=device,
        )
        if cache_intermediate_states
        else None
    )

    tile_v, _, ilp_rows = gdn_mtp._get_mtp_config(
        batch_size=batch_size,
        seq_len=seq_len,
        num_v_heads=num_v_heads,
        v_dim=head_size,
        cache_intermediate_states=cache_intermediate_states,
    )
    kernel = gdn_mtp._get_mtp_fp32_vk_smem_kernel(
        seq_len=seq_len,
        qk_head=num_q_heads,
        head=num_v_heads,
        dim_k=head_size,
        dim_v=head_size,
        input_dtype=str(dtype).split(".")[-1],
        output_dtype=str(dtype).split(".")[-1],
        dt_bias_dtype=str(dt_bias.dtype).split(".")[-1],
        use_qk_l2norm=True,
        cache_intermediate_states=cache_intermediate_states,
        disable_state_update=disable_state_update,
        use_identity_state_indices=False,
        tile_v=tile_v,
        ilp_rows=ilp_rows,
    )
    intermediate_arg = intermediate if intermediate is not None else torch.empty((1, 1, 1, 1, 1), dtype=torch.float32, device=device)
    scale = head_size**-0.5

    def fn() -> None:
        kernel(
            q,
            k,
            v,
            A_log,
            a,
            dt_bias,
            b,
            float(scale),
            state,
            state_indices,
            intermediate_arg,
            output,
        )

    fn()
    time_us = benchmark_timer(fn)
    flops = _gdn_mtp_flops(
        batch_size=batch_size,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
    )
    bytes_rw = _gdn_mtp_bytes(
        batch_size=batch_size,
        seq_len=seq_len,
        num_q_heads=num_q_heads,
        num_v_heads=num_v_heads,
        head_size=head_size,
        input_dtype=dtype,
        output_dtype=dtype,
        disable_state_update=disable_state_update,
        cache_intermediate_states=cache_intermediate_states,
    )
    return {
        "kernel": "mate/gdn_mtp",
        "operation": "gated_deltanet_mtp_fp32_vk_smem",
        "params": {
            "B": batch_size,
            "T": seq_len,
            "Hq": num_q_heads,
            "Hv": num_v_heads,
            "D": head_size,
            "dtype": str(dtype).split(".")[-1],
            "tile_v": tile_v,
            "ilp_rows": ilp_rows,
            "cache_intermediate": cache_intermediate_states,
            "update_state": not disable_state_update,
        },
        "time_us": time_us,
        "bandwidth_gbs": bytes_rw / time_us / 1e3,
        "extras": {
            "bytes_rw": bytes_rw,
            "flops": flops,
            "tflops": flops / time_us / 1e6,
        },
    }


def build_cases() -> list[CaseSpec]:
    return [
        *build_gdn_decode_cases(),
        CaseSpec("sparse_mla_prefill_v32_temp_aligned_bf16", "sparse_mla_prefill_v32", (896, 4096, 128, 2048)),
        CaseSpec("sparse_mla_prefill_model1_small_extra_bf16", "sparse_mla_prefill_model1", (128, 512, 64, 64, 512, 64)),
        CaseSpec("sparse_mla_prefill_model1_extra_bf16", "sparse_mla_prefill_model1", (896, 8192, 128, 2048, 8192, 2048)),
        CaseSpec("sparse_mla_decode_v32_temp_aligned_bf16", "sparse_mla_decode_v32", (1, 896, 8192, 128, 2048)),
        *build_gdn_mtp_cases(),
        *build_gdn_prefill_cases(),
    ]


SLOW_CASE_NAMES = {
    # This mirrors MATE's --include-large direct TileLang perf case. It is
    # useful for explicit profiling, but its topk + extra_topk shape can spend
    # a long time in TileLang compilation and should not block default sweeps.
    "sparse_mla_prefill_model1_extra_bf16",
}


REGRESSION_SUPPORTED_RUNNERS = frozenset({"gdn_decode", "gdn_mtp", "gdn_prefill"})


def default_case_names() -> list[str]:
    return [case.name for case in build_cases() if case.name not in SLOW_CASE_NAMES]


def regression_supported_case_names() -> list[str]:
    return [case.name for case in build_cases() if case.runner in REGRESSION_SUPPORTED_RUNNERS]


def build_case_map() -> dict[str, CaseSpec]:
    return {case.name: case for case in build_cases()}


def run_case(case: CaseSpec, device: str) -> dict[str, Any]:
    if case.runner == "gdn_decode":
        batch_size, num_q_heads, num_v_heads, head_size = case.args
        return benchmark_gdn_decode_case(
            batch_size=batch_size,
            num_q_heads=num_q_heads,
            num_v_heads=num_v_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            device=device,
        )
    if case.runner == "sparse_mla_prefill_v32":
        seq_len_q, seq_len_kv, num_heads, topk = case.args
        return benchmark_sparse_mla_prefill_v32_case(
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_heads=num_heads,
            topk=topk,
            device=device,
        )
    if case.runner == "sparse_mla_prefill_model1":
        seq_len_q, seq_len_kv, num_heads, topk, extra_seq_len_kv, extra_topk = case.args
        return benchmark_sparse_mla_prefill_model1_case(
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_heads=num_heads,
            topk=topk,
            extra_seq_len_kv=extra_seq_len_kv,
            extra_topk=extra_topk,
            device=device,
        )
    if case.runner == "sparse_mla_decode_v32":
        batch_size, seq_len_q, seq_len_kv, num_heads, topk = case.args
        return benchmark_sparse_mla_decode_v32_case(
            batch_size=batch_size,
            seq_len_q=seq_len_q,
            seq_len_kv=seq_len_kv,
            num_heads=num_heads,
            topk=topk,
            device=device,
        )
    if case.runner == "gdn_mtp":
        batch_size, seq_len, num_q_heads, num_v_heads, head_size = case.args
        return benchmark_gdn_mtp_case(
            batch_size=batch_size,
            seq_len=seq_len,
            num_q_heads=num_q_heads,
            num_v_heads=num_v_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            cache_intermediate_states=False,
            disable_state_update=True,
            device=device,
        )
    if case.runner == "gdn_prefill":
        endpoints, num_q_heads, num_v_heads, head_size = case.args
        return benchmark_gdn_prefill_case(
            endpoints=endpoints,
            num_q_heads=num_q_heads,
            num_v_heads=num_v_heads,
            head_size=head_size,
            dtype=torch.bfloat16,
            device=device,
        )
    raise ValueError(f"unknown runner: {case.runner}")


def aggregate_sample_records(sample_records: list[dict[str, Any]]) -> dict[str, Any]:
    if not sample_records:
        raise ValueError("sample_records must not be empty")
    aggregate = copy.deepcopy(sample_records[0])
    times = [record["time_us"] for record in sample_records]
    median_time_us = statistics.median(times)
    aggregate["time_us"] = median_time_us
    bytes_rw = aggregate.get("extras", {}).get("bytes_rw")
    if bytes_rw is not None:
        aggregate["bandwidth_gbs"] = bytes_rw / median_time_us / 1e3
    flops = aggregate.get("extras", {}).get("flops")
    if flops is not None:
        aggregate["extras"]["tflops"] = flops / median_time_us / 1e6
    return aggregate


def run_case_samples(case: CaseSpec, device: str, samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_records = []
    for sample_index in range(1, samples + 1):
        record = run_case(case, device)
        sample_records.append(record)
        if samples > 1:
            print(f"  {style(f'sample {sample_index}/{samples}', TermStyle.dim)} time={record['time_us']:.2f} us")
    return aggregate_sample_records(sample_records), sample_records


def print_sample_stats(sample_records: list[dict[str, Any]], median_time_us: float) -> None:
    if len(sample_records) <= 1:
        return
    times = [record["time_us"] for record in sample_records]
    print(
        f"  {style('samples', TermStyle.cyan):<14} "
        f"n={len(times)} median={median_time_us:.2f} us "
        f"mean={statistics.mean(times):.2f} us min={min(times):.2f} us max={max(times):.2f} us"
    )


def parse_args(
    default_case_names: list[str] | None,
    description: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--baseline",
        default=str(benchmark_root().parent / BASELINE_FILENAME),
        help="Path to the JSONL baseline file.",
    )
    parser.add_argument("--output", help="Optional JSONL output path for current benchmark records.")
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Compare current records against the baseline file.",
    )
    parser.add_argument(
        "--samples",
        type=int,
        default=1,
        help="Number of independent samples per case. The median time is used for output and regression checks.",
    )
    parser.add_argument(
        "--cases",
        nargs="*",
        help="Optional list of case names to run. Defaults to the entrypoint's built-in selection.",
    )
    parser.add_argument(
        "--allow-non-release-build",
        action="store_true",
        help="Allow benchmarks to run even when tilelang-musa is not built with CMAKE_BUILD_TYPE=Release.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Allowed slowdown ratio margin. 0.05 means current/baseline <= 1.05.",
    )
    return parser.parse_args(argv)


def run_cases(
    *,
    title: str,
    default_case_names: list[str] | None = None,
    description: str,
    argv: list[str] | None = None,
    print_final_summary: bool = True,
) -> BenchmarkRunResult:
    start_time = time.perf_counter()
    args = parse_args(default_case_names, description, argv)
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")

    ensure_release_build(strict=not args.allow_non_release_build)
    device = get_test_device()
    build_type, build_type_source = detect_tilelang_musa_build_type()

    case_map = build_case_map()
    if args.cases:
        selected_names = args.cases
    elif default_case_names is not None:
        selected_names = default_case_names
    else:
        selected_names = list(case_map)

    missing = [name for name in selected_names if name not in case_map]
    if missing:
        raise ValueError(f"unknown benchmark case(s): {', '.join(missing)}")

    if args.check_regression:
        unsupported_names = [name for name in selected_names if case_map[name].runner not in REGRESSION_SUPPORTED_RUNNERS]
        if unsupported_names:
            if args.cases:
                raise ValueError(
                    "--check-regression currently supports only MATE GDN "
                    "case families: gdn_decode, gdn_mtp, gdn_prefill. "
                    "Unsupported requested case(s): "
                    f"{', '.join(unsupported_names)}"
                )
            selected_names = [name for name in selected_names if case_map[name].runner in REGRESSION_SUPPORTED_RUNNERS]
            print(
                f"{style('[WARN]', TermStyle.bold, TermStyle.yellow)} "
                "--check-regression is currently limited to MATE GDN cases; "
                f"skipping unsupported case(s): {', '.join(unsupported_names)}"
            )

    cases = [case_map[name] for name in selected_names]
    if not cases:
        raise ValueError("no benchmark cases selected")

    build_type_detail = build_type or "unknown"
    if build_type_source is not None:
        build_type_detail = f"{build_type_detail} ({build_type_source})"
    print_banner(
        title,
        f"source=mate device={device} cases={len(cases)} samples={args.samples} build_type={build_type_detail}",
    )

    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"{style(f'[{index:02d}/{len(cases):02d}]', TermStyle.bold, TermStyle.gray)} {style(case.name, TermStyle.bold)}")
        record, sample_records = run_case_samples(case, device, args.samples)
        records.append(record)
        print_perf(record)
        print_sample_stats(sample_records, record["time_us"])
        print_json_record(record)
        print()

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"{style('[WRITE]', TermStyle.bold, TermStyle.blue)} saved {len(records)} records to {output_path}")

    regression_stats: dict[str, Any] | None = None
    if args.check_regression:
        print_banner("Regression Check", f"baseline={args.baseline}")
        regression_stats = check_regression(records, load_baselines(Path(args.baseline)), args.threshold)

    exit_code = 0
    if regression_stats is not None:
        exit_code = 1 if regression_stats["failures"] or regression_stats["missing"] else 0

    if print_final_summary:
        print_summary(len(records), regression_stats, time.perf_counter() - start_time)
        if regression_stats is None:
            print(f"{style('[DONE]', TermStyle.bold, TermStyle.green)} completed {len(records)} benchmark case(s)")

    return BenchmarkRunResult(records=records, regression_stats=regression_stats, exit_code=exit_code)


def run_cases_main(
    *,
    title: str,
    default_case_names: list[str] | None = None,
    description: str,
) -> int:
    return run_cases(
        title=title,
        default_case_names=default_case_names,
        description=description,
    ).exit_code
