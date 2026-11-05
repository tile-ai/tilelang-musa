# SPDX-License-Identifier: Apache-2.0
# SPDX-FileCopyrightText: Copyright contributors to the vLLM project

import torch

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

_FP8_DTYPES = {
    getattr(torch, "float8_e4m3fn", None),
    getattr(torch, "float8_e4m3fnuz", None),
    getattr(torch, "float8_e5m2", None),
    getattr(torch, "float8_e5m2fnuz", None),
}
_FP8_DTYPES.discard(None)
_INT8_DTYPES = {torch.int8}


def _cdiv(x, y):
    return (x + y - 1) // y


def _next_power_of_2(x):
    if x <= 1:
        return 1
    return 1 << (x - 1).bit_length()


def _scalar_from_descale(descale):
    if descale is None:
        return None
    if torch.is_tensor(descale):
        return float(descale.item())
    return float(descale)


def _apply_softcap(scores, softcap):
    if softcap is None or softcap <= 0:
        return scores
    return softcap * torch.tanh(scores / softcap)


def _pad_last_dim(x, padded_size):
    if x.shape[-1] == padded_size:
        return x
    out = torch.zeros(*x.shape[:-1], padded_size, device=x.device, dtype=x.dtype)
    out[..., : x.shape[-1]] = x
    return out


def _materialize_paged_kv(cache, seq_blocks, seq_len):
    if seq_len == 0:
        return cache.new_empty((0,) + cache.shape[2:])
    blocks = cache.index_select(0, seq_blocks)
    return blocks.reshape(-1, cache.shape[2], cache.shape[3])[:seq_len]


def _maybe_dequantize_kv(k_seq, v_seq, q_dtype, k_descale, v_descale):
    if k_seq.dtype in _FP8_DTYPES and q_dtype not in _FP8_DTYPES or k_seq.dtype in _INT8_DTYPES:
        k_scale = _scalar_from_descale(k_descale)
        k_seq = k_seq.float() if k_scale is None else k_seq.float() * k_scale
    else:
        k_seq = k_seq.float()

    if v_seq.dtype in _FP8_DTYPES and q_dtype not in _FP8_DTYPES or v_seq.dtype in _INT8_DTYPES:
        v_scale = _scalar_from_descale(v_descale)
        v_seq = v_seq.float() if v_scale is None else v_seq.float() * v_scale
    else:
        v_seq = v_seq.float()

    return k_seq, v_seq


def _find_seq_idx(query_start_len_ptr, target_idx, num_seqs, block_q, use_q_block_mode):
    left = 0
    right = num_seqs
    while left < right:
        mid = (left + right) // 2
        val = int(query_start_len_ptr[mid].item())
        mid_val = val // block_q + mid if use_q_block_mode else val
        if mid_val <= target_idx:
            left = mid + 1
        else:
            right = mid
    return left - 1


def _build_seq_states(
    q,
    k,
    v,
    cu_seqlens_q,
    seqused_k,
    block_table,
    qq_bias,
    head_size_padded,
    k_descale,
    v_descale,
):
    num_seqs = int(seqused_k.numel())
    states = []
    for seq_idx in range(num_seqs):
        q_start = int(cu_seqlens_q[seq_idx].item())
        q_end = int(cu_seqlens_q[seq_idx + 1].item())
        q_len = q_end - q_start
        seq_len = int(seqused_k[seq_idx].item())
        context_len = seq_len - q_len
        block_size = int(v.shape[1])
        num_blocks = _cdiv(seq_len, block_size)
        seq_blocks = block_table[seq_idx, :num_blocks].to(dtype=torch.long)

        k_seq = _materialize_paged_kv(k, seq_blocks, seq_len)
        v_seq = _materialize_paged_kv(v, seq_blocks, seq_len)
        k_seq, v_seq = _maybe_dequantize_kv(k_seq, v_seq, q.dtype, k_descale, v_descale)
        q_seq = q[q_start:q_end].float()

        states.append(
            {
                "q_start": q_start,
                "q_end": q_end,
                "q_len": q_len,
                "seq_len": seq_len,
                "context_len": context_len,
                "q_seq": _pad_last_dim(q_seq, head_size_padded),
                "k_seq": _pad_last_dim(k_seq, head_size_padded),
                "v_seq": _pad_last_dim(v_seq, head_size_padded),
                "qq_bias_seq": None if qq_bias is None else qq_bias[:q_len, :q_len].float(),
            }
        )
    return states


def _compute_block_scores(
    q_block,
    k_block,
    softmax_scale,
    softcap,
    context_len,
    query_offsets,
    seq_positions,
    sliding_window,
    alibi_slopes,
    qq_bias_rows,
):
    scores = torch.einsum("qhd,kd->hqk", q_block, k_block) * softmax_scale
    scores = _apply_softcap(scores, softcap)

    causal_mask = seq_positions.unsqueeze(0) < (context_len + query_offsets + 1).unsqueeze(1)
    if sliding_window > 0:
        causal_mask &= (context_len + query_offsets).unsqueeze(1) - seq_positions.unsqueeze(0) < sliding_window
    scores = scores.masked_fill(~causal_mask.unsqueeze(0), float("-inf"))

    if alibi_slopes is not None:
        scores = scores + alibi_slopes[:, None, None] * (seq_positions - context_len).to(torch.float32)

    if qq_bias_rows is not None:
        key_rel = seq_positions - context_len
        valid = (key_rel >= 0) & (key_rel < qq_bias_rows.shape[1])
        qq_bias_vals = torch.zeros(
            (query_offsets.shape[0], seq_positions.shape[0]),
            device=q_block.device,
            dtype=torch.float32,
        )
        if valid.any():
            qq_bias_vals[:, valid] = qq_bias_rows.index_select(1, key_rel[valid].long())
        scores = scores + qq_bias_vals.unsqueeze(0)

    return scores


def _run_attention_blocks(
    q_block,
    k_head,
    v_head,
    softmax_scale,
    softcap,
    context_len,
    query_offsets,
    sliding_window,
    alibi_slopes,
    qq_bias_rows,
    sinks,
    block_size,
    block_start,
    block_end,
    include_sinks,
):
    num_heads = q_block.shape[1]
    head_size_padded = q_block.shape[2]
    device = q_block.device
    dtype = torch.float32

    if include_sinks and sinks is not None:
        m = sinks.to(dtype)[:, None].expand(num_heads, query_offsets.shape[0]).clone()
    else:
        m = torch.full((num_heads, query_offsets.shape[0]), float("-inf"), device=device, dtype=dtype)
    l = torch.ones_like(m)
    acc = torch.zeros((num_heads, query_offsets.shape[0], head_size_padded), device=device, dtype=dtype)

    for block_idx in range(block_start, block_end):
        token_start = block_idx * block_size
        token_stop = min(token_start + block_size, k_head.shape[0])
        if token_start >= token_stop:
            continue

        seq_positions = torch.arange(token_start, token_stop, device=device, dtype=torch.long)
        k_block = k_head[token_start:token_stop]
        v_block = v_head[token_start:token_stop]

        scores = _compute_block_scores(
            q_block,
            k_block,
            softmax_scale,
            softcap,
            context_len,
            query_offsets,
            seq_positions,
            sliding_window,
            alibi_slopes,
            qq_bias_rows,
        )

        m_j = torch.maximum(m, scores.max(dim=-1).values)
        m_j = torch.where(torch.isfinite(m_j), m_j, torch.zeros_like(m_j))
        p = torch.exp(scores - m_j.unsqueeze(-1))
        l_j = p.sum(dim=-1)
        alpha = torch.exp(m - m_j)
        acc = acc * alpha.unsqueeze(-1) + torch.einsum("hqk,kd->hqd", p, v_block)
        l = l * alpha + l_j
        m = m_j

    return acc, m, l


def torch_kernel_unified_attention_2d(
    q,
    k,
    v,
    out_f32,
    cu_seqlens_q,
    seqused_k,
    softmax_scale,
    window_size,
    block_table,
    softcap,
    k_descale,
    v_descale,
    num_query_heads,
    num_kv_heads,
    block_m,
    block_q,
    head_size,
    head_size_padded,
    total_num_q_blocks,
    alibi_slopes=None,
    qq_bias=None,
    sinks=None,
):
    sliding_window = -1 if window_size[0] < 0 else 1 + int(window_size[0])
    block_size = int(v.shape[1])
    num_seqs = int(seqused_k.numel())
    num_queries_per_kv = num_query_heads // num_kv_heads
    states = _build_seq_states(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        qq_bias,
        head_size_padded,
        k_descale,
        v_descale,
    )

    for q_block_global_idx in range(total_num_q_blocks):
        seq_idx = _find_seq_idx(cu_seqlens_q, q_block_global_idx, num_seqs, block_q, True)
        if seq_idx < 0:
            continue
        state = states[seq_idx]

        q_block_start_idx = int(cu_seqlens_q[seq_idx].item()) // block_q + seq_idx
        q_block_local_idx = q_block_global_idx - q_block_start_idx
        if q_block_local_idx * block_q >= state["q_len"]:
            continue

        q_tok_start = q_block_local_idx * block_q
        q_tok_stop = min(q_tok_start + block_q, state["q_len"])
        query_offsets = torch.arange(q_tok_start, q_tok_stop, device=q.device, dtype=torch.long)

        max_seq_prefix_len = state["context_len"] + q_block_local_idx * block_q + (block_m - 1) // num_queries_per_kv + 1
        max_seq_prefix_len = min(max_seq_prefix_len, state["seq_len"])
        num_blocks = _cdiv(max_seq_prefix_len, block_size)

        for kv_head_idx in range(num_kv_heads):
            head_start = kv_head_idx * num_queries_per_kv
            head_stop = min(head_start + num_queries_per_kv, num_query_heads)
            if head_start >= head_stop:
                continue

            q_block = state["q_seq"][q_tok_start:q_tok_stop, head_start:head_stop, :]
            k_head = state["k_seq"][:, kv_head_idx, :]
            v_head = state["v_seq"][:, kv_head_idx, :]
            alibi_head = None if alibi_slopes is None else alibi_slopes[head_start:head_stop].float()
            sink_head = None if sinks is None else sinks[head_start:head_stop].float()
            qq_bias_rows = None if state["qq_bias_seq"] is None else state["qq_bias_seq"][q_tok_start:q_tok_stop, :]

            acc, _, l = _run_attention_blocks(
                q_block,
                k_head,
                v_head,
                softmax_scale,
                softcap,
                state["context_len"],
                query_offsets,
                sliding_window,
                alibi_head,
                qq_bias_rows,
                sink_head,
                block_size,
                0,
                num_blocks,
                True,
            )

            l_safe = torch.where(l > 0, l, torch.ones_like(l))
            output = (acc / l_safe.unsqueeze(-1)).masked_fill((l <= 0).unsqueeze(-1), 0.0).permute(1, 0, 2)
            out_f32[
                state["q_start"] + q_tok_start : state["q_start"] + q_tok_stop,
                head_start:head_stop,
                :,
            ] = output[..., :head_size]


def torch_kernel_unified_attention_3d(
    q,
    k,
    v,
    cu_seqlens_q,
    seqused_k,
    softmax_scale,
    window_size,
    block_table,
    softcap,
    k_descale,
    v_descale,
    num_query_heads,
    num_kv_heads,
    block_m,
    block_q,
    head_size,
    head_size_padded,
    total_num_q_blocks,
    num_segments=16,
    alibi_slopes=None,
    qq_bias=None,
    sinks=None,
):
    sliding_window = -1 if window_size[0] < 0 else 1 + int(window_size[0])
    block_size = int(v.shape[1])
    num_seqs = int(seqused_k.numel())
    num_queries_per_kv = num_query_heads // num_kv_heads
    states = _build_seq_states(
        q,
        k,
        v,
        cu_seqlens_q,
        seqused_k,
        block_table,
        qq_bias,
        head_size_padded,
        k_descale,
        v_descale,
    )

    segm_output = torch.zeros(
        q.shape[0],
        num_query_heads,
        num_segments,
        head_size_padded,
        device=q.device,
        dtype=torch.float32,
    )
    segm_max = torch.full(
        (q.shape[0], num_query_heads, num_segments),
        float("-inf"),
        device=q.device,
        dtype=torch.float32,
    )
    segm_expsum = torch.zeros(
        q.shape[0],
        num_query_heads,
        num_segments,
        device=q.device,
        dtype=torch.float32,
    )

    for q_block_global_idx in range(total_num_q_blocks):
        seq_idx = _find_seq_idx(cu_seqlens_q, q_block_global_idx, num_seqs, block_q, True)
        if seq_idx < 0:
            continue
        state = states[seq_idx]

        q_block_start_idx = int(cu_seqlens_q[seq_idx].item()) // block_q + seq_idx
        q_block_local_idx = q_block_global_idx - q_block_start_idx
        if q_block_local_idx * block_q >= state["q_len"]:
            continue

        blocks_per_segment = _cdiv(state["seq_len"], num_segments * block_size)
        num_blocks = _cdiv(state["seq_len"], block_size)
        q_tok_start = q_block_local_idx * block_q
        q_tok_stop = min(q_tok_start + block_q, state["q_len"])
        query_offsets = torch.arange(q_tok_start, q_tok_stop, device=q.device, dtype=torch.long)

        for kv_head_idx in range(num_kv_heads):
            head_start = kv_head_idx * num_queries_per_kv
            head_stop = min(head_start + num_queries_per_kv, num_query_heads)
            if head_start >= head_stop:
                continue

            q_block = state["q_seq"][q_tok_start:q_tok_stop, head_start:head_stop, :]
            k_head = state["k_seq"][:, kv_head_idx, :]
            v_head = state["v_seq"][:, kv_head_idx, :]
            alibi_head = None if alibi_slopes is None else alibi_slopes[head_start:head_stop].float()
            sink_head = None if sinks is None else sinks[head_start:head_stop].float()
            qq_bias_rows = None if state["qq_bias_seq"] is None else state["qq_bias_seq"][q_tok_start:q_tok_stop, :]

            for segm_idx in range(num_segments):
                block_start = segm_idx * blocks_per_segment
                block_end = min((segm_idx + 1) * blocks_per_segment, num_blocks)
                if block_start >= block_end:
                    continue

                acc, m, l = _run_attention_blocks(
                    q_block,
                    k_head,
                    v_head,
                    softmax_scale,
                    softcap,
                    state["context_len"],
                    query_offsets,
                    sliding_window,
                    alibi_head,
                    qq_bias_rows,
                    sink_head,
                    block_size,
                    block_start,
                    block_end,
                    segm_idx == 0,
                )

                segm_output[
                    state["q_start"] + q_tok_start : state["q_start"] + q_tok_stop,
                    head_start:head_stop,
                    segm_idx,
                    :,
                ] = acc.permute(1, 0, 2)
                segm_max[
                    state["q_start"] + q_tok_start : state["q_start"] + q_tok_stop,
                    head_start:head_stop,
                    segm_idx,
                ] = m.permute(1, 0)
                segm_expsum[
                    state["q_start"] + q_tok_start : state["q_start"] + q_tok_stop,
                    head_start:head_stop,
                    segm_idx,
                ] = l.permute(1, 0)

    return segm_output, segm_max, segm_expsum


def torch_reduce_segments(
    out_f32,
    segm_output,
    segm_max,
    segm_expsum,
    cu_seqlens_q,
    seqused_k,
    num_query_heads,
    head_size,
    head_size_padded,
    block_q,
    num_segments,
    block_size,
):
    num_seqs = int(seqused_k.numel())
    total_tokens = int(out_f32.shape[0])

    for query_token_idx in range(total_tokens):
        seq_idx = _find_seq_idx(cu_seqlens_q, query_token_idx, num_seqs, block_q, False)
        if seq_idx < 0:
            continue

        seq_len = int(seqused_k[seq_idx].item())
        blocks_per_segment = _cdiv(seq_len, num_segments * block_size)
        act_num_segments = _cdiv(seq_len, blocks_per_segment * block_size)

        for query_head_idx in range(num_query_heads):
            cur_segm_max = segm_max[query_token_idx, query_head_idx, :act_num_segments].contiguous()
            cur_segm_expsum = segm_expsum[query_token_idx, query_head_idx, :act_num_segments].contiguous()
            cur_segm_output = segm_output[query_token_idx, query_head_idx, :act_num_segments, :].contiguous()

            overall_max = torch.amax(cur_segm_max, dim=-1)
            scaled_expsum = cur_segm_expsum * torch.exp(cur_segm_max - overall_max)
            overall_expsum = scaled_expsum.sum(dim=-1)
            scaled_output = cur_segm_output * torch.exp(cur_segm_max - overall_max)[:, None]
            acc_sum = scaled_output.sum(dim=0)

            overall_expsum_safe = torch.where(overall_expsum > 0, overall_expsum, torch.ones_like(overall_expsum))
            acc = (acc_sum / overall_expsum_safe).masked_fill(overall_expsum <= 0, 0.0)
            out_f32[query_token_idx, query_head_idx, :] = acc[:head_size]


def unified_attention(
    q,
    k,
    v,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    softcap,
    q_descale,
    k_descale,
    v_descale,
    alibi_slopes=None,
    qq_bias=None,
    sinks=None,
):
    assert causal, "Only causal attention is supported"
    assert q_descale is None, "Q scales not supported"

    block_size = int(v.shape[1])
    assert q.element_size() >= 2 or block_size >= 32, "Block size must be at least 32 for fp8"
    if sinks is not None:
        assert sinks.shape[0] == q.shape[1], "Sinks must be num_query_heads size"

    num_seqs = int(seqused_k.numel())
    num_query_heads = int(q.shape[1])
    num_kv_heads = int(k.shape[2])
    num_queries_per_kv = num_query_heads // num_kv_heads
    assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

    head_size = int(q.shape[2])
    head_size_padded = _next_power_of_2(head_size)
    block_m = 32
    block_q = block_m // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // block_q + num_seqs
    out_f32 = torch.empty_like(out, dtype=torch.float32)

    if max_seqlen_q > 1 or total_num_q_blocks * num_kv_heads > 128:
        torch_kernel_unified_attention_2d(
            q,
            k,
            v,
            out_f32,
            cu_seqlens_q,
            seqused_k,
            softmax_scale,
            window_size,
            block_table,
            softcap,
            k_descale,
            v_descale,
            num_query_heads,
            num_kv_heads,
            block_m,
            block_q,
            head_size,
            head_size_padded,
            total_num_q_blocks,
            alibi_slopes=alibi_slopes,
            qq_bias=qq_bias,
            sinks=sinks,
        )
    else:
        num_segments = 16
        segm_output, segm_max, segm_expsum = torch_kernel_unified_attention_3d(
            q,
            k,
            v,
            cu_seqlens_q,
            seqused_k,
            softmax_scale,
            window_size,
            block_table,
            softcap,
            k_descale,
            v_descale,
            num_query_heads,
            num_kv_heads,
            block_m,
            block_q,
            head_size,
            head_size_padded,
            total_num_q_blocks,
            num_segments=num_segments,
            alibi_slopes=alibi_slopes,
            qq_bias=qq_bias,
            sinks=sinks,
        )
        torch_reduce_segments(
            out_f32,
            segm_output,
            segm_max,
            segm_expsum,
            cu_seqlens_q,
            seqused_k,
            num_query_heads,
            head_size,
            head_size_padded,
            block_q,
            num_segments,
            block_size,
        )

    out.copy_(out_f32.to(dtype=out.dtype))
