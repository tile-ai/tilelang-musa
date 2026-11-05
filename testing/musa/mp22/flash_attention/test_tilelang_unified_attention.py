import itertools

import pytest
import tilelang.testing

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

import torch

from .torch_unified_attention import unified_attention as torch_unified_attention
from .tilelang_unified_attention import unified_attention as tilelang_unified_attention


_CASE_FILTER = None
_CASE_HIT = False


class _SelectedCaseDone(Exception):
    pass


def get_device():
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def sync(device):
    if device == "musa":
        torch.musa.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def run_case(
    case_name,
    q,
    k_cache,
    v_cache,
    cu_seqlens_q,
    seqused_k,
    block_table,
    max_seqlen_q,
    window_size=(-1, -1),
    softcap=0.0,
    alibi_slopes=None,
    qq_bias=None,
    sinks=None,
    k_descale=None,
    v_descale=None,
):
    global _CASE_HIT
    if _CASE_FILTER is not None and case_name != _CASE_FILTER:
        return

    device = q.device.type
    head_dim = q.shape[2]
    max_seqlen_k = int(seqused_k.max().item())
    softmax_scale = 1.0 / (head_dim**0.5)

    out_torch = torch.empty_like(q)
    out_tilelang = torch.empty_like(q)

    torch_unified_attention(
        q,
        k_cache,
        v_cache,
        out_torch,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        True,
        window_size,
        block_table,
        softcap,
        None,
        k_descale,
        v_descale,
        alibi_slopes,
        qq_bias,
        sinks,
    )
    sync(device)

    tilelang_unified_attention(
        q,
        k_cache,
        v_cache,
        out_tilelang,
        cu_seqlens_q,
        max_seqlen_q,
        seqused_k,
        max_seqlen_k,
        softmax_scale,
        True,
        window_size,
        block_table,
        softcap,
        None,
        k_descale,
        v_descale,
        alibi_slopes,
        qq_bias,
        sinks,
    )
    sync(device)

    torch.testing.assert_close(out_tilelang.float(), out_torch.float(), rtol=1e-2, atol=1e-2)
    max_diff = (out_tilelang.float() - out_torch.float()).abs().max().item()
    mean_diff = (out_tilelang.float() - out_torch.float()).abs().mean().item()
    print(f"{case_name}: PASS max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
    _CASE_HIT = True
    if _CASE_FILTER is not None:
        raise _SelectedCaseDone


def feature_subsets():
    names = ("softcap", "window", "alibi", "bias", "sink")
    yield ()
    for r in range(1, len(names) + 1):
        for combo in itertools.combinations(names, r):
            yield combo


def build_kwargs(combo, alibi, qq_bias, sinks):
    kwargs = {}
    if "softcap" in combo:
        kwargs["softcap"] = 20.0
    if "window" in combo:
        kwargs["window_size"] = (31, 0)
    if "alibi" in combo:
        kwargs["alibi_slopes"] = alibi
    if "bias" in combo:
        kwargs["qq_bias"] = qq_bias
    if "sink" in combo:
        kwargs["sinks"] = sinks
    return kwargs


def make_block_table(seq_lens, block_size, device):
    max_blocks_per_seq = max((seq_len + block_size - 1) // block_size for seq_len in seq_lens)
    total_blocks = sum((seq_len + block_size - 1) // block_size for seq_len in seq_lens)
    block_table = torch.zeros((len(seq_lens), max_blocks_per_seq), device=device, dtype=torch.int32)
    block_cursor = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        block_table[seq_idx, :num_blocks] = torch.arange(block_cursor, block_cursor + num_blocks, device=device, dtype=torch.int32)
        block_cursor += num_blocks
    return block_table, total_blocks


def quantize_int8_kv(k, v, k_scale, v_scale):
    k_int8 = torch.clamp(torch.round(k.float() / k_scale), -127, 127).to(torch.int8)
    v_int8 = torch.clamp(torch.round(v.float() / v_scale), -127, 127).to(torch.int8)
    return k_int8, v_int8


def unified_attention_case_names():
    names = [
        "single_seq_basic",
        "multi_seq_varlen",
        "multi_seq_varlen_softcap",
        "multi_seq_varlen_window",
        "multi_seq_varlen_alibi",
        "multi_seq_varlen_bias",
        "multi_seq_varlen_sink",
        "multi_seq_varlen_feature_combo",
        "multi_seq_varlen_window_bias_sink_combo",
        "multi_seq_varlen_window_softcap_alibi_combo",
        "multi_seq_varlen_int8_kv",
        "multi_seq_varlen_int8_softcap_alibi",
        "multi_seq_varlen_int8_window_bias_sink",
        "3d_decode_basic",
        "3d_decode_int8_basic",
        "3d_decode_int8_softcap",
        "3d_decode_softcap",
        "3d_decode_alibi",
        "3d_decode_int8_alibi",
        "3d_decode_window",
        "3d_decode_int8_window",
        "3d_decode_bias",
        "3d_decode_bias_largeqq",
        "3d_decode_int8_bias",
        "3d_decode_int8_bias_largeqq",
        "3d_decode_sink",
        "3d_decode_int8_sink",
        "3d_decode_softcap_alibi_combo",
        "3d_decode_int8_softcap_alibi_combo",
        "3d_decode_window_softcap_combo",
        "3d_decode_int8_window_softcap_combo",
        "3d_decode_window_alibi_combo",
        "3d_decode_int8_window_alibi_combo",
        "3d_decode_bias_sink_combo",
        "3d_decode_int8_bias_sink_combo",
    ]
    for combo in feature_subsets():
        suffix = "base" if not combo else "_".join(combo)
        names.append(f"2d_float_{suffix}")
        names.append(f"3d_float_{suffix}")
    for name in (
        "2d_prefill_gqa_ratio4_hd64_bs8",
        "2d_prefill_gqa_ratio4_hd128_bs16",
        "3d_decode_gqa_ratio4_hd64_bs8",
        "3d_decode_gqa_ratio4_hd128_bs16",
    ):
        names.append(name.replace("_hd", "_float_hd"))
        names.append(name)
    return names


def main(case_name=None):
    global _CASE_FILTER, _CASE_HIT
    _CASE_FILTER = case_name
    _CASE_HIT = False
    device = get_device()
    dtype = torch.float16
    torch.manual_seed(0)
    total_cases = 0

    num_query_heads = 32
    num_kv_heads = 4
    head_dim = 128
    block_size = 16
    single_seq_len = 256
    single_q_tokens = 8
    single_num_blocks = (single_seq_len + block_size - 1) // block_size

    q_single = torch.randn(single_q_tokens, num_query_heads, head_dim, device=device, dtype=dtype)
    k_single = torch.randn(single_num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_single = torch.randn(single_num_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    cu_single = torch.tensor([0, single_q_tokens], device=device, dtype=torch.int32)
    seqused_single = torch.tensor([single_seq_len], device=device, dtype=torch.int32)
    block_table_single = torch.arange(single_num_blocks, device=device, dtype=torch.int32).unsqueeze(0)
    run_case(
        "single_seq_basic",
        q_single,
        k_single,
        v_single,
        cu_single,
        seqused_single,
        block_table_single,
        single_q_tokens,
    )
    total_cases += 1

    seq_lens = [192, 144]
    q_lens = [8, 12]
    max_blocks_per_seq = max((seq_len + block_size - 1) // block_size for seq_len in seq_lens)
    total_blocks = sum((seq_len + block_size - 1) // block_size for seq_len in seq_lens)
    total_q_tokens = sum(q_lens)

    q_varlen = torch.randn(total_q_tokens, num_query_heads, head_dim, device=device, dtype=dtype)
    k_varlen = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_varlen = torch.randn(total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    cu_varlen = torch.tensor([0, q_lens[0], q_lens[0] + q_lens[1]], device=device, dtype=torch.int32)
    seqused_varlen = torch.tensor(seq_lens, device=device, dtype=torch.int32)

    block_table_varlen = torch.zeros((len(seq_lens), max_blocks_per_seq), device=device, dtype=torch.int32)
    block_cursor = 0
    for seq_idx, seq_len in enumerate(seq_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        block_table_varlen[seq_idx, :num_blocks] = torch.arange(block_cursor, block_cursor + num_blocks, device=device, dtype=torch.int32)
        block_cursor += num_blocks

    run_case(
        "multi_seq_varlen",
        q_varlen,
        k_varlen,
        v_varlen,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
    )
    total_cases += 1

    run_case(
        "multi_seq_varlen_softcap",
        q_varlen,
        k_varlen,
        v_varlen,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        softcap=20.0,
    )
    total_cases += 1

    run_case(
        "multi_seq_varlen_window",
        q_varlen,
        k_varlen,
        v_varlen,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        window_size=(31, 0),
    )
    total_cases += 1

    alibi_slopes = torch.linspace(0.01, 0.32, num_query_heads, device=device, dtype=torch.float32)
    run_case(
        "multi_seq_varlen_alibi",
        q_varlen,
        k_varlen,
        v_varlen,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        alibi_slopes=alibi_slopes,
    )
    total_cases += 1

    qq_bias = torch.randn(max(q_lens), max(q_lens), device=device, dtype=torch.float32) * 0.05
    run_case(
        "multi_seq_varlen_bias",
        q_varlen,
        k_varlen,
        v_varlen,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        qq_bias=qq_bias,
    )
    total_cases += 1

    sinks = torch.linspace(-0.2, 0.2, num_query_heads, device=device, dtype=torch.float32)
    run_case(
        "multi_seq_varlen_sink",
        q_varlen,
        k_varlen,
        v_varlen,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        sinks=sinks,
    )
    total_cases += 1

    run_case(
        "multi_seq_varlen_feature_combo",
        q_varlen,
        k_varlen,
        v_varlen,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        softcap=20.0,
        alibi_slopes=alibi_slopes,
        qq_bias=qq_bias,
        sinks=sinks,
    )
    total_cases += 1

    run_case(
        "multi_seq_varlen_window_bias_sink_combo",
        q_varlen,
        k_varlen,
        v_varlen,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        window_size=(31, 0),
        qq_bias=qq_bias,
        sinks=sinks,
    )
    total_cases += 1

    run_case(
        "multi_seq_varlen_window_softcap_alibi_combo",
        q_varlen,
        k_varlen,
        v_varlen,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        window_size=(31, 0),
        softcap=20.0,
        alibi_slopes=alibi_slopes,
    )
    total_cases += 1

    k_int8_scale = 0.02
    v_int8_scale = 0.03
    k_varlen_int8 = torch.clamp(torch.round(k_varlen.float() / k_int8_scale), -127, 127).to(torch.int8)
    v_varlen_int8 = torch.clamp(torch.round(v_varlen.float() / v_int8_scale), -127, 127).to(torch.int8)
    run_case(
        "multi_seq_varlen_int8_kv",
        q_varlen,
        k_varlen_int8,
        v_varlen_int8,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        k_descale=torch.tensor(k_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1
    run_case(
        "multi_seq_varlen_int8_softcap_alibi",
        q_varlen,
        k_varlen_int8,
        v_varlen_int8,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        softcap=20.0,
        alibi_slopes=alibi_slopes,
        k_descale=torch.tensor(k_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1
    run_case(
        "multi_seq_varlen_int8_window_bias_sink",
        q_varlen,
        k_varlen_int8,
        v_varlen_int8,
        cu_varlen,
        seqused_varlen,
        block_table_varlen,
        max(q_lens),
        window_size=(31, 0),
        qq_bias=qq_bias,
        sinks=sinks,
        k_descale=torch.tensor(k_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    decode_seq_lens = [33, 17]
    decode_q_lens = [1, 1]
    decode_max_blocks_per_seq = max((seq_len + block_size - 1) // block_size for seq_len in decode_seq_lens)
    decode_total_blocks = sum((seq_len + block_size - 1) // block_size for seq_len in decode_seq_lens)
    decode_total_q = sum(decode_q_lens)

    q_decode = torch.randn(decode_total_q, num_query_heads, head_dim, device=device, dtype=dtype)
    k_decode = torch.randn(decode_total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    v_decode = torch.randn(decode_total_blocks, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    cu_decode = torch.tensor([0, 1, 2], device=device, dtype=torch.int32)
    seqused_decode = torch.tensor(decode_seq_lens, device=device, dtype=torch.int32)
    block_table_decode = torch.zeros((len(decode_seq_lens), decode_max_blocks_per_seq), device=device, dtype=torch.int32)
    decode_block_cursor = 0
    for seq_idx, seq_len in enumerate(decode_seq_lens):
        num_blocks = (seq_len + block_size - 1) // block_size
        block_table_decode[seq_idx, :num_blocks] = torch.arange(
            decode_block_cursor, decode_block_cursor + num_blocks, device=device, dtype=torch.int32
        )
        decode_block_cursor += num_blocks

    run_case(
        "3d_decode_basic",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
    )
    total_cases += 1

    k_decode_int8_scale = 0.02
    v_decode_int8_scale = 0.03
    k_decode_int8 = torch.clamp(torch.round(k_decode.float() / k_decode_int8_scale), -127, 127).to(torch.int8)
    v_decode_int8 = torch.clamp(torch.round(v_decode.float() / v_decode_int8_scale), -127, 127).to(torch.int8)
    run_case(
        "3d_decode_int8_basic",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_softcap",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        softcap=20.0,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    run_case(
        "3d_decode_softcap",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        softcap=20.0,
    )
    total_cases += 1

    decode_alibi = torch.linspace(0.01, 0.32, num_query_heads, device=device, dtype=torch.float32)
    run_case(
        "3d_decode_alibi",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        alibi_slopes=decode_alibi,
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_alibi",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        alibi_slopes=decode_alibi,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    run_case(
        "3d_decode_window",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        window_size=(31, 0),
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_window",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        window_size=(31, 0),
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    decode_qq_bias = torch.randn(1, 1, device=device, dtype=torch.float32) * 0.05
    decode_qq_bias_large = torch.randn(4, 4, device=device, dtype=torch.float32) * 0.05
    run_case(
        "3d_decode_bias",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        qq_bias=decode_qq_bias,
    )
    total_cases += 1
    run_case(
        "3d_decode_bias_largeqq",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        qq_bias=decode_qq_bias_large,
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_bias",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        qq_bias=decode_qq_bias,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_bias_largeqq",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        qq_bias=decode_qq_bias_large,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    decode_sinks = torch.linspace(-0.2, 0.2, num_query_heads, device=device, dtype=torch.float32)
    run_case(
        "3d_decode_sink",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        sinks=decode_sinks,
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_sink",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        sinks=decode_sinks,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    run_case(
        "3d_decode_softcap_alibi_combo",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        softcap=20.0,
        alibi_slopes=decode_alibi,
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_softcap_alibi_combo",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        softcap=20.0,
        alibi_slopes=decode_alibi,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    run_case(
        "3d_decode_window_softcap_combo",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        window_size=(31, 0),
        softcap=20.0,
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_window_softcap_combo",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        window_size=(31, 0),
        softcap=20.0,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    run_case(
        "3d_decode_window_alibi_combo",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        window_size=(31, 0),
        alibi_slopes=decode_alibi,
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_window_alibi_combo",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        window_size=(31, 0),
        alibi_slopes=decode_alibi,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    run_case(
        "3d_decode_bias_sink_combo",
        q_decode,
        k_decode,
        v_decode,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        qq_bias=decode_qq_bias,
        sinks=decode_sinks,
    )
    total_cases += 1
    run_case(
        "3d_decode_int8_bias_sink_combo",
        q_decode,
        k_decode_int8,
        v_decode_int8,
        cu_decode,
        seqused_decode,
        block_table_decode,
        1,
        qq_bias=decode_qq_bias,
        sinks=decode_sinks,
        k_descale=torch.tensor(k_decode_int8_scale, device=device, dtype=torch.float32),
        v_descale=torch.tensor(v_decode_int8_scale, device=device, dtype=torch.float32),
    )
    total_cases += 1

    combo_block_table_2d, combo_total_blocks_2d = make_block_table(seq_lens, block_size, device)
    combo_q_2d = torch.randn(total_q_tokens, num_query_heads, head_dim, device=device, dtype=dtype)
    combo_k_2d = torch.randn(combo_total_blocks_2d, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    combo_v_2d = torch.randn(combo_total_blocks_2d, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    combo_cu_2d = cu_varlen
    combo_seq_2d = seqused_varlen
    combo_alibi_2d = alibi_slopes
    combo_bias_2d = qq_bias
    combo_sinks_2d = sinks

    combo_block_table_3d, combo_total_blocks_3d = make_block_table(decode_seq_lens, block_size, device)
    combo_q_3d = torch.randn(decode_total_q, num_query_heads, head_dim, device=device, dtype=dtype)
    combo_k_3d = torch.randn(combo_total_blocks_3d, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    combo_v_3d = torch.randn(combo_total_blocks_3d, block_size, num_kv_heads, head_dim, device=device, dtype=dtype)
    combo_cu_3d = cu_decode
    combo_seq_3d = seqused_decode
    combo_alibi_3d = decode_alibi
    combo_bias_3d = decode_qq_bias_large
    combo_sinks_3d = decode_sinks

    for combo in feature_subsets():
        suffix = "base" if not combo else "_".join(combo)
        run_case(
            f"2d_float_{suffix}",
            combo_q_2d,
            combo_k_2d,
            combo_v_2d,
            combo_cu_2d,
            combo_seq_2d,
            combo_block_table_2d,
            max(q_lens),
            **build_kwargs(combo, combo_alibi_2d, combo_bias_2d, combo_sinks_2d),
        )
        total_cases += 1
        run_case(
            f"3d_float_{suffix}",
            combo_q_3d,
            combo_k_3d,
            combo_v_3d,
            combo_cu_3d,
            combo_seq_3d,
            combo_block_table_3d,
            1,
            **build_kwargs(combo, combo_alibi_3d, combo_bias_3d, combo_sinks_3d),
        )
        total_cases += 1

    def make_gqa_case(seq_lens_case, query_lens_case, head_dim_case, block_size_case, use_alibi, use_qq_bias, use_sinks):
        total_q = sum(query_lens_case)
        block_table_case, total_blocks_case = make_block_table(seq_lens_case, block_size_case, device)
        q_case = torch.randn(total_q, 8, head_dim_case, device=device, dtype=dtype)
        k_case = torch.randn(total_blocks_case, block_size_case, 2, head_dim_case, device=device, dtype=dtype)
        v_case = torch.randn(total_blocks_case, block_size_case, 2, head_dim_case, device=device, dtype=dtype)
        cu_vals = [0]
        for q_len in query_lens_case:
            cu_vals.append(cu_vals[-1] + q_len)
        cu_case = torch.tensor(cu_vals, device=device, dtype=torch.int32)
        seqused_case = torch.tensor(seq_lens_case, device=device, dtype=torch.int32)
        alibi_case = torch.linspace(-0.05, 0.05, 8, device=device, dtype=torch.float32) if use_alibi else None
        qq_bias_case = (
            torch.randn(max(query_lens_case), max(query_lens_case), device=device, dtype=torch.float32) * 0.1 if use_qq_bias else None
        )
        sinks_case = torch.linspace(-0.2, 0.2, 8, device=device, dtype=torch.float32) if use_sinks else None
        return q_case, k_case, v_case, cu_case, seqused_case, block_table_case, alibi_case, qq_bias_case, sinks_case

    gqa_configs = [
        {
            "case_name": "2d_prefill_gqa_ratio4_hd64_bs8",
            "seq_lens": [25, 19],
            "query_lens": [5, 4],
            "head_dim": 64,
            "block_size": 8,
            "window_size": (5, 0),
            "softcap": 0.3,
            "max_seqlen_q": 5,
        },
        {
            "case_name": "2d_prefill_gqa_ratio4_hd128_bs16",
            "seq_lens": [33, 21],
            "query_lens": [6, 5],
            "head_dim": 128,
            "block_size": 16,
            "window_size": (7, 0),
            "softcap": 0.3,
            "max_seqlen_q": 6,
        },
        {
            "case_name": "3d_decode_gqa_ratio4_hd64_bs8",
            "seq_lens": [21, 14],
            "query_lens": [1, 1],
            "head_dim": 64,
            "block_size": 8,
            "window_size": (4, 0),
            "softcap": 0.4,
            "max_seqlen_q": 1,
        },
        {
            "case_name": "3d_decode_gqa_ratio4_hd128_bs16",
            "seq_lens": [49, 37],
            "query_lens": [1, 1],
            "head_dim": 128,
            "block_size": 16,
            "window_size": (8, 0),
            "softcap": 0.4,
            "max_seqlen_q": 1,
        },
    ]

    for cfg in gqa_configs:
        q_gqa, k_gqa, v_gqa, cu_gqa, seqused_gqa, block_table_gqa, alibi_gqa, qq_bias_gqa, sinks_gqa = make_gqa_case(
            cfg["seq_lens"], cfg["query_lens"], cfg["head_dim"], cfg["block_size"], True, True, True
        )
        run_case(
            cfg["case_name"].replace("_hd", "_float_hd"),
            q_gqa,
            k_gqa,
            v_gqa,
            cu_gqa,
            seqused_gqa,
            block_table_gqa,
            cfg["max_seqlen_q"],
            window_size=cfg["window_size"],
            softcap=cfg["softcap"],
            alibi_slopes=alibi_gqa,
            qq_bias=qq_bias_gqa,
            sinks=sinks_gqa,
        )
        total_cases += 1

        k_gqa_int8, v_gqa_int8 = quantize_int8_kv(k_gqa, v_gqa, 0.02, 0.03)
        run_case(
            cfg["case_name"],
            q_gqa,
            k_gqa_int8,
            v_gqa_int8,
            cu_gqa,
            seqused_gqa,
            block_table_gqa,
            cfg["max_seqlen_q"],
            window_size=cfg["window_size"],
            softcap=cfg["softcap"],
            alibi_slopes=alibi_gqa,
            qq_bias=qq_bias_gqa,
            sinks=sinks_gqa,
            k_descale=torch.tensor(0.02, device=device, dtype=torch.float32),
            v_descale=torch.tensor(0.03, device=device, dtype=torch.float32),
        )
        total_cases += 1

    print(f"all tilelang basic cases passed ({total_cases} cases)")


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("case_name", unified_attention_case_names(), ids=unified_attention_case_names())
def test_tilelang_unified_attention(case_name):
    try:
        main(case_name)
    except _SelectedCaseDone:
        pass
    finally:
        global _CASE_FILTER
        _CASE_FILTER = None
    assert _CASE_HIT, f"Unified attention case was not found: {case_name}"


if __name__ == "__main__":
    tilelang.testing.main()
