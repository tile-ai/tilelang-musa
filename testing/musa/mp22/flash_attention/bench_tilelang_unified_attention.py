import os
import random
import time

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

import torch

import tilelang_unified_attention as ua_impl
from torch_unified_attention import unified_attention as torch_unified_attention


def get_device():
    if hasattr(torch, "musa"):
        try:
            if torch.musa.is_available():
                return "musa"
        except Exception:
            pass
    if torch.cuda.is_available():
        return "cuda"
    return "cpu"


def get_sync():
    if get_device() == "musa":
        return torch.musa.synchronize
    if get_device() == "cuda":
        return torch.cuda.synchronize
    return lambda: None


def get_event():
    if get_device() == "musa":
        return torch.musa.Event
    if get_device() == "cuda":
        return torch.cuda.Event
    return None


def benchmark_kernel(func, *args, num_iters=100, num_warmup=10, **kwargs):
    sync = get_sync()
    Event = get_event()

    if Event is None:
        for _ in range(num_warmup):
            func(*args, **kwargs)
        sync()
        start = time.perf_counter()
        for _ in range(num_iters):
            func(*args, **kwargs)
        sync()
        return (time.perf_counter() - start) * 1000.0 / num_iters

    for _ in range(num_warmup):
        func(*args, **kwargs)
    sync()

    start_event = Event(enable_timing=True)
    end_event = Event(enable_timing=True)
    start_event.record()
    for _ in range(num_iters):
        func(*args, **kwargs)
    end_event.record()
    sync()
    return start_event.elapsed_time(end_event) / num_iters


def build_3d_stage_runners(
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    *,
    softcap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    del max_seqlen_q, causal, q_descale

    num_seqs = q.shape[0]
    total_num_q_tokens = q.shape[0]
    num_query_heads = q.shape[1]
    head_size = q.shape[2]
    num_total_blocks = k_cache.shape[0]
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = max(1, num_query_heads // num_kv_heads)
    max_num_blocks_per_seq = block_table.shape[1]
    is_int8_kv = k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    sliding_window = -1 if window_size[0] < 0 else 1 + int(window_size[0])

    use_alibi = False
    use_sink = False
    use_qq_bias = False
    use_softcap = float(softcap) > 0.0
    alibi_tensor, sink_tensor, qq_bias_tensor = ua_impl._prepare_feature_tensors(q, None, None, None, False, max_seqlen_q=None)

    num_segments = ua_impl._default_num_segments(max_seqlen_k)
    split_kwargs, reduce_kwargs, cache_key = ua_impl._get_cached_3d_plan(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        num_seqs=num_seqs,
        total_num_q_tokens=total_num_q_tokens,
        num_total_blocks=num_total_blocks,
        block_size=block_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        softmax_scale=softmax_scale,
        softcap=softcap,
        use_softcap=use_softcap,
        sliding_window=sliding_window,
        use_alibi=use_alibi,
        use_sink=use_sink,
        use_qq_bias=use_qq_bias,
        num_segments=num_segments,
        is_int8_kv=is_int8_kv,
    )
    split_kernel, reduce_kernel = ua_impl._get_cached_3d_kernels(
        cache_key,
        split_kwargs,
        reduce_kwargs,
        is_int8_kv=is_int8_kv,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    block_h = split_kwargs["block_h"]
    segm_output, segm_max, segm_expsum = ua_impl._get_cached_3d_workspace(
        device=q.device,
        num_seqs=num_seqs,
        num_kv_heads=num_kv_heads,
        num_segments=num_segments,
        block_h=block_h,
        head_size=head_size,
        total_num_q_tokens=total_num_q_tokens,
        num_query_heads=num_query_heads,
    )

    split_args = (
        q,
        k_cache,
        v_cache,
        block_table,
        seqused_k,
        cu_seqlens_q,
        alibi_tensor,
        sink_tensor,
        qq_bias_tensor,
    )

    def run_split():
        split_kernel(*split_args, segm_output, segm_max, segm_expsum)

    def run_reduce():
        reduce_kernel(segm_output, segm_max, segm_expsum, cu_seqlens_q, out)

    return run_split, run_reduce


def build_3d_fused_runner(
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    max_seqlen_q,
    seqused_k,
    max_seqlen_k,
    softmax_scale,
    causal,
    window_size,
    block_table,
    *,
    softcap=0.0,
    q_descale=None,
    k_descale=None,
    v_descale=None,
):
    del max_seqlen_q, causal, q_descale

    num_seqs = q.shape[0]
    total_num_q_tokens = q.shape[0]
    num_query_heads = q.shape[1]
    head_size = q.shape[2]
    num_total_blocks = k_cache.shape[0]
    block_size = k_cache.shape[1]
    num_kv_heads = k_cache.shape[2]
    num_queries_per_kv = max(1, num_query_heads // num_kv_heads)
    max_num_blocks_per_seq = block_table.shape[1]
    is_int8_kv = k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8
    sliding_window = -1 if window_size[0] < 0 else 1 + int(window_size[0])

    use_alibi = False
    use_sink = False
    use_qq_bias = False
    use_softcap = float(softcap) > 0.0
    alibi_tensor, sink_tensor, qq_bias_tensor = ua_impl._prepare_feature_tensors(q, None, None, None, False, max_seqlen_q=None)

    num_segments = ua_impl._default_num_segments(max_seqlen_k)
    if num_segments != 1:
        raise ValueError("Direct fused runner is only valid when num_segments == 1")

    split_kwargs, _reduce_kwargs, cache_key = ua_impl._get_cached_3d_plan(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        num_seqs=num_seqs,
        total_num_q_tokens=total_num_q_tokens,
        num_total_blocks=num_total_blocks,
        block_size=block_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        softmax_scale=softmax_scale,
        softcap=softcap,
        use_softcap=use_softcap,
        sliding_window=sliding_window,
        use_alibi=use_alibi,
        use_sink=use_sink,
        use_qq_bias=use_qq_bias,
        num_segments=num_segments,
        is_int8_kv=is_int8_kv,
    )
    fused_kernel = ua_impl._get_cached_3d_fused_kernel(
        cache_key,
        split_kwargs,
        is_int8_kv=is_int8_kv,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    fused_args = (
        q,
        k_cache,
        v_cache,
        block_table,
        seqused_k,
        cu_seqlens_q,
        alibi_tensor,
        sink_tensor,
        qq_bias_tensor,
        out,
    )

    def run_fused():
        fused_kernel(*fused_args)

    return run_fused


def build_3d_direct_runner(*args, **kwargs):
    max_seqlen_k = args[7]
    if ua_impl._default_num_segments(max_seqlen_k) == 1 and os.getenv("UA_DISABLE_3D_FUSED", "0") != "1":
        return build_3d_fused_runner(*args, **kwargs), "Direct Fused"

    split_runner, reduce_runner = build_3d_stage_runners(*args, **kwargs)

    def run_split_reduce():
        split_runner()
        reduce_runner()

    return run_split_reduce, "Direct Split+Reduce"


def main():
    os.environ.pop("UA_V_SPLIT", None)
    os.environ.pop("UA_HEAD_TILE", None)
    os.environ.pop("UA_QK_SPLIT", None)
    batch_size = 1
    num_query_heads = 32
    num_kv_heads = 4
    head_dim = 128
    device = get_device()
    dtype = torch.float16

    seq_lens_env = os.getenv("UA_BENCH_SEQ_LENS")
    if seq_lens_env:
        seq_lens = [int(x.strip()) for x in seq_lens_env.split(",") if x.strip()]
    else:
        seq_lens = [256, 1024, 2048, 4096, 8192, 16384]
    stage_timing = os.getenv("UA_BENCH_STAGE_TIMING", "0") == "1"
    configs_by_block = {
        16: [
            {"block_n": 32, "kv_threads": 128},
        ],
    }

    print(f"Batch Size: {batch_size}, Q_Heads: {num_query_heads}, KV_Heads: {num_kv_heads}, Dim: {head_dim}")
    print(
        "Config: "
        f"UA_KV_FRAGMENT={os.getenv('UA_KV_FRAGMENT', '<impl default:1>')} "
        f"UA_NUM_SEGMENTS={os.getenv('UA_NUM_SEGMENTS', '<default>')} "
        f"UA_DISABLE_3D_FUSED={os.getenv('UA_DISABLE_3D_FUSED', '0')} "
        f"UA_BENCH_STAGE_TIMING={int(stage_timing)} "
        "V_LAYOUT=non_transposed"
    )
    print("-" * 50)

    random.seed(0)
    torch.manual_seed(0)
    if device == "musa" and hasattr(torch, "musa"):
        torch.musa.manual_seed(0)

    for block_size in configs_by_block:
        print(f"Block Size: {block_size}")
        for cfg in configs_by_block[block_size]:
            block_n = cfg["block_n"]
            kv_threads = cfg["kv_threads"]
            print(f"BLOCK_N: {block_n}, KV Threads: {kv_threads}")
            os.environ["UA_BLOCK_N"] = str(block_n)
            os.environ["UA_KV_THREADS"] = str(kv_threads)
            for run_idx in range(2):
                if run_idx == 0:
                    print("Pass 1: warm cache")
                else:
                    print("Pass 2: measured")
                for seq_len in seq_lens:
                    max_blocks_per_seq = (seq_len + block_size - 1) // block_size
                    num_blocks = batch_size * max_blocks_per_seq
                    k_cache = torch.randn(
                        (num_blocks, block_size, num_kv_heads, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    v_cache = torch.randn(
                        (num_blocks, block_size, num_kv_heads, head_dim),
                        dtype=dtype,
                        device=device,
                    )
                    q = torch.randn((batch_size, num_query_heads, head_dim), dtype=dtype, device=device)
                    out = torch.empty_like(q)
                    args = (
                        q,
                        k_cache,
                        v_cache,
                        out,
                        torch.arange(0, batch_size + 1, dtype=torch.int32, device=device),
                        1,
                        torch.full((batch_size,), seq_len, dtype=torch.int32, device=device),
                        seq_len,
                        1.0 / (head_dim**0.5),
                        True,
                        (-1, -1),
                        torch.arange(0, num_blocks, dtype=torch.int32, device=device).view(batch_size, max_blocks_per_seq),
                    )
                    kwargs = {
                        "softcap": 0.0,
                        "q_descale": None,
                        "k_descale": None,
                        "v_descale": None,
                    }
                    try:
                        lat = benchmark_kernel(ua_impl.unified_attention, *args, **kwargs)
                        if run_idx == 1:
                            print(f"Seq Len: {seq_len:5d} | Latency: {lat:.3f} ms")
                            if stage_timing:
                                direct_runner, direct_label = build_3d_direct_runner(*args, **kwargs)
                                direct_lat = benchmark_kernel(direct_runner, num_iters=1000, num_warmup=20)
                                print(f"Seq Len: {seq_len:5d} | {direct_label}: {direct_lat:.3f} ms")
                                split_runner, reduce_runner = build_3d_stage_runners(
                                    *args,
                                    **kwargs,
                                )
                                split_lat = benchmark_kernel(split_runner, num_iters=1000, num_warmup=20)
                                reduce_lat = benchmark_kernel(reduce_runner, num_iters=1000, num_warmup=20)
                                print(f"Seq Len: {seq_len:5d} | Stage Split: {split_lat:.3f} ms | Reduce: {reduce_lat:.3f} ms")
                            ua_impl.unified_attention(*args, **kwargs)
                            q_ref = q.clone()
                            k_ref = k_cache.clone()
                            v_ref = v_cache.clone()
                            out_ref = torch.empty_like(out)
                            cu_seqlens_q = args[4]
                            max_seqlen_q = args[5]
                            seqused_k = args[6]
                            max_seqlen_k = args[7]
                            softmax_scale = args[8]
                            causal = args[9]
                            sliding_window = args[10]
                            block_table_ref = args[11]
                            try:
                                torch_unified_attention(
                                    q_ref,
                                    k_ref,
                                    v_ref,
                                    out_ref,
                                    cu_seqlens_q,
                                    max_seqlen_q,
                                    seqused_k,
                                    max_seqlen_k,
                                    softmax_scale,
                                    causal,
                                    sliding_window,
                                    block_table_ref,
                                    **kwargs,
                                )
                                torch.testing.assert_close(out, out_ref, atol=1e-3, rtol=1e-2)
                                max_diff = (out - out_ref).abs().max().item()
                                mean_diff = (out - out_ref).abs().mean().item()
                                print(f"Seq Len: {seq_len:5d} | Accuracy Torch: PASS max_diff={max_diff:.6f} mean_diff={mean_diff:.6f}")
                            except Exception as e:
                                print(f"Seq Len: {seq_len:5d} | Accuracy Torch: FAILED ({e})")
                    except Exception as e:
                        if run_idx == 1:
                            print(f"Seq Len: {seq_len:5d} | Latency: FAILED ({e})")
            print("-" * 50)
    os.environ.pop("UA_BLOCK_N", None)
    os.environ.pop("UA_KV_THREADS", None)
    os.environ.pop("UA_V_SPLIT", None)
    os.environ.pop("UA_HEAD_TILE", None)
    os.environ.pop("UA_QK_SPLIT", None)


if __name__ == "__main__":
    main()
