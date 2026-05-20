import os

import torch

try:
    import torch_musa  # noqa: F401
except ImportError:
    torch_musa = None

import tilelang
import tilelang.language as T

_3D_KERNEL_CACHE = {}
_2D_KERNEL_CACHE = {}
_3D_WORKSPACE_CACHE = {}
_3D_PLAN_CACHE = {}
_3D_FUSED_KERNEL_CACHE = {}
_FEATURE_DUMMY_CACHE = {}

_UA_GEMV_EXTRA_MCC_FLAGS = [
    "-save-temps=obj",
    "-fverbose-asm",
    "-O3",
    "-ffast-math",
    "-fmusa-flush-denormals-to-zero",
    "-mllvm",
    "-mtgpu-enable-max-ilp-scheduling-strategy=0",
    "-mllvm",
    "-mtgpu-enchanced-minreg-schedule=1",
    "-mllvm",
    "-mtgpu-enable-cse=0",
]


def _cdiv(x, y):
    return (x + y - 1) // y


def _fallback_num_segments(max_seqlen_k):
    return 8 if max_seqlen_k >= 4096 else (4 if max_seqlen_k >= 2048 else 1)


def _default_num_segments(max_seqlen_k):
    env_num_segments = os.getenv("UA_NUM_SEGMENTS")
    if env_num_segments is not None and env_num_segments != "":
        return int(env_num_segments)
    return _fallback_num_segments(max_seqlen_k)


def _scalar_from_descale(descale):
    if descale is None:
        return None
    if torch.is_tensor(descale):
        return float(descale.item())
    return float(descale)


def _resolve_3d_runtime_config(
    *,
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    total_num_q_tokens,
    num_total_blocks,
    block_size,
    max_num_blocks_per_seq,
    softmax_scale,
    softcap,
    use_softcap,
    sliding_window,
    use_alibi,
    use_sink,
    use_qq_bias,
    num_segments,
    is_int8_kv,
):
    reduce_threads = int(os.getenv("UA_REDUCE_THREADS", "1024"))
    block_h = 32
    valid_block_h = min(block_h, num_queries_per_kv)
    block_n = int(os.getenv("UA_BLOCK_N", str(max(32, ((block_size + 31) // 32) * 32))))
    vec = 16
    kv_threads = int(
        os.getenv(
            "UA_KV_THREADS",
            str(max(128, block_size * _cdiv(head_size, vec)) if is_int8_kv else 128),
        )
    )
    split_num_stages = int(os.getenv("UA_NUM_STAGES", "2"))
    if block_n % block_size != 0 or (block_n // block_size) not in (1, 2, 4):
        raise ValueError(
            f"Unsupported BLOCK_N/block_size combination for current split kernel: "
            f"BLOCK_N={block_n}, block_size={block_size}. "
            "Current implementation only supports 1, 2, or 4 logical block slots per KV tile."
        )
    split_kwargs = dict(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        num_seqs=num_seqs,
        total_num_q_tokens=total_num_q_tokens,
        num_total_blocks=num_total_blocks,
        block_size=block_size,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        num_segments=num_segments,
        softmax_scale=softmax_scale,
        softcap=float(softcap or 0.0),
        use_softcap=use_softcap,
        use_alibi=use_alibi,
        use_sink=use_sink,
        use_qq_bias=use_qq_bias,
        sliding_window=sliding_window,
        use_window=sliding_window >= 0,
        block_h=block_h,
        valid_block_h=valid_block_h,
        BLOCK_N=block_n,
        threads=kv_threads,
        num_stages=split_num_stages,
    )
    reduce_kwargs = dict(
        total_num_q_tokens=total_num_q_tokens,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        num_seqs=num_seqs,
        num_segments=num_segments,
        block_h=block_h,
        valid_block_h=valid_block_h,
        threads=reduce_threads,
        out_dtype="float16",
    )
    cache_key = (
        "3d",
        tuple(sorted(split_kwargs.items())),
        tuple(sorted(reduce_kwargs.items())),
        is_int8_kv,
    )
    return split_kwargs, reduce_kwargs, cache_key


def _get_cached_3d_plan(
    *,
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    total_num_q_tokens,
    num_total_blocks,
    block_size,
    max_num_blocks_per_seq,
    softmax_scale,
    softcap,
    use_softcap,
    sliding_window,
    use_alibi,
    use_sink,
    use_qq_bias,
    num_segments,
    is_int8_kv,
):
    env_key = (
        os.getenv("UA_REDUCE_THREADS", "1024"),
        os.getenv("UA_BLOCK_N"),
        os.getenv("UA_KV_THREADS"),
        os.getenv("UA_NUM_STAGES", "2"),
        os.getenv("UA_Q_LOAD_ROWS"),
    )
    key = (
        num_query_heads,
        num_kv_heads,
        num_queries_per_kv,
        head_size,
        num_seqs,
        total_num_q_tokens,
        num_total_blocks,
        block_size,
        max_num_blocks_per_seq,
        float(softmax_scale),
        float(softcap or 0.0),
        bool(use_softcap),
        int(sliding_window),
        bool(use_alibi),
        bool(use_sink),
        bool(use_qq_bias),
        num_segments,
        bool(is_int8_kv),
        env_key,
    )
    plan = _3D_PLAN_CACHE.get(key)
    if plan is None:
        plan = _resolve_3d_runtime_config(
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
        _3D_PLAN_CACHE[key] = plan
    return plan


def _get_cached_3d_kernels(cache_key, split_kwargs, reduce_kwargs, *, is_int8_kv, k_descale, v_descale):
    kernels = _3D_KERNEL_CACHE.get(cache_key)
    if kernels is None:
        split_kernel = tilelang_unified_attention_3d_split_basic(
            **split_kwargs,
            is_int8_kv=is_int8_kv,
            k_descale=1.0 if k_descale is None else k_descale,
            v_descale=1.0 if v_descale is None else v_descale,
        )
        reduce_kernel = tilelang_unified_attention_3d_reduce_basic(**reduce_kwargs)
        kernels = (split_kernel, reduce_kernel)
        _3D_KERNEL_CACHE[cache_key] = kernels
    return kernels


def _get_cached_3d_nofeature_kernels(cache_key, split_kwargs, reduce_kwargs, *, is_int8_kv, k_descale, v_descale):
    nofeature_cache_key = ("3d_nofeature", cache_key)
    kernels = _3D_KERNEL_CACHE.get(nofeature_cache_key)
    if kernels is None:
        nofeature_split_kwargs = {
            key: value
            for key, value in split_kwargs.items()
            if key
            not in (
                "softcap",
                "use_softcap",
                "use_alibi",
                "use_sink",
                "use_qq_bias",
                "sliding_window",
                "use_window",
            )
        }
        split_kernel = tilelang_unified_attention_3d_split_nofeature_basic(
            **nofeature_split_kwargs,
            is_int8_kv=is_int8_kv,
            k_descale=1.0 if k_descale is None else k_descale,
            v_descale=1.0 if v_descale is None else v_descale,
        )
        reduce_kernel = tilelang_unified_attention_3d_reduce_basic(**reduce_kwargs)
        kernels = (split_kernel, reduce_kernel)
        _3D_KERNEL_CACHE[nofeature_cache_key] = kernels
    return kernels


def _get_cached_3d_fused_kernel(cache_key, split_kwargs, *, is_int8_kv, k_descale, v_descale):
    fused_cache_key = ("3d_fused", cache_key)
    kernel = _3D_FUSED_KERNEL_CACHE.get(fused_cache_key)
    if kernel is None:
        kernel = tilelang_unified_attention_3d_fused_basic(
            **split_kwargs,
            is_int8_kv=is_int8_kv,
            k_descale=1.0 if k_descale is None else k_descale,
            v_descale=1.0 if v_descale is None else v_descale,
        )
        _3D_FUSED_KERNEL_CACHE[fused_cache_key] = kernel
    return kernel


def _get_cached_2d_kernel(kernel_fn, kwargs):
    kernel_name = getattr(kernel_fn, "__name__", None)
    if kernel_name is None:
        kernel_name = getattr(kernel_fn, "func", kernel_fn).__name__
    cache_key = ("2d", kernel_name, tuple(sorted(kwargs.items())))
    kernel = _2D_KERNEL_CACHE.get(cache_key)
    if kernel is None:
        kernel = kernel_fn(**kwargs)
        _2D_KERNEL_CACHE[cache_key] = kernel
    return kernel


def _get_cached_3d_workspace(*, device, num_seqs, num_kv_heads, num_segments, block_h, head_size, total_num_q_tokens, num_query_heads):
    key = (
        str(device),
        num_seqs,
        num_kv_heads,
        num_segments,
        block_h,
        head_size,
        total_num_q_tokens,
        num_query_heads,
    )
    ws = _3D_WORKSPACE_CACHE.get(key)
    if ws is None:
        segm_output = torch.empty(
            (num_seqs, num_kv_heads, num_segments, block_h, head_size),
            dtype=torch.float32,
            device=device,
        )
        segm_max = torch.empty(
            (num_seqs, num_kv_heads, num_segments, block_h),
            dtype=torch.float32,
            device=device,
        )
        segm_expsum = torch.empty(
            (num_seqs, num_kv_heads, num_segments, block_h),
            dtype=torch.float32,
            device=device,
        )
        ws = (segm_output, segm_max, segm_expsum)
        _3D_WORKSPACE_CACHE[key] = ws
    return ws


def _get_feature_dummy_tensors(q):
    num_query_heads = int(q.shape[1])
    key = (str(q.device), num_query_heads)
    cached = _FEATURE_DUMMY_CACHE.get(key)
    if cached is None:
        dummy_headwise = torch.empty((num_query_heads,), device=q.device, dtype=torch.float32)
        dummy_scalar = torch.empty((1, 1), device=q.device, dtype=torch.float32)
        cached = (dummy_headwise, dummy_scalar)
        _FEATURE_DUMMY_CACHE[key] = cached
    return cached


def _prepare_feature_tensors(q, alibi_slopes, sinks, qq_bias, use_2d, max_seqlen_q=None):
    if use_2d:
        assert max_seqlen_q is not None
        dummy_headwise = q[0, :, 0].to(device=q.device, dtype=torch.float32).contiguous()
        dummy_col = q[:max_seqlen_q, 0, 0].to(device=q.device, dtype=torch.float32).contiguous()
        dummy_square = dummy_col[:, None].expand(max_seqlen_q, max_seqlen_q).contiguous()
        alibi_tensor = alibi_slopes.to(device=q.device, dtype=torch.float32).contiguous() if alibi_slopes is not None else dummy_headwise
        sink_tensor = sinks.to(device=q.device, dtype=torch.float32).contiguous() if sinks is not None else dummy_headwise
        qq_bias_tensor = qq_bias.to(device=q.device, dtype=torch.float32).contiguous() if qq_bias is not None else dummy_square
        return alibi_tensor, sink_tensor, qq_bias_tensor

    # 3D decode kernels receive feature tensors even when the features are
    # disabled. Reuse tiny placeholders to avoid per-token device allocation.
    dummy_headwise, dummy_scalar = _get_feature_dummy_tensors(q)
    alibi_tensor = alibi_slopes.to(device=q.device, dtype=torch.float32).contiguous() if alibi_slopes is not None else dummy_headwise
    sink_tensor = sinks.to(device=q.device, dtype=torch.float32).contiguous() if sinks is not None else dummy_headwise
    if qq_bias is not None:
        qq_bias_tensor = torch.empty((1, 1), device=q.device, dtype=torch.float32)
        qq_bias_tensor.copy_(qq_bias[:1, :1])
    else:
        qq_bias_tensor = dummy_scalar
    return alibi_tensor, sink_tensor, qq_bias_tensor


def _run_3d_native(
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    total_num_q_tokens,
    num_total_blocks,
    block_size,
    max_num_blocks_per_seq,
    softmax_scale,
    softcap,
    use_softcap,
    sliding_window,
    use_alibi,
    use_sink,
    use_qq_bias,
    alibi_tensor,
    sink_tensor,
    qq_bias_tensor,
    num_segments,
    # num_segments,
    is_int8_kv=False,
    k_descale=None,
    v_descale=None,
):
    split_kwargs, reduce_kwargs, cache_key = _get_cached_3d_plan(
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
    if num_segments == 1 and os.getenv("UA_DISABLE_3D_FUSED", "0") != "1":
        fused_kernel = _get_cached_3d_fused_kernel(
            cache_key,
            split_kwargs,
            is_int8_kv=is_int8_kv,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        fused_kernel(
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
        return
    split_kernel, reduce_kernel = _get_cached_3d_kernels(
        cache_key,
        split_kwargs,
        reduce_kwargs,
        is_int8_kv=is_int8_kv,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    block_h = split_kwargs["block_h"]
    segm_output_part, segm_max_part, segm_expsum_part = _get_cached_3d_workspace(
        device=q.device,
        num_seqs=num_seqs,
        num_kv_heads=num_kv_heads,
        num_segments=num_segments,
        block_h=block_h,
        head_size=head_size,
        total_num_q_tokens=total_num_q_tokens,
        num_query_heads=num_query_heads,
    )
    split_kernel(
        q,
        k_cache,
        v_cache,
        block_table,
        seqused_k,
        cu_seqlens_q,
        alibi_tensor,
        sink_tensor,
        qq_bias_tensor,
        segm_output_part,
        segm_max_part,
        segm_expsum_part,
    )
    reduce_kernel(
        segm_output_part,
        segm_max_part,
        segm_expsum_part,
        cu_seqlens_q,
        out,
    )


def _run_3d_native_nofeature(
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    total_num_q_tokens,
    num_total_blocks,
    block_size,
    max_num_blocks_per_seq,
    softmax_scale,
    num_segments,
    is_int8_kv=False,
    k_descale=None,
    v_descale=None,
):
    split_kwargs, reduce_kwargs, cache_key = _get_cached_3d_plan(
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
        softcap=0.0,
        use_softcap=False,
        sliding_window=-1,
        use_alibi=False,
        use_sink=False,
        use_qq_bias=False,
        num_segments=num_segments,
        is_int8_kv=is_int8_kv,
    )
    if num_segments == 1 and os.getenv("UA_DISABLE_3D_FUSED", "0") != "1":
        dummy_headwise, dummy_scalar = _get_feature_dummy_tensors(q)
        _run_3d_native(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            block_table=block_table,
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
            softcap=0.0,
            use_softcap=False,
            sliding_window=-1,
            use_alibi=False,
            use_sink=False,
            use_qq_bias=False,
            alibi_tensor=dummy_headwise,
            sink_tensor=dummy_headwise,
            qq_bias_tensor=dummy_scalar,
            num_segments=num_segments,
            is_int8_kv=is_int8_kv,
            k_descale=k_descale,
            v_descale=v_descale,
        )
        return
    split_kernel, reduce_kernel = _get_cached_3d_nofeature_kernels(
        cache_key,
        split_kwargs,
        reduce_kwargs,
        is_int8_kv=is_int8_kv,
        k_descale=k_descale,
        v_descale=v_descale,
    )
    block_h = split_kwargs["block_h"]
    segm_output_part, segm_max_part, segm_expsum_part = _get_cached_3d_workspace(
        device=q.device,
        num_seqs=num_seqs,
        num_kv_heads=num_kv_heads,
        num_segments=num_segments,
        block_h=block_h,
        head_size=head_size,
        total_num_q_tokens=total_num_q_tokens,
        num_query_heads=num_query_heads,
    )
    split_kernel(
        q,
        k_cache,
        v_cache,
        block_table,
        seqused_k,
        cu_seqlens_q,
        segm_output_part,
        segm_max_part,
        segm_expsum_part,
    )
    reduce_kernel(
        segm_output_part,
        segm_max_part,
        segm_expsum_part,
        cu_seqlens_q,
        out,
    )


def _run_2d_native_common(
    kernel_fn,
    q,
    k_cache,
    v_cache,
    out,
    cu_seqlens_q,
    seqused_k,
    block_table,
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    total_num_q_tokens,
    num_total_blocks,
    block_size,
    max_seqlen_q,
    max_num_blocks_per_seq,
    softmax_scale,
    softcap,
    use_softcap,
    sliding_window,
    use_alibi,
    use_sink,
    use_qq_bias,
    alibi_tensor,
    sink_tensor,
    qq_bias_tensor,
    is_int8_kv=False,
    k_descale=None,
    v_descale=None,
):
    block_m = 32
    block_q = block_m // num_queries_per_kv
    block_n = max(32, ((block_size + 31) // 32) * 32)
    threads = 128
    kwargs = dict(
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        num_seqs=num_seqs,
        num_total_blocks=num_total_blocks,
        block_size=block_size,
        total_num_q_tokens=total_num_q_tokens,
        max_seqlen_q=max_seqlen_q,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        softmax_scale=softmax_scale,
        softcap=float(softcap or 0.0),
        sliding_window=sliding_window,
        use_softcap=use_softcap,
        use_alibi=use_alibi,
        use_sink=use_sink,
        use_qq_bias=use_qq_bias,
        BLOCK_M=block_m,
        BLOCK_Q=block_q,
        BLOCK_N=block_n,
        threads=threads,
        num_stages=2,
        is_int8_kv=is_int8_kv,
    )
    if k_descale is not None:
        kwargs["k_descale"] = k_descale
    if v_descale is not None:
        kwargs["v_descale"] = v_descale
    kernel = _get_cached_2d_kernel(kernel_fn, kwargs)
    result = kernel(
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
    out.copy_(result)


def unified_attention(
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
    assert window_size[1] in (-1, 0), "Only causal left-window semantics are supported"
    num_seqs = int(seqused_k.numel())
    block_size = int(k_cache.shape[1])
    num_query_heads = int(q.shape[1])
    num_kv_heads = int(k_cache.shape[2])
    head_size = int(q.shape[2])
    num_queries_per_kv = num_query_heads // num_kv_heads
    assert num_query_heads % num_kv_heads == 0, "num_query_heads must be divisible by num_kv_heads"

    total_num_q_tokens = int(q.shape[0])
    max_num_blocks_per_seq = int(block_table.shape[1])
    sliding_window = -1 if window_size[0] < 0 else 1 + int(window_size[0])
    use_softcap = bool(softcap and softcap > 0)
    use_alibi = alibi_slopes is not None
    use_sink = sinks is not None
    use_qq_bias = qq_bias is not None
    block_m = 32
    block_q = block_m // num_queries_per_kv
    total_num_q_blocks = q.shape[0] // block_q + num_seqs
    use_2d = max_seqlen_q > 1 or total_num_q_blocks * num_kv_heads > 128
    is_int8_kv = k_cache.dtype == torch.int8 and v_cache.dtype == torch.int8

    num_total_blocks = int(k_cache.shape[0])
    num_segments = _default_num_segments(max_seqlen_k)
    no_feature = (not use_2d) and (not use_softcap) and (sliding_window < 0) and (not use_alibi) and (not use_sink) and (not use_qq_bias)

    if no_feature:
        _run_3d_native_nofeature(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            block_table=block_table,
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
            num_segments=num_segments,
            is_int8_kv=is_int8_kv,
            k_descale=1.0 if is_int8_kv and k_descale is None else (_scalar_from_descale(k_descale) if is_int8_kv else None),
            v_descale=1.0 if is_int8_kv and v_descale is None else (_scalar_from_descale(v_descale) if is_int8_kv else None),
        )
        return

    alibi_tensor, sink_tensor, qq_bias_tensor = _prepare_feature_tensors(
        q, alibi_slopes, sinks, qq_bias, use_2d=use_2d, max_seqlen_q=max_seqlen_q
    )

    if not use_2d:
        _run_3d_native(
            q=q,
            k_cache=k_cache,
            v_cache=v_cache,
            out=out,
            cu_seqlens_q=cu_seqlens_q,
            seqused_k=seqused_k,
            block_table=block_table,
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
            alibi_tensor=alibi_tensor,
            sink_tensor=sink_tensor,
            qq_bias_tensor=qq_bias_tensor,
            num_segments=num_segments,
            is_int8_kv=is_int8_kv,
            k_descale=1.0 if is_int8_kv and k_descale is None else (_scalar_from_descale(k_descale) if is_int8_kv else None),
            v_descale=1.0 if is_int8_kv and v_descale is None else (_scalar_from_descale(v_descale) if is_int8_kv else None),
        )
        return

    _run_2d_native_common(
        kernel_fn=tilelang_unified_attention_2d_basic,
        q=q,
        k_cache=k_cache,
        v_cache=v_cache,
        out=out,
        cu_seqlens_q=cu_seqlens_q,
        seqused_k=seqused_k,
        block_table=block_table,
        num_query_heads=num_query_heads,
        num_kv_heads=num_kv_heads,
        num_queries_per_kv=num_queries_per_kv,
        head_size=head_size,
        num_seqs=num_seqs,
        total_num_q_tokens=total_num_q_tokens,
        num_total_blocks=num_total_blocks,
        block_size=block_size,
        max_seqlen_q=max_seqlen_q,
        max_num_blocks_per_seq=max_num_blocks_per_seq,
        softmax_scale=softmax_scale,
        softcap=softcap,
        use_softcap=use_softcap,
        sliding_window=sliding_window,
        use_alibi=use_alibi,
        use_sink=use_sink,
        use_qq_bias=use_qq_bias,
        alibi_tensor=alibi_tensor,
        sink_tensor=sink_tensor,
        qq_bias_tensor=qq_bias_tensor,
        is_int8_kv=is_int8_kv,
        k_descale=1.0 if is_int8_kv and k_descale is None else (_scalar_from_descale(k_descale) if is_int8_kv else None),
        v_descale=1.0 if is_int8_kv and v_descale is None else (_scalar_from_descale(v_descale) if is_int8_kv else None),
    )


# 3D kernels
@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    },
    compile_flags=_UA_GEMV_EXTRA_MCC_FLAGS,
)
def tilelang_unified_attention_3d_split_basic(
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    total_num_q_tokens,
    num_total_blocks,
    block_size,
    max_num_blocks_per_seq,
    num_segments,
    softmax_scale,
    softcap,
    use_softcap,
    use_alibi,
    use_sink,
    use_qq_bias,
    sliding_window,
    use_window,
    block_h,
    valid_block_h,
    BLOCK_N,
    threads,
    num_stages,
    is_int8_kv=False,
    k_descale=1.0,
    v_descale=1.0,
):
    dtype = "float16"
    kv_storage_dtype = "int8" if is_int8_kv else dtype
    accum_dtype = "float"
    scale_log2e = 1.44269504
    vec = 16
    vec_groups = _cdiv(head_size, vec)
    blocks_per_kv_tile = 2 if block_size * 2 == BLOCK_N else 1
    kv_tile_tokens = blocks_per_kv_tile * block_size
    enable_full_tile_scale_fast_path = (
        (not use_softcap) and (not use_window) and (not use_alibi) and (not use_qq_bias) and (kv_tile_tokens == BLOCK_N)
    )
    q_load_rows = max(
        valid_block_h,
        min(block_h, int(os.getenv("UA_Q_LOAD_ROWS", str(valid_block_h)))),
    )
    dtype_bytes = 2
    # kv_storage_bytes = 1 if is_int8_kv else dtype_bytes
    # float32_bytes = 4

    @T.macro
    def ApplySoftcap(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        token_start: T.int32,
        seq_len: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j < seq_len),
                softcap * T.tanh(score_shared[i, j] * softmax_scale / softcap),
                -T.infinity(accum_dtype),
            )

    @T.macro
    def ApplyScale(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        token_start: T.int32,
        seq_len: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j < seq_len),
                score_shared[i, j] * softmax_scale,
                -T.infinity(accum_dtype),
            )

    @T.macro
    def ApplySlidingWindowMask(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        token_start: T.int32,
        seq_len: T.int32,
        visible_start_token: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j < seq_len) and (token_start + j < visible_start_token),
                -T.infinity(accum_dtype),
                score_shared[i, j],
            )

    @T.macro
    def ApplyAlibi(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        alibi_shared: T.SharedBuffer([block_h], "float32"),
        token_start: T.int32,
        seq_len: T.int32,
        context_len: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j < seq_len),
                score_shared[i, j] + alibi_shared[i] * (token_start + j - context_len),
                score_shared[i, j],
            )

    @T.macro
    def ApplyQQBias(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        qq_bias_shared: T.SharedBuffer([1], "float32"),
        token_start: T.int32,
        context_len: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j == context_len),
                score_shared[i, j] + qq_bias_shared[0],
                score_shared[i, j],
            )

    @T.macro
    def DecodeSignedInt8Vec(src_i8, tmp_f32, dst_f16, descale):
        for vi in T.serial(vec):
            tmp_f32[vi] = src_i8[vi]
            tmp_f32[vi] = T.if_then_else(tmp_f32[vi] > 127.0, tmp_f32[vi] - 256.0, tmp_f32[vi])
            tmp_f32[vi] = tmp_f32[vi] * descale
            dst_f16[vi] = tmp_f32[vi]

    @T.prim_func
    def main(
        query_ptr: T.Tensor([total_num_q_tokens, num_query_heads, head_size], dtype),
        key_cache_ptr: T.Tensor([num_total_blocks, block_size, num_kv_heads, head_size], kv_storage_dtype),
        value_cache_ptr: T.Tensor([num_total_blocks, block_size, num_kv_heads, head_size], kv_storage_dtype),
        block_tables_ptr: T.Tensor([num_seqs, max_num_blocks_per_seq], "int32"),
        seq_lens_ptr: T.Tensor([num_seqs], "int32"),
        cu_seqlens_q_ptr: T.Tensor([num_seqs + 1], "int32"),
        alibi_slopes_ptr: T.Tensor([num_query_heads], "float32"),
        sinks_ptr: T.Tensor([num_query_heads], "float32"),
        qq_bias_ptr: T.Tensor([1, 1], "float32"),
        segm_output_ptr: T.Tensor([num_seqs, num_kv_heads, num_segments, block_h, head_size], accum_dtype),
        segm_max_ptr: T.Tensor([num_seqs, num_kv_heads, num_segments, block_h], accum_dtype),
        segm_expsum_ptr: T.Tensor([num_seqs, num_kv_heads, num_segments, block_h], accum_dtype),
    ):
        with T.Kernel(num_seqs, num_kv_heads, num_segments, threads=threads) as (bx, by, bz):
            q_shared = T.alloc_shared([block_h, head_size], dtype)
            k_shared = T.alloc_shared([BLOCK_N, head_size], dtype)
            v_shared = T.alloc_shared([BLOCK_N, head_size], dtype)
            k_local = T.alloc_fragment([BLOCK_N, head_size], dtype)
            v_local = T.alloc_fragment([BLOCK_N, head_size], dtype)
            alibi_shared = T.alloc_shared([block_h], "float32")
            sink_shared = T.alloc_shared([block_h], "float32")
            qq_bias_shared = T.alloc_shared([1], "float32")
            acc_s = T.alloc_fragment([block_h, BLOCK_N], accum_dtype)
            score_shared = T.alloc_shared([block_h, BLOCK_N], accum_dtype)
            prob_shared = T.alloc_shared([block_h, BLOCK_N], dtype)
            acc_o = T.alloc_fragment([block_h, head_size], accum_dtype)
            scores_max = T.alloc_fragment([block_h], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_h], accum_dtype)
            scores_sum = T.alloc_fragment([block_h], accum_dtype)
            logsum = T.alloc_fragment([block_h], accum_dtype)

            T.use_swizzle(10)
            T.annotate_layout(
                {
                    # Materialize the post-softmax probabilities in shared memory so the
                    # second GEMM can read them directly, then swizzle that shared tile
                    # because it is the hottest matrix in the 3D decode path on mp22.
                    prob_shared: tilelang.layout.make_swizzled_layout(prob_shared),
                }
            )

            seq_len = seq_lens_ptr[bx]
            num_blocks = T.ceildiv(seq_len, block_size)
            blocks_per_segment = T.ceildiv(seq_len, num_segments * block_size)
            block_start = bz * blocks_per_segment
            block_end = T.min((bz + 1) * blocks_per_segment, num_blocks)
            head_start = by * num_queries_per_kv
            context_len = seq_len - 1
            q_tok_idx = cu_seqlens_q_ptr[bx]
            visible_start_token = T.max(0, context_len - sliding_window + 1)
            effective_block_start = T.max(block_start, visible_start_token // block_size)
            has_valid_block = T.alloc_local([1], "int32")
            block_has_visible_tokens = T.alloc_local([1], "int32")
            tile_token_limit = T.alloc_local([1], "int32")
            if enable_full_tile_scale_fast_path:
                full_tile_scale_fast_path = T.alloc_local([1], "int32")
                full_tile_scale_fast_path[0] = T.if_then_else(
                    (seq_len % block_size == 0)
                    and (blocks_per_segment * num_segments == num_blocks)
                    and (blocks_per_segment % blocks_per_kv_tile == 0),
                    1,
                    0,
                )
            phys_block_idx_first = T.alloc_local([1], "int32")
            phys_block_idx_second = T.alloc_local([1], "int32")
            phys_block_idx_single = T.alloc_local([1], "int32")
            query_robust_desc = T.make_robust_desc(
                T.address_of(query_ptr[0, 0, 0]),
                total_num_q_tokens * num_query_heads * head_size * dtype_bytes,
            )
            T.copy(
                query_ptr[q_tok_idx, head_start : head_start + q_load_rows, 0:head_size],
                q_shared[0:q_load_rows, 0:head_size],
                force_async_copy=True,
                src_robust_desc=query_robust_desc,
            )
            for i in T.Parallel(valid_block_h):
                alibi_shared[i] = T.if_then_else(
                    use_alibi and (head_start + i < num_query_heads),
                    alibi_slopes_ptr[head_start + i],
                    T.cast(0, "float32"),
                )
                sink_shared[i] = T.if_then_else(
                    use_sink and (head_start + i < num_query_heads),
                    sinks_ptr[head_start + i],
                    T.cast(0, "float32"),
                )
            qq_bias_shared[0] = T.if_then_else(use_qq_bias, qq_bias_ptr[0, 0], T.cast(0, "float32"))
            T.clear(acc_o)
            for i in T.Parallel(valid_block_h):
                if use_sink and bz == 0 and i < num_queries_per_kv:
                    scores_max[i] = sink_shared[i]
                    logsum[i] = 1
                else:
                    scores_max[i] = -T.infinity(accum_dtype)
                    logsum[i] = 0
            has_valid_block[0] = T.if_then_else(use_sink and (bz == 0), 1, 0)

            for block_idx in T.serial(
                T.ceildiv(
                    T.if_then_else(use_window, block_end - effective_block_start, block_end - block_start),
                    blocks_per_kv_tile,
                )
            ):
                logical_block_idx = T.if_then_else(
                    use_window,
                    effective_block_start + block_idx * blocks_per_kv_tile,
                    block_start + block_idx * blocks_per_kv_tile,
                )
                token_start = logical_block_idx * block_size
                tile_token_limit[0] = T.if_then_else(
                    blocks_per_kv_tile == 2,
                    T.if_then_else(logical_block_idx + 1 < block_end, kv_tile_tokens, block_size),
                    block_size,
                )
                block_has_visible_tokens[0] = T.if_then_else(
                    token_start < seq_len,
                    T.if_then_else((not use_window) or (token_start + tile_token_limit[0] > visible_start_token), 1, 0),
                    0,
                )
                if blocks_per_kv_tile == 2:
                    phys_block_idx_first[0] = block_tables_ptr[bx, logical_block_idx]
                    has_second_block = logical_block_idx + 1 < block_end
                    if is_int8_kv:
                        k_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                        k_reg_f32 = T.alloc_local([vec], "float32")
                        k_reg_f16 = T.alloc_local([vec], dtype)
                        if has_second_block:
                            phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                        if has_second_block:
                            for row, group in T.Parallel(BLOCK_N, vec_groups):
                                src_row = row % block_size
                                use_second = row >= block_size
                                src_block = T.if_then_else(
                                    use_second,
                                    phys_block_idx_second[0],
                                    phys_block_idx_first[0],
                                )
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_reg_i8[vi] = key_cache_ptr[src_block, src_row, by, d]
                                    else:
                                        k_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_shared[row, d] = k_reg_f16[vi]
                        else:
                            T.fill(k_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                            for row, group in T.Parallel(block_size, vec_groups):
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_reg_i8[vi] = key_cache_ptr[phys_block_idx_first[0], row, by, d]
                                    else:
                                        k_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_shared[row, d] = k_reg_f16[vi]
                        T.sync_threads()
                    else:
                        if enable_full_tile_scale_fast_path:
                            if full_tile_scale_fast_path[0] != 0:
                                phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                                T.copy(
                                    key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    k_local[0:block_size, 0:head_size],
                                )
                                T.copy(
                                    key_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                    k_local[block_size : block_size * 2, 0:head_size],
                                )
                            else:
                                if has_second_block:
                                    phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                                if has_second_block:
                                    T.copy(
                                        key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                        k_local[0:block_size, 0:head_size],
                                    )
                                    T.copy(
                                        key_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                        k_local[block_size : block_size * 2, 0:head_size],
                                    )
                                else:
                                    if block_size * 4 == BLOCK_N:
                                        T.fill(k_local, T.cast(0, dtype))
                                    else:
                                        T.fill(k_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                                    T.copy(
                                        key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                        k_local[0:block_size, 0:head_size],
                                    )
                        else:
                            if has_second_block:
                                phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                            if has_second_block:
                                T.copy(
                                    key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    k_local[0:block_size, 0:head_size],
                                )
                                T.copy(
                                    key_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                    k_local[block_size : block_size * 2, 0:head_size],
                                )
                            else:
                                if block_size * 4 == BLOCK_N:
                                    T.fill(k_local, T.cast(0, dtype))
                                else:
                                    T.fill(k_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                                T.copy(
                                    key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    k_local[0:block_size, 0:head_size],
                                )
                elif is_int8_kv:
                    k_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                    k_reg_f32 = T.alloc_local([vec], "float32")
                    k_reg_f16 = T.alloc_local([vec], dtype)
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    T.fill(k_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    for row, group in T.Parallel(block_size, vec_groups):
                        for vi in T.serial(vec):
                            d = group * vec + vi
                            if d < head_size:
                                k_reg_i8[vi] = key_cache_ptr[phys_block_idx_single[0], row, by, d]
                            else:
                                k_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                        DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                        for vi in T.vectorized(vec):
                            d = group * vec + vi
                            if d < head_size:
                                k_shared[row, d] = k_reg_f16[vi]
                    T.sync_threads()
                else:
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    if block_size * 4 == BLOCK_N:
                        T.fill(k_local, T.cast(0, dtype))
                    else:
                        T.fill(k_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    T.copy(
                        key_cache_ptr[phys_block_idx_single[0], 0:block_size, by, 0:head_size],
                        k_local[0:block_size, 0:head_size],
                    )
                T.clear(acc_s)
                if not is_int8_kv:
                    T.gemm(q_shared, k_local, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if use_softcap or use_window or use_alibi or use_qq_bias:
                    for i, j in T.Parallel(valid_block_h, BLOCK_N):
                        score_shared[i, j] = acc_s[i, j]
                    if use_softcap:
                        ApplySoftcap(score_shared, token_start, seq_len, tile_token_limit[0])
                    else:
                        ApplyScale(score_shared, token_start, seq_len, tile_token_limit[0])
                    if use_window:
                        ApplySlidingWindowMask(
                            score_shared,
                            token_start,
                            seq_len,
                            visible_start_token,
                            tile_token_limit[0],
                        )
                    if use_alibi:
                        ApplyAlibi(
                            score_shared,
                            alibi_shared,
                            token_start,
                            seq_len,
                            context_len,
                            tile_token_limit[0],
                        )
                    if use_qq_bias:
                        ApplyQQBias(
                            score_shared,
                            qq_bias_shared,
                            token_start,
                            context_len,
                            tile_token_limit[0],
                        )
                    for i, j in T.Parallel(valid_block_h, BLOCK_N):
                        acc_s[i, j] = score_shared[i, j]
                else:
                    if enable_full_tile_scale_fast_path:
                        if full_tile_scale_fast_path[0] != 0:
                            for i, j in T.Parallel(valid_block_h, BLOCK_N):
                                acc_s[i, j] *= softmax_scale
                        else:
                            for i, j in T.Parallel(valid_block_h, BLOCK_N):
                                acc_s[i, j] = T.if_then_else(
                                    (j < tile_token_limit[0]) and (token_start + j < seq_len),
                                    acc_s[i, j] * softmax_scale,
                                    -T.infinity(accum_dtype),
                                )
                    else:
                        for i, j in T.Parallel(valid_block_h, BLOCK_N):
                            acc_s[i, j] = T.if_then_else(
                                (j < tile_token_limit[0]) and (token_start + j < seq_len),
                                acc_s[i, j] * softmax_scale,
                                -T.infinity(accum_dtype),
                            )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(valid_block_h):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_max[i] = T.if_then_else(
                        scores_max[i] == -T.infinity(accum_dtype),
                        0,
                        scores_max[i],
                    )
                for i, j in T.Parallel(valid_block_h, BLOCK_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale_log2e - scores_max[i] * scale_log2e)
                T.fill(scores_sum, 0)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(valid_block_h):
                    scores_max_prev[i] = T.exp2(scores_max_prev[i] * scale_log2e - scores_max[i] * scale_log2e)
                    logsum[i] = logsum[i] * scores_max_prev[i] + scores_sum[i]

                for i, j in T.Parallel(valid_block_h, head_size):
                    acc_o[i, j] *= scores_max_prev[i]
                for i, j in T.Parallel(valid_block_h, BLOCK_N):
                    prob_shared[i, j] = T.cast(acc_s[i, j], dtype)
                if blocks_per_kv_tile == 2:
                    if is_int8_kv:
                        v_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                        v_reg_f32 = T.alloc_local([vec], "float32")
                        v_reg_f16 = T.alloc_local([vec], dtype)
                        if has_second_block:
                            for row, group in T.Parallel(BLOCK_N, vec_groups):
                                src_row = row % block_size
                                use_second = row >= block_size
                                src_block = T.if_then_else(
                                    use_second,
                                    phys_block_idx_second[0],
                                    phys_block_idx_first[0],
                                )
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_reg_i8[vi] = value_cache_ptr[src_block, src_row, by, d]
                                    else:
                                        v_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_shared[row, d] = v_reg_f16[vi]
                        else:
                            T.fill(v_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                            for row, group in T.Parallel(block_size, vec_groups):
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_reg_i8[vi] = value_cache_ptr[phys_block_idx_first[0], row, by, d]
                                    else:
                                        v_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_shared[row, d] = v_reg_f16[vi]
                        T.sync_threads()
                    else:
                        if enable_full_tile_scale_fast_path:
                            if full_tile_scale_fast_path[0] != 0:
                                T.copy(
                                    value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    v_local[0:block_size, 0:head_size],
                                )
                                T.copy(
                                    value_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                    v_local[block_size : block_size * 2, 0:head_size],
                                )
                            else:
                                if has_second_block:
                                    T.copy(
                                        value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                        v_local[0:block_size, 0:head_size],
                                    )
                                    T.copy(
                                        value_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                        v_local[block_size : block_size * 2, 0:head_size],
                                    )
                                else:
                                    if block_size * 4 == BLOCK_N:
                                        T.fill(v_local, T.cast(0, dtype))
                                    else:
                                        T.fill(v_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                                    T.copy(
                                        value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                        v_local[0:block_size, 0:head_size],
                                    )
                        else:
                            if has_second_block:
                                T.copy(
                                    value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    v_local[0:block_size, 0:head_size],
                                )
                                T.copy(
                                    value_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                    v_local[block_size : block_size * 2, 0:head_size],
                                )
                            else:
                                if block_size * 4 == BLOCK_N:
                                    T.fill(v_local, T.cast(0, dtype))
                                else:
                                    T.fill(v_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                                T.copy(
                                    value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    v_local[0:block_size, 0:head_size],
                                )
                elif is_int8_kv:
                    v_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                    v_reg_f32 = T.alloc_local([vec], "float32")
                    v_reg_f16 = T.alloc_local([vec], dtype)
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    T.fill(v_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    for row, group in T.Parallel(block_size, vec_groups):
                        for vi in T.serial(vec):
                            d = group * vec + vi
                            if d < head_size:
                                v_reg_i8[vi] = value_cache_ptr[phys_block_idx_single[0], row, by, d]
                            else:
                                v_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                        DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                        for vi in T.vectorized(vec):
                            d = group * vec + vi
                            if d < head_size:
                                v_shared[row, d] = v_reg_f16[vi]
                    T.sync_threads()
                else:
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    if block_size * 4 == BLOCK_N:
                        T.fill(v_local, T.cast(0, dtype))
                    else:
                        T.fill(v_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    T.copy(
                        value_cache_ptr[phys_block_idx_single[0], 0:block_size, by, 0:head_size],
                        v_local[0:block_size, 0:head_size],
                    )
                if not is_int8_kv:
                    T.gemm(prob_shared, v_local, acc_o, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(prob_shared, v_shared, acc_o, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)
                has_valid_block[0] = T.if_then_else(
                    block_has_visible_tokens[0] != 0,
                    1,
                    has_valid_block[0],
                )

            for i, j in T.Parallel(valid_block_h, head_size):
                segm_output_ptr[bx, by, bz, i, j] = acc_o[i, j]
            for i in T.Parallel(valid_block_h):
                segm_max_ptr[bx, by, bz, i] = T.if_then_else(has_valid_block[0] != 0, scores_max[i], -T.infinity(accum_dtype))
                segm_expsum_ptr[bx, by, bz, i] = T.if_then_else(has_valid_block[0] != 0, logsum[i], 0)

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    },
    compile_flags=_UA_GEMV_EXTRA_MCC_FLAGS,
)
def tilelang_unified_attention_3d_split_nofeature_basic(
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    total_num_q_tokens,
    num_total_blocks,
    block_size,
    max_num_blocks_per_seq,
    num_segments,
    softmax_scale,
    block_h,
    valid_block_h,
    BLOCK_N,
    threads,
    num_stages,
    is_int8_kv=False,
    k_descale=1.0,
    v_descale=1.0,
):
    dtype = "float16"
    kv_storage_dtype = "int8" if is_int8_kv else dtype
    accum_dtype = "float"
    scale_log2e = 1.44269504
    vec = 16
    vec_groups = _cdiv(head_size, vec)
    blocks_per_kv_tile = 2 if block_size * 2 == BLOCK_N else 1
    kv_tile_tokens = blocks_per_kv_tile * block_size
    enable_full_tile_scale_fast_path = kv_tile_tokens == BLOCK_N
    q_load_rows = max(
        valid_block_h,
        min(block_h, int(os.getenv("UA_Q_LOAD_ROWS", str(valid_block_h)))),
    )
    dtype_bytes = 2
    # kv_storage_bytes = 1 if is_int8_kv else dtype_bytes
    # float32_bytes = 4

    @T.macro
    def DecodeSignedInt8Vec(src_i8, tmp_f32, dst_f16, descale):
        for vi in T.serial(vec):
            tmp_f32[vi] = src_i8[vi]
            tmp_f32[vi] = T.if_then_else(tmp_f32[vi] > 127.0, tmp_f32[vi] - 256.0, tmp_f32[vi])
            tmp_f32[vi] = tmp_f32[vi] * descale
            dst_f16[vi] = tmp_f32[vi]

    @T.prim_func
    def main(
        query_ptr: T.Tensor([total_num_q_tokens, num_query_heads, head_size], dtype),
        key_cache_ptr: T.Tensor([num_total_blocks, block_size, num_kv_heads, head_size], kv_storage_dtype),
        value_cache_ptr: T.Tensor([num_total_blocks, block_size, num_kv_heads, head_size], kv_storage_dtype),
        block_tables_ptr: T.Tensor([num_seqs, max_num_blocks_per_seq], "int32"),
        seq_lens_ptr: T.Tensor([num_seqs], "int32"),
        cu_seqlens_q_ptr: T.Tensor([num_seqs + 1], "int32"),
        segm_output_ptr: T.Tensor([num_seqs, num_kv_heads, num_segments, block_h, head_size], accum_dtype),
        segm_max_ptr: T.Tensor([num_seqs, num_kv_heads, num_segments, block_h], accum_dtype),
        segm_expsum_ptr: T.Tensor([num_seqs, num_kv_heads, num_segments, block_h], accum_dtype),
    ):
        with T.Kernel(num_seqs, num_kv_heads, num_segments, threads=threads) as (bx, by, bz):
            q_shared = T.alloc_shared([block_h, head_size], dtype)
            k_shared = T.alloc_shared([BLOCK_N, head_size], dtype)
            v_shared = T.alloc_shared([BLOCK_N, head_size], dtype)
            k_local = T.alloc_fragment([BLOCK_N, head_size], dtype)
            v_local = T.alloc_fragment([BLOCK_N, head_size], dtype)
            acc_s = T.alloc_fragment([block_h, BLOCK_N], accum_dtype)
            # score_shared = T.alloc_shared([block_h, BLOCK_N], accum_dtype)
            prob_shared = T.alloc_shared([block_h, BLOCK_N], dtype)
            acc_o = T.alloc_fragment([block_h, head_size], accum_dtype)
            scores_max = T.alloc_fragment([block_h], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_h], accum_dtype)
            scores_sum = T.alloc_fragment([block_h], accum_dtype)
            logsum = T.alloc_fragment([block_h], accum_dtype)

            T.use_swizzle(10)
            T.annotate_layout(
                {
                    # Materialize the post-softmax probabilities in shared memory so the
                    # second GEMM can read them directly, then swizzle that shared tile
                    # because it is the hottest matrix in the 3D decode path on mp22.
                    prob_shared: tilelang.layout.make_swizzled_layout(prob_shared),
                }
            )

            seq_len = seq_lens_ptr[bx]
            num_blocks = T.ceildiv(seq_len, block_size)
            blocks_per_segment = T.ceildiv(seq_len, num_segments * block_size)
            block_start = bz * blocks_per_segment
            block_end = T.min((bz + 1) * blocks_per_segment, num_blocks)
            head_start = by * num_queries_per_kv
            q_tok_idx = cu_seqlens_q_ptr[bx]
            has_valid_block = T.alloc_local([1], "int32")
            block_has_visible_tokens = T.alloc_local([1], "int32")
            tile_token_limit = T.alloc_local([1], "int32")
            if enable_full_tile_scale_fast_path:
                full_tile_scale_fast_path = T.alloc_local([1], "int32")
                full_tile_scale_fast_path[0] = T.if_then_else(
                    (seq_len % block_size == 0)
                    and (blocks_per_segment * num_segments == num_blocks)
                    and (blocks_per_segment % blocks_per_kv_tile == 0),
                    1,
                    0,
                )
            phys_block_idx_first = T.alloc_local([1], "int32")
            phys_block_idx_second = T.alloc_local([1], "int32")
            phys_block_idx_single = T.alloc_local([1], "int32")
            query_robust_desc = T.make_robust_desc(
                T.address_of(query_ptr[0, 0, 0]),
                total_num_q_tokens * num_query_heads * head_size * dtype_bytes,
            )
            T.copy(
                query_ptr[q_tok_idx, head_start : head_start + q_load_rows, 0:head_size],
                q_shared[0:q_load_rows, 0:head_size],
                force_async_copy=True,
                src_robust_desc=query_robust_desc,
            )
            T.clear(acc_o)
            for i in T.Parallel(valid_block_h):
                scores_max[i] = -T.infinity(accum_dtype)
                logsum[i] = 0
            has_valid_block[0] = 0

            for block_idx in T.serial(T.ceildiv(block_end - block_start, blocks_per_kv_tile)):
                logical_block_idx = block_start + block_idx * blocks_per_kv_tile
                token_start = logical_block_idx * block_size
                tile_token_limit[0] = T.if_then_else(
                    blocks_per_kv_tile == 2,
                    T.if_then_else(logical_block_idx + 1 < block_end, kv_tile_tokens, block_size),
                    block_size,
                )
                block_has_visible_tokens[0] = T.if_then_else(token_start < seq_len, 1, 0)
                if blocks_per_kv_tile == 2:
                    phys_block_idx_first[0] = block_tables_ptr[bx, logical_block_idx]
                    has_second_block = logical_block_idx + 1 < block_end
                    if is_int8_kv:
                        k_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                        k_reg_f32 = T.alloc_local([vec], "float32")
                        k_reg_f16 = T.alloc_local([vec], dtype)
                        if has_second_block:
                            phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                        if has_second_block:
                            for row, group in T.Parallel(BLOCK_N, vec_groups):
                                src_row = row % block_size
                                use_second = row >= block_size
                                src_block = T.if_then_else(
                                    use_second,
                                    phys_block_idx_second[0],
                                    phys_block_idx_first[0],
                                )
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_reg_i8[vi] = key_cache_ptr[src_block, src_row, by, d]
                                    else:
                                        k_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_shared[row, d] = k_reg_f16[vi]
                        else:
                            T.fill(k_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                            for row, group in T.Parallel(block_size, vec_groups):
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_reg_i8[vi] = key_cache_ptr[phys_block_idx_first[0], row, by, d]
                                    else:
                                        k_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_shared[row, d] = k_reg_f16[vi]
                        T.sync_threads()
                    else:
                        if enable_full_tile_scale_fast_path:
                            if full_tile_scale_fast_path[0] != 0:
                                phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                                T.copy(
                                    key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    k_local[0:block_size, 0:head_size],
                                )
                                T.copy(
                                    key_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                    k_local[block_size : block_size * 2, 0:head_size],
                                )
                            else:
                                if has_second_block:
                                    phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                                if has_second_block:
                                    T.copy(
                                        key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                        k_local[0:block_size, 0:head_size],
                                    )
                                    T.copy(
                                        key_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                        k_local[block_size : block_size * 2, 0:head_size],
                                    )
                                else:
                                    if block_size * 4 == BLOCK_N:
                                        T.fill(k_local, T.cast(0, dtype))
                                    else:
                                        T.fill(k_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                                    T.copy(
                                        key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                        k_local[0:block_size, 0:head_size],
                                    )
                        else:
                            if has_second_block:
                                phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                            if has_second_block:
                                T.copy(
                                    key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    k_local[0:block_size, 0:head_size],
                                )
                                T.copy(
                                    key_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                    k_local[block_size : block_size * 2, 0:head_size],
                                )
                            else:
                                if block_size * 4 == BLOCK_N:
                                    T.fill(k_local, T.cast(0, dtype))
                                else:
                                    T.fill(k_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                                T.copy(
                                    key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    k_local[0:block_size, 0:head_size],
                                )
                elif is_int8_kv:
                    k_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                    k_reg_f32 = T.alloc_local([vec], "float32")
                    k_reg_f16 = T.alloc_local([vec], dtype)
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    T.fill(k_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    for row, group in T.Parallel(block_size, vec_groups):
                        for vi in T.serial(vec):
                            d = group * vec + vi
                            if d < head_size:
                                k_reg_i8[vi] = key_cache_ptr[phys_block_idx_single[0], row, by, d]
                            else:
                                k_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                        DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                        for vi in T.vectorized(vec):
                            d = group * vec + vi
                            if d < head_size:
                                k_shared[row, d] = k_reg_f16[vi]
                    T.sync_threads()
                else:
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    if block_size * 4 == BLOCK_N:
                        T.fill(k_local, T.cast(0, dtype))
                    else:
                        T.fill(k_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    T.copy(
                        key_cache_ptr[phys_block_idx_single[0], 0:block_size, by, 0:head_size],
                        k_local[0:block_size, 0:head_size],
                    )
                T.clear(acc_s)
                if not is_int8_kv:
                    T.gemm(q_shared, k_local, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if enable_full_tile_scale_fast_path:
                    if full_tile_scale_fast_path[0] != 0:
                        for i, j in T.Parallel(valid_block_h, BLOCK_N):
                            acc_s[i, j] *= softmax_scale
                    else:
                        for i, j in T.Parallel(valid_block_h, BLOCK_N):
                            acc_s[i, j] = T.if_then_else(
                                (j < tile_token_limit[0]) and (token_start + j < seq_len),
                                acc_s[i, j] * softmax_scale,
                                -T.infinity(accum_dtype),
                            )
                else:
                    for i, j in T.Parallel(valid_block_h, BLOCK_N):
                        acc_s[i, j] = T.if_then_else(
                            (j < tile_token_limit[0]) and (token_start + j < seq_len),
                            acc_s[i, j] * softmax_scale,
                            -T.infinity(accum_dtype),
                        )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(valid_block_h):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_max[i] = T.if_then_else(
                        scores_max[i] == -T.infinity(accum_dtype),
                        0,
                        scores_max[i],
                    )
                for i, j in T.Parallel(valid_block_h, BLOCK_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale_log2e - scores_max[i] * scale_log2e)
                T.fill(scores_sum, 0)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(valid_block_h):
                    scores_max_prev[i] = T.exp2(scores_max_prev[i] * scale_log2e - scores_max[i] * scale_log2e)
                    logsum[i] = logsum[i] * scores_max_prev[i] + scores_sum[i]

                for i, j in T.Parallel(valid_block_h, head_size):
                    acc_o[i, j] *= scores_max_prev[i]
                for i, j in T.Parallel(valid_block_h, BLOCK_N):
                    prob_shared[i, j] = T.cast(acc_s[i, j], dtype)
                if blocks_per_kv_tile == 2:
                    if is_int8_kv:
                        v_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                        v_reg_f32 = T.alloc_local([vec], "float32")
                        v_reg_f16 = T.alloc_local([vec], dtype)
                        if has_second_block:
                            for row, group in T.Parallel(BLOCK_N, vec_groups):
                                src_row = row % block_size
                                use_second = row >= block_size
                                src_block = T.if_then_else(
                                    use_second,
                                    phys_block_idx_second[0],
                                    phys_block_idx_first[0],
                                )
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_reg_i8[vi] = value_cache_ptr[src_block, src_row, by, d]
                                    else:
                                        v_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_shared[row, d] = v_reg_f16[vi]
                        else:
                            T.fill(v_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                            for row, group in T.Parallel(block_size, vec_groups):
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_reg_i8[vi] = value_cache_ptr[phys_block_idx_first[0], row, by, d]
                                    else:
                                        v_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_shared[row, d] = v_reg_f16[vi]
                        T.sync_threads()
                    else:
                        if enable_full_tile_scale_fast_path:
                            if full_tile_scale_fast_path[0] != 0:
                                T.copy(
                                    value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    v_local[0:block_size, 0:head_size],
                                )
                                T.copy(
                                    value_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                    v_local[block_size : block_size * 2, 0:head_size],
                                )
                            else:
                                if has_second_block:
                                    T.copy(
                                        value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                        v_local[0:block_size, 0:head_size],
                                    )
                                    T.copy(
                                        value_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                        v_local[block_size : block_size * 2, 0:head_size],
                                    )
                                else:
                                    if block_size * 4 == BLOCK_N:
                                        T.fill(v_local, T.cast(0, dtype))
                                    else:
                                        T.fill(v_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                                    T.copy(
                                        value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                        v_local[0:block_size, 0:head_size],
                                    )
                        else:
                            if has_second_block:
                                T.copy(
                                    value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    v_local[0:block_size, 0:head_size],
                                )
                                T.copy(
                                    value_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                    v_local[block_size : block_size * 2, 0:head_size],
                                )
                            else:
                                if block_size * 4 == BLOCK_N:
                                    T.fill(v_local, T.cast(0, dtype))
                                else:
                                    T.fill(v_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                                T.copy(
                                    value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                    v_local[0:block_size, 0:head_size],
                                )
                elif is_int8_kv:
                    v_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                    v_reg_f32 = T.alloc_local([vec], "float32")
                    v_reg_f16 = T.alloc_local([vec], dtype)
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    T.fill(v_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    for row, group in T.Parallel(block_size, vec_groups):
                        for vi in T.serial(vec):
                            d = group * vec + vi
                            if d < head_size:
                                v_reg_i8[vi] = value_cache_ptr[phys_block_idx_single[0], row, by, d]
                            else:
                                v_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                        DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                        for vi in T.vectorized(vec):
                            d = group * vec + vi
                            if d < head_size:
                                v_shared[row, d] = v_reg_f16[vi]
                    T.sync_threads()
                else:
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    if block_size * 4 == BLOCK_N:
                        T.fill(v_local, T.cast(0, dtype))
                    else:
                        T.fill(v_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    T.copy(
                        value_cache_ptr[phys_block_idx_single[0], 0:block_size, by, 0:head_size],
                        v_local[0:block_size, 0:head_size],
                    )
                if not is_int8_kv:
                    T.gemm(prob_shared, v_local, acc_o, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(prob_shared, v_shared, acc_o, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)
                has_valid_block[0] = T.if_then_else(
                    block_has_visible_tokens[0] != 0,
                    1,
                    has_valid_block[0],
                )

            for i, j in T.Parallel(valid_block_h, head_size):
                segm_output_ptr[bx, by, bz, i, j] = acc_o[i, j]
            for i in T.Parallel(valid_block_h):
                segm_max_ptr[bx, by, bz, i] = T.if_then_else(has_valid_block[0] != 0, scores_max[i], -T.infinity(accum_dtype))
                segm_expsum_ptr[bx, by, bz, i] = T.if_then_else(has_valid_block[0] != 0, logsum[i], 0)

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    },
    compile_flags=_UA_GEMV_EXTRA_MCC_FLAGS,
    verbose=False,
)
def tilelang_unified_attention_3d_fused_basic(
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    total_num_q_tokens,
    num_total_blocks,
    block_size,
    max_num_blocks_per_seq,
    num_segments,
    softmax_scale,
    softcap,
    use_softcap,
    use_alibi,
    use_sink,
    use_qq_bias,
    sliding_window,
    use_window,
    block_h,
    valid_block_h,
    BLOCK_N,
    threads,
    num_stages,
    is_int8_kv=False,
    k_descale=1.0,
    v_descale=1.0,
):
    dtype = "float16"
    kv_storage_dtype = "int8" if is_int8_kv else dtype
    accum_dtype = "float"
    scale_log2e = 1.44269504
    vec = 16
    vec_groups = _cdiv(head_size, vec)
    blocks_per_kv_tile = 2 if block_size * 2 == BLOCK_N else 1
    kv_tile_tokens = blocks_per_kv_tile * block_size
    q_load_rows = max(
        valid_block_h,
        min(block_h, int(os.getenv("UA_Q_LOAD_ROWS", str(valid_block_h)))),
    )
    dtype_bytes = 2
    # kv_storage_bytes = 1 if is_int8_kv else dtype_bytes

    @T.macro
    def ApplySoftcap(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        token_start: T.int32,
        seq_len: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j < seq_len),
                softcap * T.tanh(score_shared[i, j] * softmax_scale / softcap),
                -T.infinity(accum_dtype),
            )

    @T.macro
    def ApplyScale(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        token_start: T.int32,
        seq_len: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j < seq_len),
                score_shared[i, j] * softmax_scale,
                -T.infinity(accum_dtype),
            )

    @T.macro
    def ApplySlidingWindowMask(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        token_start: T.int32,
        seq_len: T.int32,
        visible_start_token: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j < seq_len) and (token_start + j < visible_start_token),
                -T.infinity(accum_dtype),
                score_shared[i, j],
            )

    @T.macro
    def ApplyAlibi(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        alibi_shared: T.SharedBuffer([block_h], "float32"),
        token_start: T.int32,
        seq_len: T.int32,
        context_len: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j < seq_len),
                score_shared[i, j] + alibi_shared[i] * (token_start + j - context_len),
                score_shared[i, j],
            )

    @T.macro
    def ApplyQQBias(
        score_shared: T.SharedBuffer([block_h, BLOCK_N], accum_dtype),
        qq_bias_shared: T.SharedBuffer([1], "float32"),
        token_start: T.int32,
        context_len: T.int32,
        tile_token_limit: T.int32,
    ):
        for i, j in T.Parallel(valid_block_h, BLOCK_N):
            score_shared[i, j] = T.if_then_else(
                (j < tile_token_limit) and (token_start + j == context_len),
                score_shared[i, j] + qq_bias_shared[0],
                score_shared[i, j],
            )

    @T.macro
    def DecodeSignedInt8Vec(src_i8, tmp_f32, dst_f16, descale):
        for vi in T.serial(vec):
            tmp_f32[vi] = src_i8[vi]
            tmp_f32[vi] = T.if_then_else(tmp_f32[vi] > 127.0, tmp_f32[vi] - 256.0, tmp_f32[vi])
            tmp_f32[vi] = tmp_f32[vi] * descale
            dst_f16[vi] = tmp_f32[vi]

    @T.prim_func
    def main(
        query_ptr: T.Tensor([total_num_q_tokens, num_query_heads, head_size], dtype),
        key_cache_ptr: T.Tensor([num_total_blocks, block_size, num_kv_heads, head_size], kv_storage_dtype),
        value_cache_ptr: T.Tensor([num_total_blocks, block_size, num_kv_heads, head_size], kv_storage_dtype),
        block_tables_ptr: T.Tensor([num_seqs, max_num_blocks_per_seq], "int32"),
        seq_lens_ptr: T.Tensor([num_seqs], "int32"),
        cu_seqlens_q_ptr: T.Tensor([num_seqs + 1], "int32"),
        alibi_slopes_ptr: T.Tensor([num_query_heads], "float32"),
        sinks_ptr: T.Tensor([num_query_heads], "float32"),
        qq_bias_ptr: T.Tensor([1, 1], "float32"),
        output_ptr: T.Tensor([total_num_q_tokens, num_query_heads, head_size], dtype),
    ):
        with T.Kernel(num_seqs, num_kv_heads, 1, threads=threads) as (bx, by, _):
            q_shared = T.alloc_shared([block_h, head_size], dtype)
            k_shared = T.alloc_shared([BLOCK_N, head_size], dtype)
            v_shared = T.alloc_shared([BLOCK_N, head_size], dtype)
            k_local = T.alloc_fragment([BLOCK_N, head_size], dtype)
            v_local = T.alloc_fragment([BLOCK_N, head_size], dtype)
            alibi_shared = T.alloc_shared([block_h], "float32")
            sink_shared = T.alloc_shared([block_h], "float32")
            qq_bias_shared = T.alloc_shared([1], "float32")
            acc_s = T.alloc_fragment([block_h, BLOCK_N], accum_dtype)
            score_shared = T.alloc_shared([block_h, BLOCK_N], accum_dtype)
            prob_shared = T.alloc_shared([block_h, BLOCK_N], dtype)
            acc_o = T.alloc_fragment([block_h, head_size], accum_dtype)
            scores_max = T.alloc_fragment([block_h], accum_dtype)
            scores_max_prev = T.alloc_fragment([block_h], accum_dtype)
            scores_sum = T.alloc_fragment([block_h], accum_dtype)
            logsum = T.alloc_fragment([block_h], accum_dtype)

            T.use_swizzle(10)
            T.annotate_layout(
                {
                    prob_shared: tilelang.layout.make_swizzled_layout(prob_shared),
                }
            )

            seq_len = seq_lens_ptr[bx]
            num_blocks = T.ceildiv(seq_len, block_size)
            block_start = 0
            block_end = num_blocks
            head_start = by * num_queries_per_kv
            context_len = seq_len - 1
            q_tok_idx = cu_seqlens_q_ptr[bx]
            visible_start_token = T.max(0, context_len - sliding_window + 1)
            effective_block_start = T.max(block_start, visible_start_token // block_size)
            has_valid_block = T.alloc_local([1], "int32")
            block_has_visible_tokens = T.alloc_local([1], "int32")
            tile_token_limit = T.alloc_local([1], "int32")
            phys_block_idx_first = T.alloc_local([1], "int32")
            phys_block_idx_second = T.alloc_local([1], "int32")
            phys_block_idx_single = T.alloc_local([1], "int32")
            query_robust_desc = T.make_robust_desc(
                T.address_of(query_ptr[0, 0, 0]),
                total_num_q_tokens * num_query_heads * head_size * dtype_bytes,
            )
            T.copy(
                query_ptr[q_tok_idx, head_start : head_start + q_load_rows, 0:head_size],
                q_shared[0:q_load_rows, 0:head_size],
                force_async_copy=True,
                src_robust_desc=query_robust_desc,
            )
            for i in T.Parallel(valid_block_h):
                alibi_shared[i] = T.if_then_else(
                    use_alibi and (head_start + i < num_query_heads),
                    alibi_slopes_ptr[head_start + i],
                    T.cast(0, "float32"),
                )
                sink_shared[i] = T.if_then_else(
                    use_sink and (head_start + i < num_query_heads),
                    sinks_ptr[head_start + i],
                    T.cast(0, "float32"),
                )
            qq_bias_shared[0] = T.if_then_else(use_qq_bias, qq_bias_ptr[0, 0], T.cast(0, "float32"))
            T.clear(acc_o)
            for i in T.Parallel(valid_block_h):
                if use_sink and i < num_queries_per_kv:
                    scores_max[i] = sink_shared[i]
                    logsum[i] = 1
                else:
                    scores_max[i] = -T.infinity(accum_dtype)
                    logsum[i] = 0
            has_valid_block[0] = T.if_then_else(use_sink, 1, 0)

            for block_idx in T.serial(
                T.ceildiv(
                    T.if_then_else(use_window, block_end - effective_block_start, block_end - block_start),
                    blocks_per_kv_tile,
                )
            ):
                logical_block_idx = T.if_then_else(
                    use_window,
                    effective_block_start + block_idx * blocks_per_kv_tile,
                    block_start + block_idx * blocks_per_kv_tile,
                )
                token_start = logical_block_idx * block_size
                tile_token_limit[0] = T.if_then_else(
                    blocks_per_kv_tile == 2,
                    T.if_then_else(logical_block_idx + 1 < block_end, kv_tile_tokens, block_size),
                    block_size,
                )
                block_has_visible_tokens[0] = T.if_then_else(
                    token_start < seq_len,
                    T.if_then_else((not use_window) or (token_start + tile_token_limit[0] > visible_start_token), 1, 0),
                    0,
                )
                if blocks_per_kv_tile == 2:
                    phys_block_idx_first[0] = block_tables_ptr[bx, logical_block_idx]
                    has_second_block = logical_block_idx + 1 < block_end
                    if is_int8_kv:
                        k_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                        k_reg_f32 = T.alloc_local([vec], "float32")
                        k_reg_f16 = T.alloc_local([vec], dtype)
                        if has_second_block:
                            phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                        if has_second_block:
                            for row, group in T.Parallel(BLOCK_N, vec_groups):
                                src_row = row % block_size
                                use_second = row >= block_size
                                src_block = T.if_then_else(use_second, phys_block_idx_second[0], phys_block_idx_first[0])
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_reg_i8[vi] = key_cache_ptr[src_block, src_row, by, d]
                                    else:
                                        k_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_shared[row, d] = k_reg_f16[vi]
                        else:
                            T.fill(k_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                            for row, group in T.Parallel(block_size, vec_groups):
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_reg_i8[vi] = key_cache_ptr[phys_block_idx_first[0], row, by, d]
                                    else:
                                        k_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        k_shared[row, d] = k_reg_f16[vi]
                        T.sync_threads()
                    else:
                        if has_second_block:
                            phys_block_idx_second[0] = block_tables_ptr[bx, logical_block_idx + 1]
                        if has_second_block:
                            T.copy(
                                key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size], k_local[0:block_size, 0:head_size]
                            )
                            T.copy(
                                key_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                k_local[block_size : block_size * 2, 0:head_size],
                            )
                        else:
                            if block_size * 4 == BLOCK_N:
                                T.fill(k_local, T.cast(0, dtype))
                            else:
                                T.fill(k_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                            T.copy(
                                key_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size], k_local[0:block_size, 0:head_size]
                            )
                elif is_int8_kv:
                    k_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                    k_reg_f32 = T.alloc_local([vec], "float32")
                    k_reg_f16 = T.alloc_local([vec], dtype)
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    T.fill(k_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    for row, group in T.Parallel(block_size, vec_groups):
                        for vi in T.serial(vec):
                            d = group * vec + vi
                            if d < head_size:
                                k_reg_i8[vi] = key_cache_ptr[phys_block_idx_single[0], row, by, d]
                            else:
                                k_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                        DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                        for vi in T.vectorized(vec):
                            d = group * vec + vi
                            if d < head_size:
                                k_shared[row, d] = k_reg_f16[vi]
                    T.sync_threads()
                else:
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    if block_size * 4 == BLOCK_N:
                        T.fill(k_local, T.cast(0, dtype))
                    else:
                        T.fill(k_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    T.copy(key_cache_ptr[phys_block_idx_single[0], 0:block_size, by, 0:head_size], k_local[0:block_size, 0:head_size])
                T.clear(acc_s)
                if not is_int8_kv:
                    T.gemm(q_shared, k_local, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if use_softcap or use_window or use_alibi or use_qq_bias:
                    for i, j in T.Parallel(valid_block_h, BLOCK_N):
                        score_shared[i, j] = acc_s[i, j]
                    if use_softcap:
                        ApplySoftcap(score_shared, token_start, seq_len, tile_token_limit[0])
                    else:
                        ApplyScale(score_shared, token_start, seq_len, tile_token_limit[0])
                    if use_window:
                        ApplySlidingWindowMask(score_shared, token_start, seq_len, visible_start_token, tile_token_limit[0])
                    if use_alibi:
                        ApplyAlibi(score_shared, alibi_shared, token_start, seq_len, context_len, tile_token_limit[0])
                    if use_qq_bias:
                        ApplyQQBias(score_shared, qq_bias_shared, token_start, context_len, tile_token_limit[0])
                    for i, j in T.Parallel(valid_block_h, BLOCK_N):
                        acc_s[i, j] = score_shared[i, j]
                else:
                    for i, j in T.Parallel(valid_block_h, BLOCK_N):
                        acc_s[i, j] = T.if_then_else(
                            (j < tile_token_limit[0]) and (token_start + j < seq_len),
                            acc_s[i, j] * softmax_scale,
                            -T.infinity(accum_dtype),
                        )

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity(accum_dtype))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(valid_block_h):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_max[i] = T.if_then_else(scores_max[i] == -T.infinity(accum_dtype), 0, scores_max[i])
                for i, j in T.Parallel(valid_block_h, BLOCK_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale_log2e - scores_max[i] * scale_log2e)
                T.fill(scores_sum, 0)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(valid_block_h):
                    scores_max_prev[i] = T.exp2(scores_max_prev[i] * scale_log2e - scores_max[i] * scale_log2e)
                    logsum[i] = logsum[i] * scores_max_prev[i] + scores_sum[i]

                for i, j in T.Parallel(valid_block_h, head_size):
                    acc_o[i, j] *= scores_max_prev[i]
                for i, j in T.Parallel(valid_block_h, BLOCK_N):
                    prob_shared[i, j] = T.cast(acc_s[i, j], dtype)
                if blocks_per_kv_tile == 2:
                    if is_int8_kv:
                        v_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                        v_reg_f32 = T.alloc_local([vec], "float32")
                        v_reg_f16 = T.alloc_local([vec], dtype)
                        if has_second_block:
                            for row, group in T.Parallel(BLOCK_N, vec_groups):
                                src_row = row % block_size
                                use_second = row >= block_size
                                src_block = T.if_then_else(use_second, phys_block_idx_second[0], phys_block_idx_first[0])
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_reg_i8[vi] = value_cache_ptr[src_block, src_row, by, d]
                                    else:
                                        v_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_shared[row, d] = v_reg_f16[vi]
                        else:
                            T.fill(v_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                            for row, group in T.Parallel(block_size, vec_groups):
                                for vi in T.serial(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_reg_i8[vi] = value_cache_ptr[phys_block_idx_first[0], row, by, d]
                                    else:
                                        v_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                                DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                                for vi in T.vectorized(vec):
                                    d = group * vec + vi
                                    if d < head_size:
                                        v_shared[row, d] = v_reg_f16[vi]
                        T.sync_threads()
                    else:
                        if has_second_block:
                            T.copy(
                                value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                v_local[0:block_size, 0:head_size],
                            )
                            T.copy(
                                value_cache_ptr[phys_block_idx_second[0], 0:block_size, by, 0:head_size],
                                v_local[block_size : block_size * 2, 0:head_size],
                            )
                        else:
                            if block_size * 4 == BLOCK_N:
                                T.fill(v_local, T.cast(0, dtype))
                            else:
                                T.fill(v_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                            T.copy(
                                value_cache_ptr[phys_block_idx_first[0], 0:block_size, by, 0:head_size],
                                v_local[0:block_size, 0:head_size],
                            )
                elif is_int8_kv:
                    v_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
                    v_reg_f32 = T.alloc_local([vec], "float32")
                    v_reg_f16 = T.alloc_local([vec], dtype)
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    T.fill(v_shared[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    for row, group in T.Parallel(block_size, vec_groups):
                        for vi in T.serial(vec):
                            d = group * vec + vi
                            if d < head_size:
                                v_reg_i8[vi] = value_cache_ptr[phys_block_idx_single[0], row, by, d]
                            else:
                                v_reg_i8[vi] = T.cast(0, kv_storage_dtype)
                        DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                        for vi in T.vectorized(vec):
                            d = group * vec + vi
                            if d < head_size:
                                v_shared[row, d] = v_reg_f16[vi]
                    T.sync_threads()
                else:
                    phys_block_idx_single[0] = block_tables_ptr[bx, logical_block_idx]
                    if block_size * 4 == BLOCK_N:
                        T.fill(v_local, T.cast(0, dtype))
                    else:
                        T.fill(v_local[block_size:BLOCK_N, 0:head_size], T.cast(0, dtype))
                    T.copy(
                        value_cache_ptr[phys_block_idx_single[0], 0:block_size, by, 0:head_size],
                        v_local[0:block_size, 0:head_size],
                    )
                if not is_int8_kv:
                    T.gemm(prob_shared, v_local, acc_o, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)
                else:
                    T.gemm(prob_shared, v_shared, acc_o, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)
                has_valid_block[0] = T.if_then_else(block_has_visible_tokens[0] != 0, 1, has_valid_block[0])

            for i, j in T.Parallel(valid_block_h, head_size):
                output_ptr[q_tok_idx, head_start + i, j] = T.cast(
                    T.if_then_else(logsum[i] == 0, 0, acc_o[i, j] / logsum[i]),
                    dtype,
                )

    return main


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    },
    compile_flags=_UA_GEMV_EXTRA_MCC_FLAGS,
)
def tilelang_unified_attention_3d_reduce_basic(
    total_num_q_tokens,
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    num_segments,
    block_h,
    valid_block_h,
    threads,
    out_dtype="float16",
):
    accum_dtype = "float"
    scale_log2e = 1.44269504

    @T.prim_func
    def main(
        segm_output_ptr: T.Tensor([num_seqs, num_kv_heads, num_segments, block_h, head_size], accum_dtype),
        segm_max_ptr: T.Tensor([num_seqs, num_kv_heads, num_segments, block_h], accum_dtype),
        segm_expsum_ptr: T.Tensor([num_seqs, num_kv_heads, num_segments, block_h], accum_dtype),
        cu_seqlens_q_ptr: T.Tensor([num_seqs + 1], "int32"),
        output_ptr: T.Tensor([total_num_q_tokens, num_query_heads, head_size], out_dtype),
    ):
        with T.Kernel(num_seqs, num_kv_heads, 1, threads=threads) as (bx, by, _):
            tid = T.get_thread_binding()
            warp_size = 128
            subgroup_size = 32
            num_warps = threads // warp_size
            warp_idx = tid // warp_size
            lane_idx = tid % warp_size
            subgroup_idx = lane_idx // subgroup_size
            lane32 = lane_idx % subgroup_size
            row_max_shared = T.alloc_shared([block_h], accum_dtype)
            row_sum_shared = T.alloc_shared([block_h], accum_dtype)
            partial_max_shared = T.alloc_shared([block_h, 4], accum_dtype)
            partial_sum_shared = T.alloc_shared([block_h, 4], accum_dtype)
            head_start = by * num_queries_per_kv
            q_tok_idx = cu_seqlens_q_ptr[bx]

            for row in T.serial(block_h):
                if row < valid_block_h and row % num_warps == warp_idx:
                    max_local = T.alloc_local([1], accum_dtype)
                    seg_idx = subgroup_idx * subgroup_size + lane32
                    if seg_idx < num_segments:
                        max_local[0] = segm_max_ptr[bx, by, seg_idx, row]
                    else:
                        max_local[0] = -T.infinity(accum_dtype)
                    max_local[0] = T.max(max_local[0], T.shfl_xor(max_local[0], 16))
                    max_local[0] = T.max(max_local[0], T.shfl_xor(max_local[0], 8))
                    max_local[0] = T.max(max_local[0], T.shfl_xor(max_local[0], 4))
                    max_local[0] = T.max(max_local[0], T.shfl_xor(max_local[0], 2))
                    max_local[0] = T.max(max_local[0], T.shfl_xor(max_local[0], 1))
                    if lane32 == 0:
                        partial_max_shared[row, subgroup_idx] = max_local[0]

            T.sync_threads()

            for row in T.serial(block_h):
                if row < valid_block_h and row % num_warps == warp_idx and subgroup_idx == 0 and lane32 == 0:
                    row_max_shared[row] = partial_max_shared[row, 0]
                    for g in T.serial(1, 4):
                        row_max_shared[row] = T.max(row_max_shared[row], partial_max_shared[row, g])

            T.sync_threads()

            for row in T.serial(block_h):
                if row < valid_block_h and row % num_warps == warp_idx:
                    sum_local = T.alloc_local([1], accum_dtype)
                    seg_idx = subgroup_idx * subgroup_size + lane32
                    if seg_idx < num_segments:
                        sum_local[0] = segm_expsum_ptr[bx, by, seg_idx, row] * T.exp2(
                            (segm_max_ptr[bx, by, seg_idx, row] - row_max_shared[row]) * scale_log2e
                        )
                    else:
                        sum_local[0] = 0
                    sum_local[0] += T.shfl_xor(sum_local[0], 16)
                    sum_local[0] += T.shfl_xor(sum_local[0], 8)
                    sum_local[0] += T.shfl_xor(sum_local[0], 4)
                    sum_local[0] += T.shfl_xor(sum_local[0], 2)
                    sum_local[0] += T.shfl_xor(sum_local[0], 1)
                    if lane32 == 0:
                        partial_sum_shared[row, subgroup_idx] = sum_local[0]

            T.sync_threads()

            for row in T.serial(block_h):
                if row < valid_block_h and row % num_warps == warp_idx and subgroup_idx == 0 and lane32 == 0:
                    row_sum_shared[row] = 0
                    for g in T.serial(4):
                        row_sum_shared[row] += partial_sum_shared[row, g]

            T.sync_threads()

            for row in T.serial(block_h):
                if row < valid_block_h and row % num_warps == warp_idx and lane_idx < head_size:
                    out_local = T.alloc_local([1], accum_dtype)
                    out_local[0] = 0
                    for seg_idx in T.serial(num_segments):
                        scale_local = T.alloc_local([1], accum_dtype)
                        scale_local[0] = T.if_then_else(
                            row_sum_shared[row] == 0,
                            0,
                            T.exp2((segm_max_ptr[bx, by, seg_idx, row] - row_max_shared[row]) * scale_log2e) / row_sum_shared[row],
                        )
                        out_local[0] += segm_output_ptr[bx, by, seg_idx, row, lane_idx] * scale_local[0]
                    output_ptr[q_tok_idx, head_start + row, lane_idx] = T.cast(out_local[0], out_dtype)

    return main


# 2D float kernel
@tilelang.jit(
    out_idx=[0],
    pass_configs={
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
    },
    compile_flags=_UA_GEMV_EXTRA_MCC_FLAGS,
)
def tilelang_unified_attention_2d_basic(
    num_query_heads,
    num_kv_heads,
    num_queries_per_kv,
    head_size,
    num_seqs,
    num_total_blocks,
    block_size,
    total_num_q_tokens,
    max_seqlen_q,
    max_num_blocks_per_seq,
    softmax_scale,
    softcap,
    sliding_window,
    use_softcap,
    use_alibi,
    use_sink,
    use_qq_bias,
    BLOCK_M,
    BLOCK_Q,
    BLOCK_N,
    threads,
    num_stages,
    is_int8_kv=False,
    k_descale=1.0,
    v_descale=1.0,
):
    dtype = "float16"
    kv_storage_dtype = "int8" if is_int8_kv else dtype
    accum_dtype = "float"
    scale_log2e = 1.44269504
    mask_neg = -1.0e20
    vec = 16
    vec_groups = _cdiv(head_size, vec)
    row_groups = threads // vec_groups
    rows_per_thread = _cdiv(BLOCK_N, row_groups)

    @T.macro
    def SoftmaxInt8(
        acc_s: T.FragmentBuffer([BLOCK_M, BLOCK_N], accum_dtype),
        acc_s_cast: T.FragmentBuffer([BLOCK_M, BLOCK_N], dtype),
        scores_max: T.FragmentBuffer([BLOCK_M], accum_dtype),
        scores_max_safe: T.FragmentBuffer([BLOCK_M], accum_dtype),
        scores_max_prev: T.FragmentBuffer([BLOCK_M], accum_dtype),
        scores_scale: T.FragmentBuffer([BLOCK_M], accum_dtype),
        scores_sum: T.FragmentBuffer([BLOCK_M], accum_dtype),
        logsum: T.FragmentBuffer([BLOCK_M], accum_dtype),
    ):
        T.copy(scores_max, scores_max_prev)
        T.fill(scores_max, -T.infinity(accum_dtype))
        T.reduce_max(acc_s, scores_max, dim=1, clear=False)
        row_empty = T.alloc_fragment([BLOCK_M], "int32")

        for i in T.Parallel(BLOCK_M):
            row_empty[i] = T.if_then_else(
                (scores_max[i] == -T.infinity(accum_dtype)) or (scores_max[i] < mask_neg * 0.5),
                1,
                0,
            )
            scores_max[i] = T.if_then_else(
                row_empty[i] != 0,
                scores_max_prev[i],
                T.max(scores_max[i], scores_max_prev[i]),
            )
            scores_max_safe[i] = T.if_then_else(
                scores_max[i] == -T.infinity(accum_dtype),
                0,
                scores_max[i],
            )
            scores_scale[i] = T.if_then_else(
                row_empty[i] != 0,
                T.if_then_else(scores_max_prev[i] == -T.infinity(accum_dtype), 0, 1),
                T.exp2(scores_max_prev[i] * scale_log2e - scores_max_safe[i] * scale_log2e),
            )

        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            acc_s[i, j] = T.if_then_else(
                row_empty[i] != 0,
                0,
                T.exp2(acc_s[i, j] * scale_log2e - scores_max_safe[i] * scale_log2e),
            )

        T.reduce_sum(acc_s, scores_sum, dim=1)

        for i in T.Parallel(BLOCK_M):
            logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]

        temp_shared = T.alloc_shared([BLOCK_M, BLOCK_N], accum_dtype)
        T.copy(acc_s, temp_shared)
        T.copy(temp_shared, acc_s_cast)

    @T.macro
    def RescaleInt8(
        acc_o: T.FragmentBuffer([BLOCK_M, head_size], accum_dtype),
        scores_scale: T.FragmentBuffer([BLOCK_M], accum_dtype),
    ):
        for i, j in T.Parallel(BLOCK_M, head_size):
            acc_o[i, j] *= scores_scale[i]

    @T.macro
    def ApplySoftcap(
        acc_s: T.FragmentBuffer([BLOCK_M, BLOCK_N], accum_dtype),
        q_block_local_idx: T.int32,
        cur_batch_query_len: T.int32,
        context_len: T.int32,
        k: T.int32,
    ):
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            query_pos = q_block_local_idx * BLOCK_Q + i // num_queries_per_kv
            seq_offset = k * BLOCK_N + j
            causal_limit = context_len + query_pos + 1
            if sliding_window >= 0:
                acc_s[i, j] = T.if_then_else(
                    (query_pos >= cur_batch_query_len)
                    or (seq_offset >= causal_limit)
                    or ((context_len + query_pos - seq_offset) >= sliding_window),
                    mask_neg,
                    softcap * T.tanh(acc_s[i, j] * softmax_scale / softcap),
                )
            else:
                acc_s[i, j] = T.if_then_else(
                    (query_pos >= cur_batch_query_len) or (seq_offset >= causal_limit),
                    -T.infinity(accum_dtype),
                    softcap * T.tanh(acc_s[i, j] * softmax_scale / softcap),
                )

    @T.macro
    def ApplyScale(
        acc_s: T.FragmentBuffer([BLOCK_M, BLOCK_N], accum_dtype),
        q_block_local_idx: T.int32,
        cur_batch_query_len: T.int32,
        context_len: T.int32,
        k: T.int32,
    ):
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            query_pos = q_block_local_idx * BLOCK_Q + i // num_queries_per_kv
            seq_offset = k * BLOCK_N + j
            causal_limit = context_len + query_pos + 1
            if sliding_window >= 0:
                acc_s[i, j] = T.if_then_else(
                    (query_pos >= cur_batch_query_len)
                    or (seq_offset >= causal_limit)
                    or ((context_len + query_pos - seq_offset) >= sliding_window),
                    mask_neg,
                    acc_s[i, j] * softmax_scale,
                )
            else:
                acc_s[i, j] = T.if_then_else(
                    (query_pos >= cur_batch_query_len) or (seq_offset >= causal_limit),
                    -T.infinity(accum_dtype),
                    acc_s[i, j] * softmax_scale,
                )

    @T.macro
    def ApplyAlibi(
        acc_s: T.FragmentBuffer([BLOCK_M, BLOCK_N], accum_dtype),
        alibi_shared: T.SharedBuffer([BLOCK_M], "float32"),
        context_len: T.int32,
        k: T.int32,
    ):
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            seq_offset = k * BLOCK_N + j
            acc_s[i, j] = T.if_then_else(
                acc_s[i, j] == -T.infinity(accum_dtype),
                acc_s[i, j],
                acc_s[i, j] + alibi_shared[i] * T.Cast(accum_dtype, seq_offset - context_len),
            )

    @T.macro
    def ApplyQQBias(
        acc_s: T.FragmentBuffer([BLOCK_M, BLOCK_N], accum_dtype),
        qq_bias_shared: T.SharedBuffer([BLOCK_M, BLOCK_N], "float32"),
        cur_batch_query_len: T.int32,
        context_len: T.int32,
        k: T.int32,
    ):
        for i, j in T.Parallel(BLOCK_M, BLOCK_N):
            seq_offset = k * BLOCK_N + j
            key_rel = seq_offset - context_len
            acc_s[i, j] = T.if_then_else(
                (acc_s[i, j] == -T.infinity(accum_dtype)) or (key_rel < 0) or (key_rel >= cur_batch_query_len),
                acc_s[i, j],
                acc_s[i, j] + qq_bias_shared[i, j],
            )

    @T.macro
    def DecodeSignedInt8Vec(src_i8, tmp_f32, dst_f16, descale):
        for vi in T.serial(vec):
            tmp_f32[vi] = src_i8[vi]
            tmp_f32[vi] = T.if_then_else(tmp_f32[vi] > 127.0, tmp_f32[vi] - 256.0, tmp_f32[vi])
            tmp_f32[vi] = tmp_f32[vi] * descale
            dst_f16[vi] = tmp_f32[vi]

    @T.prim_func
    def main(
        output_ptr: T.Tensor([total_num_q_tokens, num_query_heads, head_size], dtype),
        query_ptr: T.Tensor([total_num_q_tokens, num_query_heads, head_size], dtype),
        key_cache_ptr: T.Tensor([num_total_blocks, block_size, num_kv_heads, head_size], kv_storage_dtype),
        value_cache_ptr: T.Tensor([num_total_blocks, block_size, num_kv_heads, head_size], kv_storage_dtype),
        block_tables_ptr: T.Tensor([num_seqs, max_num_blocks_per_seq], "int32"),
        seq_lens_ptr: T.Tensor([num_seqs], "int32"),
        query_start_len_ptr: T.Tensor([num_seqs + 1], "int32"),
        alibi_slopes_ptr: T.Tensor([num_query_heads], "float32"),
        sinks_ptr: T.Tensor([num_query_heads], "float32"),
        qq_bias_ptr: T.Tensor([max_seqlen_q, max_seqlen_q], "float32"),
    ):
        with T.Kernel(T.ceildiv(max_seqlen_q, BLOCK_Q), num_kv_heads, num_seqs, threads=threads) as (bx, by, bz):
            cur_batch_in_all_start_index = query_start_len_ptr[bz]
            cur_batch_in_all_stop_index = query_start_len_ptr[bz + 1]
            cur_batch_query_len = cur_batch_in_all_stop_index - cur_batch_in_all_start_index
            q_block_local_idx = bx
            q_block_valid = q_block_local_idx * BLOCK_Q < cur_batch_query_len

            q_shared = T.alloc_shared([BLOCK_M, head_size], dtype)
            k_shared = T.alloc_shared([BLOCK_N, head_size], dtype)
            v_shared = T.alloc_shared([BLOCK_N, head_size], dtype)
            o_shared = T.alloc_shared([BLOCK_M, head_size], dtype)
            alibi_shared = T.alloc_shared([BLOCK_M], "float32")
            sink_shared = T.alloc_shared([BLOCK_M], "float32")
            qq_bias_shared = T.alloc_shared([BLOCK_M, BLOCK_N], "float32")

            acc_s = T.alloc_fragment([BLOCK_M, BLOCK_N], accum_dtype)
            acc_s_cast = T.alloc_fragment([BLOCK_M, BLOCK_N], dtype)
            acc_o = T.alloc_fragment([BLOCK_M, head_size], accum_dtype)
            scores_max = T.alloc_fragment([BLOCK_M], accum_dtype)
            scores_max_safe = T.alloc_fragment([BLOCK_M], accum_dtype)
            scores_max_prev = T.alloc_fragment([BLOCK_M], accum_dtype)
            scores_scale = T.alloc_fragment([BLOCK_M], accum_dtype)
            scores_sum = T.alloc_fragment([BLOCK_M], accum_dtype)
            logsum = T.alloc_fragment([BLOCK_M], accum_dtype)
            tid = T.get_thread_binding()
            tx = tid % vec_groups
            ty = tid // vec_groups
            k_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
            v_reg_i8 = T.alloc_local([vec], kv_storage_dtype)
            k_reg_f32 = T.alloc_local([vec], "float32")
            v_reg_f32 = T.alloc_local([vec], "float32")
            k_reg_f16 = T.alloc_local([vec], dtype)
            v_reg_f16 = T.alloc_local([vec], dtype)

            for i, d in T.Parallel(BLOCK_M, head_size):
                query_pos = q_block_local_idx * BLOCK_Q + i // num_queries_per_kv
                query_head_idx = by * num_queries_per_kv + i % num_queries_per_kv
                q_shared[i, d] = T.if_then_else(
                    (query_pos < cur_batch_query_len) and (query_head_idx < num_query_heads),
                    query_ptr[cur_batch_in_all_start_index + query_pos, query_head_idx, d],
                    0,
                )
            for i in T.Parallel(BLOCK_M):
                query_head_idx = by * num_queries_per_kv + i % num_queries_per_kv
                alibi_shared[i] = T.if_then_else(
                    use_alibi and (query_head_idx < num_query_heads),
                    alibi_slopes_ptr[query_head_idx],
                    T.cast(0, "float32"),
                )
                sink_shared[i] = T.if_then_else(
                    use_sink and (query_head_idx < num_query_heads),
                    sinks_ptr[query_head_idx],
                    T.cast(0, "float32"),
                )

            T.fill(acc_o, 0)
            for i in T.Parallel(BLOCK_M):
                query_pos = q_block_local_idx * BLOCK_Q + i // num_queries_per_kv
                query_head_idx = by * num_queries_per_kv + i % num_queries_per_kv
                if use_sink and query_pos < cur_batch_query_len and query_head_idx < num_query_heads:
                    logsum[i] = 1
                    scores_max[i] = sink_shared[i]
                else:
                    logsum[i] = 0
                    scores_max[i] = -T.infinity(accum_dtype)

            seq_len = seq_lens_ptr[bz]
            context_len = seq_len - cur_batch_query_len
            max_seq_prefix_len = T.min(seq_len, context_len + q_block_local_idx * BLOCK_Q + BLOCK_Q)
            num_blocks = T.if_then_else(q_block_valid, T.ceildiv(max_seq_prefix_len, BLOCK_N), 0)
            start_block = T.alloc_local([1], "int32")
            if sliding_window >= 0:
                start_block[0] = T.max(0, (context_len + q_block_local_idx * BLOCK_Q - sliding_window) // BLOCK_N)
            else:
                start_block[0] = 0

            loop_range = T.alloc_local([1], "int32")
            loop_range[0] = T.max(0, num_blocks - start_block[0])
            for k_iter in T.Pipelined(loop_range[0], num_stages=num_stages):
                k = start_block[0] + k_iter
                if is_int8_kv:
                    T.fill(k_shared, T.cast(0, dtype))
                    if tx < vec_groups:
                        for row_iter in T.serial(rows_per_thread):
                            row = ty + row_iter * row_groups
                            if row < BLOCK_N:
                                seq_offset = k * BLOCK_N + row
                                logical_block_idx = seq_offset // block_size
                                block_offset = seq_offset % block_size
                                physical_block_idx = block_tables_ptr[bz, logical_block_idx]
                                valid = seq_offset < seq_len
                                for vi in T.serial(vec):
                                    d = tx * vec + vi
                                    k_reg_i8[vi] = T.if_then_else(
                                        valid and (d < head_size),
                                        key_cache_ptr[physical_block_idx, block_offset, by, d],
                                        T.cast(0, kv_storage_dtype),
                                    )
                                DecodeSignedInt8Vec(k_reg_i8, k_reg_f32, k_reg_f16, k_descale)
                                for vi in T.vectorized(vec):
                                    d = tx * vec + vi
                                    if d < head_size:
                                        k_shared[row, d] = k_reg_f16[vi]
                    T.sync_threads()
                else:
                    for i, d in T.Parallel(BLOCK_N, head_size):
                        seq_offset = k * BLOCK_N + i
                        logical_block_idx = seq_offset // block_size
                        block_offset = seq_offset % block_size
                        physical_block_idx = block_tables_ptr[bz, logical_block_idx]
                        k_shared[i, d] = T.if_then_else(
                            seq_offset < seq_len,
                            key_cache_ptr[physical_block_idx, block_offset, by, d],
                            0,
                        )
                for i, j in T.Parallel(BLOCK_M, BLOCK_N):
                    query_pos = q_block_local_idx * BLOCK_Q + i // num_queries_per_kv
                    seq_offset = k * BLOCK_N + j
                    key_rel = seq_offset - context_len
                    qq_bias_shared[i, j] = T.if_then_else(
                        use_qq_bias and (query_pos < cur_batch_query_len) and (key_rel >= 0) and (key_rel < cur_batch_query_len),
                        qq_bias_ptr[query_pos, key_rel],
                        T.cast(0, "float32"),
                    )

                T.clear(acc_s)
                T.gemm(q_shared, k_shared, acc_s, transpose_B=True, policy=T.GemmWarpPolicy.FullRow)
                if use_softcap:
                    ApplySoftcap(acc_s, q_block_local_idx, cur_batch_query_len, context_len, k)
                else:
                    ApplyScale(acc_s, q_block_local_idx, cur_batch_query_len, context_len, k)
                if use_alibi:
                    ApplyAlibi(acc_s, alibi_shared, context_len, k)
                if use_qq_bias:
                    ApplyQQBias(acc_s, qq_bias_shared, cur_batch_query_len, context_len, k)

                SoftmaxInt8(
                    acc_s,
                    acc_s_cast,
                    scores_max,
                    scores_max_safe,
                    scores_max_prev,
                    scores_scale,
                    scores_sum,
                    logsum,
                )
                RescaleInt8(acc_o, scores_scale)
                if is_int8_kv:
                    T.fill(v_shared, T.cast(0, dtype))
                    if tx < vec_groups:
                        for row_iter in T.serial(rows_per_thread):
                            row = ty + row_iter * row_groups
                            if row < BLOCK_N:
                                seq_offset = k * BLOCK_N + row
                                logical_block_idx = seq_offset // block_size
                                block_offset = seq_offset % block_size
                                physical_block_idx = block_tables_ptr[bz, logical_block_idx]
                                valid = seq_offset < seq_len
                                for vi in T.serial(vec):
                                    d = tx * vec + vi
                                    v_reg_i8[vi] = T.if_then_else(
                                        valid and (d < head_size),
                                        value_cache_ptr[physical_block_idx, block_offset, by, d],
                                        T.cast(0, kv_storage_dtype),
                                    )
                                DecodeSignedInt8Vec(v_reg_i8, v_reg_f32, v_reg_f16, v_descale)
                                for vi in T.vectorized(vec):
                                    d = tx * vec + vi
                                    if d < head_size:
                                        v_shared[row, d] = v_reg_f16[vi]
                    T.sync_threads()
                else:
                    for i, d in T.Parallel(BLOCK_N, head_size):
                        seq_offset = k * BLOCK_N + i
                        logical_block_idx = seq_offset // block_size
                        block_offset = seq_offset % block_size
                        physical_block_idx = block_tables_ptr[bz, logical_block_idx]
                        v_shared[i, d] = T.if_then_else(
                            seq_offset < seq_len,
                            value_cache_ptr[physical_block_idx, block_offset, by, d],
                            0,
                        )
                    T.sync_threads()
                T.gemm(acc_s_cast, v_shared, acc_o, transpose_B=False, policy=T.GemmWarpPolicy.FullRow)

            for i, d in T.Parallel(BLOCK_M, head_size):
                safe_logsum = T.if_then_else(logsum[i] > 0, logsum[i], 1.0)
                o_shared[i, d] = T.if_then_else(logsum[i] > 0, acc_o[i, d] / safe_logsum, 0)

            for i, d in T.Parallel(BLOCK_M, head_size):
                query_pos = q_block_local_idx * BLOCK_Q + i // num_queries_per_kv
                query_head_idx = by * num_queries_per_kv + i % num_queries_per_kv
                if query_pos < cur_batch_query_len and query_head_idx < num_query_heads:
                    output_ptr[cur_batch_in_all_start_index + query_pos, query_head_idx, d] = o_shared[i, d]

    return main
