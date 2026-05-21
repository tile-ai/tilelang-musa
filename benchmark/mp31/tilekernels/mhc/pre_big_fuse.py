import torch

from .norm_fn_kernel import _mhc_pre_norm_fn_fwd_mul, round_to_tf32
from .pre_big_fuse_kernel import _mhc_pre_big_fuse


def _resolve_big_fuse_config(
    num_tokens: int,
    threads: int | None,
    hidden_block: int | None,
    pass_config: str,
) -> tuple[int, int, str]:
    is_decode_like = num_tokens <= 64
    if threads is None or threads <= 0:
        threads = 128
    if hidden_block is None or hidden_block <= 0:
        hidden_block = 256 if is_decode_like else 1024
    pass_config = pass_config.strip().lower()
    if pass_config == "auto":
        pass_config = "burst" if is_decode_like else "safe"
    if pass_config not in ("safe", "burst", "aggressive", "none"):
        raise ValueError(f"pass_config must be one of 'auto', 'safe', 'burst', 'aggressive', or 'none', got {pass_config!r}")
    return threads, hidden_block, pass_config


def mhc_pre_big_fuse(
    residual: torch.Tensor,
    fn: torch.Tensor,
    mhc_scale: torch.Tensor,
    mhc_base: torch.Tensor,
    rms_eps: float,
    mhc_pre_eps: float,
    mhc_sinkhorn_eps: float,
    mhc_post_mult_value: float,
    sinkhorn_repeat: int,
    n_splits: int = 16,
    threads: int | None = None,
    hidden_block: int | None = None,
    pass_config: str = "auto",
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    assert residual.dtype == torch.bfloat16
    assert fn.dtype == torch.float32
    assert mhc_scale.dtype == torch.float32
    assert mhc_base.dtype == torch.float32

    mhc_mult = residual.shape[-2]
    hidden_size = residual.shape[-1]
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2

    mhc_hidden_size = mhc_mult * hidden_size
    assert fn.shape[0] == mhc_mult3
    assert fn.shape[1] == mhc_hidden_size
    assert mhc_scale.shape == (3,)
    assert mhc_base.shape == (mhc_mult3,)

    outer_shape = residual.shape[:-2]

    residual_flat = residual.view(-1, mhc_mult, hidden_size)
    num_tokens = residual_flat.shape[0]
    fn_flat = fn

    post_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=residual.device)
    comb_mix = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=residual.device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=residual.device)

    gemm_out_mul = torch.empty(n_splits, num_tokens, mhc_mult3, dtype=torch.float32, device=residual.device)
    gemm_out_sqrsum = torch.empty(n_splits, num_tokens, dtype=torch.float32, device=residual.device)

    # TileLang implementation doesn't support split-k, so we set n_splits to 1
    # You may want to adopt the DeepGEMM implementation with split-k for better performance
    n_splits = 1
    gemm_out_mul = gemm_out_mul[:1]
    gemm_out_sqrsum = gemm_out_sqrsum[:1]

    fn = round_to_tf32(fn)

    fwd_mul_kernel = _mhc_pre_norm_fn_fwd_mul(mhc_mult3, 1, mhc_hidden_size)
    fwd_mul_kernel(
        residual_flat.view(-1, mhc_hidden_size),
        fn,
        gemm_out_mul.view(-1, 1, mhc_mult3),
        gemm_out_sqrsum.view(-1, 1),
    )
    # END of TileLang implementation of pre-norm-fn forward matmul

    threads, hidden_block, pass_config = _resolve_big_fuse_config(
        num_tokens,
        threads,
        hidden_block,
        pass_config,
    )
    _mhc_pre_big_fuse(
        hidden_size,
        rms_eps,
        mhc_pre_eps,
        mhc_sinkhorn_eps,
        mhc_post_mult_value,
        sinkhorn_repeat,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
    )(
        gemm_out_mul,
        gemm_out_sqrsum,
        mhc_scale,
        mhc_base,
        residual_flat,
        post_mix,
        comb_mix,
        layer_input,
    )

    post_mix = post_mix.view(*outer_shape, mhc_mult, 1)
    comb_mix = comb_mix.view(*outer_shape, mhc_mult, mhc_mult)
    layer_input = layer_input.view(*outer_shape, hidden_size)

    return post_mix, comb_mix, layer_input
