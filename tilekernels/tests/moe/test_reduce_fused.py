import os
import torch

import pytest

import tile_kernels
from tile_kernels.testing.bench import dtype_to_str, make_param_id
from tile_kernels.testing.generator import generate_topk_idx, generate_hidden_sizes, generate_moe_params
from tile_kernels.testing.numeric import assert_equal, count_bytes
import tilelang.testing

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def _assert_float_close(x: torch.Tensor, y: torch.Tensor) -> None:
    assert x.dtype == y.dtype, f'Tensor dtypes are not equal: {x.dtype} vs {y.dtype}'
    assert x.shape == y.shape, f'Tensor shapes are not equal: {x.shape} vs {y.shape}'
    assert x.device == y.device, f'Tensor devices are not equal: {x.device} vs {y.device}'
    if x.numel() == 0:
        return
    if x.dtype == torch.bfloat16:
        torch.testing.assert_close(x, y, atol=4e-3, rtol=1e-2)
    elif x.dtype == torch.float8_e4m3fn:
        torch.testing.assert_close(x.float(), y.float(), atol=5e-1, rtol=1e-1)
    else:
        torch.testing.assert_close(x, y, atol=1e-5, rtol=1e-5)


def _has_musa() -> bool:
    return hasattr(torch, 'musa') and torch.musa.is_available()


def _reduce_fused_ref(
    expanded: torch.Tensor,
    topk_weights: torch.Tensor,
    token_topk_to_pos: torch.Tensor,
) -> torch.Tensor:
    num_tokens, num_topk = token_topk_to_pos.shape
    hidden = expanded.shape[1]
    out = torch.zeros((num_tokens, hidden), dtype=torch.float32, device=expanded.device)
    for token in range(num_tokens):
        for topk in range(num_topk):
            pos = int(token_topk_to_pos[token, topk].item())
            if pos >= 0:
                out[token] += expanded[pos].float() * topk_weights[token, topk]
    return out.to(expanded.dtype)


def generate_test_data(params):
    hidden = params['hidden']
    with_weights = params['with_weights']
    in_dtype = params['in_dtype']
    out_dtype = params['out_dtype']
    with_sf = params['with_sf']
    num_experts = params['num_experts']
    num_ep_ranks = params['num_ep_ranks']
    num_topk = params['num_topk']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]
    num_expanded_tokens = num_tokens * num_topk
    expanded = torch.randn((num_expanded_tokens, hidden), dtype=in_dtype, device='musa')
    _, _, _, token_topk_to_pos, _, _, _, _ = tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, 1)

    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device='musa') if with_weights else None
    if out_dtype == torch.float8_e4m3fn:
        sf = torch.randn((1,), dtype=torch.float32, device='musa')
    else:
        sf = None
    if with_sf:
        x_sf = torch.randn((num_expanded_tokens,), dtype=torch.float32, device='musa')
    else:
        x_sf = None
    fp8_format = 'e4m3' if out_dtype == torch.float8_e4m3fn else ''

    x_input = (expanded, x_sf) if x_sf is not None else expanded

    return (expanded, token_topk_to_pos, topk_weights, sf, x_sf, fp8_format, x_input, num_tokens)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {**moe, 'hidden': hidden, 'with_weights': with_weights,
         'in_dtype': in_dtype, 'out_dtype': out_dtype, 'with_sf': with_sf}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for hidden in generate_hidden_sizes(256)
        for with_weights in (True, False)
        for in_dtype in (torch.float32, torch.bfloat16)
        for out_dtype in (in_dtype, torch.float8_e4m3fn)
        for with_sf in (True, False)
    ]
    if is_benchmark:
        params = [p for p in params if p['num_topk'] == 6 and p['with_weights']]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_reduce_fused(params):
    (expanded, token_topk_to_pos, topk_weights, sf, x_sf, fp8_format, x_input,
     _) = generate_test_data(params)

    # Test correctness: tile_kernels kernel
    func = lambda: tile_kernels.moe.reduce_fused(
        x_input, topk_weights, token_topk_to_pos, fp8_format, sf, None
    )
    r_tk = func()

    # Test correctness: torch reference
    r_ref = tile_kernels.torch.reduce_fused(
        x_input, topk_weights, token_topk_to_pos, fp8_format, sf
    )
    if r_tk.is_floating_point() and r_tk.dtype in (torch.float32, torch.bfloat16, torch.float8_e4m3fn):
        _assert_float_close(r_tk, r_ref)
    else:
        assert_equal(r_tk, r_ref)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_reduce_fused_benchmark(benchmark_timer, benchmark_record, params):
    hidden = params['hidden']
    out_dtype = params['out_dtype']

    (expanded, token_topk_to_pos, topk_weights, sf, x_sf, fp8_format, x_input,
     num_tokens) = generate_test_data(params)
    in_dtype = params['in_dtype']

    func = lambda: tile_kernels.moe.reduce_fused(
        x_input, topk_weights, token_topk_to_pos, fp8_format, sf, None
    )
    r_tk = func()

    num_bytes = count_bytes(token_topk_to_pos, x_sf, r_tk)
    num_bytes += torch.count_nonzero(token_topk_to_pos != -1).item() * hidden * (torch.finfo(in_dtype).bits // 8)
    if topk_weights is not None:
        num_bytes += count_bytes(topk_weights)

    t_us = benchmark_timer(func)

    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='reduce_fused',
        operation='fwd',
        params={'num_tokens': num_tokens, **params, 'in_dtype': dtype_to_str(in_dtype), 'out_dtype': dtype_to_str(out_dtype)},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_reduce_fused_musa_focused_correctness() -> None:
    if not _has_musa():
        pytest.skip("MUSA is not available")

    expanded = torch.randn((8, 256), dtype=torch.float16, device='musa').contiguous()
    token_topk_to_pos = torch.tensor(
        [
            [0, 2],
            [3, 4],
            [1, 5],
            [6, 7],
        ],
        device='musa',
        dtype=torch.int32,
    ).contiguous()
    topk_weights = torch.tensor(
        [
            [0.25, 0.75],
            [0.60, 0.40],
            [0.50, 0.50],
            [0.10, 0.90],
        ],
        device='musa',
        dtype=torch.float32,
    ).contiguous()

    reduced = tile_kernels.moe.reduce_fused(expanded, topk_weights, token_topk_to_pos)
    reduced_ref = _reduce_fused_ref(expanded, topk_weights, token_topk_to_pos)
    torch.testing.assert_close(reduced, reduced_ref, atol=1e-3, rtol=1e-3)
