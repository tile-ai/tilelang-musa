import os

import pytest

import torch

import tile_kernels
from tile_kernels.config import set_num_sms
from tile_kernels.testing.generator import generate_topk_idx, generate_moe_params, generate_num_sms
from tile_kernels.testing.numeric import count_bytes
from tile_kernels.testing.bench import make_param_id
import tilelang.testing

# Disable TileLang prints
os.environ['TILELANG_PRINT_ON_COMPILATION'] = '0'


def _has_musa() -> bool:
    return hasattr(torch, 'musa') and torch.musa.is_available()


def _get_fused_mapping_ref(
    topk_idx: torch.Tensor,
    num_experts: int,
    alignment: int,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, list[int]]:
    device = topk_idx.device
    num_tokens, num_topk = topk_idx.shape
    flat_topk_idx = topk_idx.reshape(-1)
    valid_mask = flat_topk_idx >= 0
    valid_flat_idx = torch.nonzero(valid_mask, as_tuple=False).flatten()
    valid_expert_idx = flat_topk_idx[valid_mask].to(torch.int64)

    counts = torch.bincount(valid_expert_idx, minlength=num_experts).to(torch.int32)
    aligned_counts = ((counts + alignment - 1) // alignment) * alignment
    expert_end = torch.cumsum(aligned_counts, dim=0)
    expert_start = expert_end - aligned_counts
    total_expanded = int(expert_end[-1].item()) if num_experts > 0 else 0

    pos_to_expert = torch.full((total_expanded,), -1, dtype=torch.int32, device=device)
    pos_to_token = torch.full((total_expanded,), -1, dtype=torch.int32, device=device)
    pos_to_token_topk = torch.full((total_expanded,), -1, dtype=torch.int32, device=device)
    token_topk_to_pos = torch.full((num_tokens, num_topk), -1, dtype=torch.int32, device=device)

    if valid_flat_idx.numel() > 0:
        sort_order = torch.argsort(valid_expert_idx, stable=True)
        sorted_expert_idx = valid_expert_idx[sort_order]
        sorted_flat_idx = valid_flat_idx[sort_order].to(torch.int32)
        counts_i64 = counts.to(torch.int64)
        expert_prefix = torch.cumsum(counts_i64, dim=0) - counts_i64
        occurrence = torch.arange(sorted_flat_idx.numel(), device=device, dtype=torch.int64) - expert_prefix[sorted_expert_idx]
        dst_pos = expert_start[sorted_expert_idx].to(torch.int64) + occurrence

        pos_to_expert[dst_pos] = sorted_expert_idx.to(torch.int32)
        pos_to_token[dst_pos] = sorted_flat_idx // num_topk
        pos_to_token_topk[dst_pos] = sorted_flat_idx
        token_topk_to_pos.view(-1)[sorted_flat_idx.to(torch.int64)] = dst_pos.to(torch.int32)

    return (
        pos_to_expert,
        pos_to_token,
        pos_to_token_topk,
        token_topk_to_pos,
        expert_start.contiguous(),
        expert_end.contiguous(),
        aligned_counts.contiguous(),
        aligned_counts.tolist(),
    )


def generate_test_data(params):
    num_experts = params['num_experts']

    topk_idx = generate_topk_idx(params)
    num_tokens = topk_idx.shape[0]

    return (topk_idx, num_tokens)


def generate_test_params(is_benchmark: bool) -> list[dict]:
    params = [
        {**moe, 'alignment': alignment}
        for moe in generate_moe_params(is_benchmark=is_benchmark)
        for alignment in (64, 128)
    ]
    if is_benchmark:
        params = [{**param, 'alignment': 128} for param in params]
    return params


@pytest.mark.parametrize('params', generate_test_params(is_benchmark=False), ids=make_param_id)
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_get_fused_mapping(params):
    alignment = params['alignment']

    topk_idx, num_tokens = generate_test_data(params)
    num_topk = params['num_topk']
    num_experts = params['num_experts']

    func = lambda: tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, alignment)

    for num_sms in generate_num_sms():
        set_num_sms(num_sms)

        pos_to_expert, pos_to_token, pos_to_token_topk, token_topk_to_pos, expert_start, expert_end, num_tokens_per_expert, num_tokens_per_expert_list = func()
        assert num_tokens_per_expert.tolist() == num_tokens_per_expert_list
        start = 0

        # Check `pos_to_expert`, `num_tokens_per_expert`, `expert_start`, `expert_end` correctness
        for i in range(num_experts):
            assert start == expert_start[i].item()
            s = pos_to_expert[start:start + num_tokens_per_expert_list[i]]
            assert (s == i).int().sum().item() == (topk_idx == i).int().sum().item()
            s = (s == i) + (s == -1)
            assert s.int().sum().item() == s.numel()
            start += num_tokens_per_expert_list[i]
            assert start == expert_end[i].item()

        non_negative_mask = pos_to_expert >= 0

        if non_negative_mask.any():
            t_values = pos_to_token_topk[non_negative_mask]
            token_indices = t_values // num_topk
            topk_indices = t_values % num_topk
            expected_indices = torch.arange(pos_to_token_topk.numel(), device=pos_to_token_topk.device)[non_negative_mask]
            actual_indices = token_topk_to_pos[token_indices, topk_indices]
            assert torch.equal(actual_indices, expected_indices)
            assert torch.equal(topk_idx[token_indices, topk_indices], pos_to_expert[non_negative_mask])
            assert torch.equal(pos_to_token_topk[non_negative_mask] // num_topk, pos_to_token[non_negative_mask])

        negative_mask = pos_to_expert < 0
        assert torch.equal(negative_mask, pos_to_token < 0)
        assert torch.equal(negative_mask, pos_to_token_topk < 0)


@pytest.mark.benchmark
@pytest.mark.parametrize('params', generate_test_params(is_benchmark=True), ids=make_param_id)
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_get_fused_mapping_benchmark(benchmark_timer, benchmark_record, params):
    alignment = params['alignment']

    topk_idx, num_tokens = generate_test_data(params)
    num_experts = params['num_experts']

    func = lambda: tile_kernels.moe.get_fused_mapping(topk_idx, num_experts, 0, alignment)

    t_us = benchmark_timer(func)
    result = func()
    num_bytes = count_bytes(topk_idx, *result[:7])
    bandwidth_gbs = num_bytes / t_us / 1e3

    params.pop('num_send_tokens')
    benchmark_record(
        kernel='get_fused_mapping',
        operation='fwd',
        params={'num_tokens': num_tokens, **params},
        time_us=t_us,
        bandwidth_gbs=bandwidth_gbs,
    )


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_get_fused_mapping_musa_focused_correctness() -> None:
    if not _has_musa():
        pytest.skip("MUSA is not available")

    topk_idx = torch.tensor(
        [
            [0, 1],
            [1, 2],
            [0, 2],
            [2, 3],
        ],
        device='musa',
        dtype=torch.int64,
    ).contiguous()

    result = tile_kernels.moe.get_fused_mapping(topk_idx, num_experts=4, num_expanded_tokens=0, alignment=1)
    ref = _get_fused_mapping_ref(topk_idx, num_experts=4, alignment=1)

    for got, expected in zip(result[:7], ref[:7]):
        assert torch.equal(got, expected), f"Mismatch in get_fused_mapping output\n{got}\nvs\n{expected}"
    assert result[7] == ref[7]
