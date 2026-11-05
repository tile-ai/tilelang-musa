from __future__ import annotations

import argparse
import copy
import json
import statistics
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import torch

from .benchmark_common import (
    BASELINE_FILENAME,
    DEFAULT_THRESHOLD,
    benchmark_root,
    benchmark_timer,
    check_regression,
    count_bytes,
    detect_tilelang_musa_build_type,
    ensure_release_build,
    get_test_device,
    import_ops,
    load_baselines,
    print_banner,
    print_json_record,
    print_perf,
    print_sample_stats,
    print_summary,
    sequential_topk_mapping,
    style,
    TermStyle,
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


def benchmark_mhc_pre_big_fuse_case(
    num_tokens: int,
    hidden_size: int,
    device: str,
    ops,
) -> dict[str, Any]:
    torch.manual_seed(0)
    mhc_mult = 4
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    n_splits = 1
    rms_eps = 1e-6
    mhc_pre_eps = 1e-6
    mhc_sinkhorn_eps = 1e-6
    mhc_post_mult_value = 1.0
    sinkhorn_repeat = 10

    gemm_out_mul = torch.randn((n_splits, num_tokens, mhc_mult3), dtype=torch.float32, device=device)
    gemm_out_sqrsum = torch.rand((n_splits, num_tokens), dtype=torch.float32, device=device)
    gemm_out_sqrsum *= mhc_mult * hidden_size
    mhc_scale = torch.randn((3,), dtype=torch.float32, device=device) * 0.1
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float32, device=device) * 0.1
    residual = torch.randn((num_tokens, mhc_mult, hidden_size), dtype=torch.float32, device=device).bfloat16()
    post_mix = torch.empty(num_tokens, mhc_mult, dtype=torch.float32, device=device)
    comb_mix = torch.empty(num_tokens, mhc_mult2, dtype=torch.float32, device=device)
    layer_input = torch.empty(num_tokens, hidden_size, dtype=torch.bfloat16, device=device)

    threads, hidden_block, pass_config = ops._resolve_big_fuse_config(
        num_tokens,
        threads=None,
        hidden_block=None,
        pass_config="auto",
    )
    kernel = ops._mhc_pre_big_fuse(
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
    )

    def fn() -> None:
        kernel(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
        )

    fn()
    t_us = benchmark_timer(fn)
    bytes_rw = count_bytes(
        gemm_out_mul,
        gemm_out_sqrsum,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    )
    return {
        "kernel": "mhc_pre_big_fuse",
        "operation": "kernel",
        "params": {
            "num_tokens": num_tokens,
            "hidden": hidden_size,
            "threads": threads,
            "hidden_block": hidden_block,
            "pass_config": pass_config,
        },
        "time_us": t_us,
        "bandwidth_gbs": bytes_rw / t_us / 1e3,
        "extras": {"bytes_rw": bytes_rw},
    }


def benchmark_representative_per_token_cast(ops, device: str) -> dict[str, Any]:
    num_tokens = 8001
    hidden = 4096
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)

    def fn():
        return ops.per_token_cast(
            x,
            "e4m3",
            num_per_channels=128,
            use_tma_aligned_col_major_sf=True,
            round_sf=True,
            use_packed_ue8m0=True,
        )

    out, out_sf = fn()
    t_us = benchmark_timer(fn)
    bytes_rw = count_bytes(x, out, out_sf)
    return {
        "kernel": "representative/per_token_cast",
        "operation": "bf16_to_fp8",
        "params": {"num_tokens": num_tokens, "hidden": hidden, "channels": 128},
        "time_us": t_us,
        "bandwidth_gbs": bytes_rw / t_us / 1e3,
        "extras": {"bytes_rw": bytes_rw},
    }


def benchmark_representative_per_block_cast(ops, device: str) -> dict[str, Any]:
    num_tokens = 8001
    hidden = 7168
    block_size = (128, 128)
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)

    def fn():
        return ops.per_block_cast(
            x,
            "e4m3",
            block_size=block_size,
            use_tma_aligned_col_major_sf=True,
            round_sf=True,
            use_packed_ue8m0=True,
        )

    out, out_sf = fn()
    t_us = benchmark_timer(fn)
    bytes_rw = count_bytes(x, out, out_sf)
    return {
        "kernel": "representative/per_block_cast",
        "operation": "bf16_to_fp8",
        "params": {"num_tokens": num_tokens, "hidden": hidden, "block": "128x128"},
        "time_us": t_us,
        "bandwidth_gbs": bytes_rw / t_us / 1e3,
        "extras": {"bytes_rw": bytes_rw},
    }


def benchmark_representative_swiglu(ops, device: str) -> dict[str, Any]:
    num_expanded_tokens = 4001
    hidden = 4096
    x = torch.randn((num_expanded_tokens, hidden * 2), dtype=torch.bfloat16, device=device)

    def fn():
        return ops.swiglu_forward_and_per_token_cast(
            x,
            "e4m3",
            num_per_channels=128,
            use_tma_aligned_col_major_sf=True,
            round_sf=True,
            use_packed_ue8m0=True,
        )

    out, out_sf = fn()
    t_us = benchmark_timer(fn)
    bytes_rw = count_bytes(x, out, out_sf)
    return {
        "kernel": "representative/swiglu_forward_and_per_token_cast",
        "operation": "fwd",
        "params": {"num_tokens": num_expanded_tokens, "hidden": hidden, "channels": 128},
        "time_us": t_us,
        "bandwidth_gbs": bytes_rw / t_us / 1e3,
        "extras": {"bytes_rw": bytes_rw},
    }


def benchmark_representative_topk_gate(ops, device: str) -> dict[str, Any]:
    num_tokens = 8001
    num_experts = 256
    num_topk = 8
    scores = torch.randn((num_tokens, num_experts), dtype=torch.float32, device=device)

    def fn():
        return ops.topk_gate(scores, num_topk)

    topk_idx = fn()
    t_us = benchmark_timer(fn)
    bytes_rw = count_bytes(scores, topk_idx)
    return {
        "kernel": "representative/topk_gate",
        "operation": "fwd",
        "params": {"num_tokens": num_tokens, "experts": num_experts, "topk": num_topk},
        "time_us": t_us,
        "bandwidth_gbs": bytes_rw / t_us / 1e3,
        "extras": {"bytes_rw": bytes_rw},
    }


def benchmark_representative_topk_group(
    ops,
    device: str,
    num_groups: int,
    num_experts_per_group: int,
) -> dict[str, Any]:
    num_tokens = 8001
    num_group_sum_topk = 2
    num_topk_groups = 4
    scores = torch.randn((num_tokens, num_groups, num_experts_per_group), dtype=torch.float32, device=device)

    def fn():
        return ops.topk_sum_and_topk_group_idx(scores, num_group_sum_topk, num_topk_groups)

    group_idx = fn()
    t_us = benchmark_timer(fn)
    bytes_rw = count_bytes(scores, group_idx)
    return {
        "kernel": "representative/topk_sum_and_topk_group_idx",
        "operation": "fwd",
        "params": {
            "num_tokens": num_tokens,
            "experts": num_groups * num_experts_per_group,
            "groups": num_groups,
        },
        "time_us": t_us,
        "bandwidth_gbs": bytes_rw / t_us / 1e3,
        "extras": {"bytes_rw": bytes_rw},
    }


def benchmark_representative_expand(ops, device: str) -> dict[str, Any]:
    num_tokens = 4001
    hidden = 4096
    num_topk = 6
    num_experts = 72
    x = torch.randn((num_tokens, hidden), dtype=torch.bfloat16, device=device)
    token_topk_to_pos, pos_to_expert = sequential_topk_mapping(num_tokens, num_topk, num_experts, device)

    def fn():
        return ops.expand_to_fused(x, token_topk_to_pos, pos_to_expert)

    expanded = fn()
    t_us = benchmark_timer(fn)
    bytes_rw = count_bytes(x, token_topk_to_pos, pos_to_expert, expanded)
    return {
        "kernel": "representative/expand_to_fused",
        "operation": "fwd",
        "params": {"num_tokens": num_tokens, "hidden": hidden, "topk": num_topk},
        "time_us": t_us,
        "bandwidth_gbs": bytes_rw / t_us / 1e3,
        "extras": {"bytes_rw": bytes_rw},
    }


def benchmark_representative_reduce(ops, device: str) -> dict[str, Any]:
    num_tokens = 4001
    hidden = 4096
    num_topk = 6
    num_experts = 72
    token_topk_to_pos, _ = sequential_topk_mapping(num_tokens, num_topk, num_experts, device)
    x = torch.randn((num_tokens * num_topk, hidden), dtype=torch.bfloat16, device=device)
    topk_weights = torch.rand((num_tokens, num_topk), dtype=torch.float32, device=device)

    def fn():
        return ops.reduce_fused(x, topk_weights, token_topk_to_pos)

    reduced = fn()
    t_us = benchmark_timer(fn)
    bytes_rw = count_bytes(token_topk_to_pos, topk_weights, reduced) + x.numel() * x.element_size()
    return {
        "kernel": "representative/reduce_fused",
        "operation": "fwd",
        "params": {"num_tokens": num_tokens, "hidden": hidden, "topk": num_topk},
        "time_us": t_us,
        "bandwidth_gbs": bytes_rw / t_us / 1e3,
        "extras": {"bytes_rw": bytes_rw},
    }


def benchmark_representative_mhc(
    num_tokens: int,
    hidden_size: int,
    device: str,
    ops,
) -> dict[str, Any]:
    mhc_mult = 4
    mhc_mult2 = mhc_mult * mhc_mult
    mhc_mult3 = mhc_mult * 2 + mhc_mult2
    n_splits = 1
    gemm_out_mul = torch.randn((n_splits, num_tokens, mhc_mult3), dtype=torch.float32, device=device)
    gemm_out_sqrsum = torch.rand((n_splits, num_tokens), dtype=torch.float32, device=device)
    mhc_scale = torch.randn((3,), dtype=torch.float32, device=device)
    mhc_base = torch.randn((mhc_mult3,), dtype=torch.float32, device=device)
    residual = torch.randn((num_tokens, mhc_mult, hidden_size), dtype=torch.float32, device=device).bfloat16()
    post_mix = torch.empty((num_tokens, mhc_mult), dtype=torch.float32, device=device)
    comb_mix = torch.empty((num_tokens, mhc_mult2), dtype=torch.float32, device=device)
    layer_input = torch.empty((num_tokens, hidden_size), dtype=torch.bfloat16, device=device)

    threads, hidden_block, pass_config = ops._resolve_big_fuse_config(
        num_tokens,
        threads=None,
        hidden_block=None,
        pass_config="auto",
    )
    kernel = ops._mhc_pre_big_fuse(
        hidden_size,
        1e-6,
        1e-6,
        1e-6,
        1.0,
        10,
        n_splits=n_splits,
        mhc_mult=mhc_mult,
        threads=threads,
        hidden_block=hidden_block,
        pass_config=pass_config,
    )

    def fn() -> None:
        kernel(
            gemm_out_mul,
            gemm_out_sqrsum,
            mhc_scale,
            mhc_base,
            residual,
            post_mix,
            comb_mix,
            layer_input,
        )

    fn()
    t_us = benchmark_timer(fn)
    bytes_rw = count_bytes(
        gemm_out_mul,
        gemm_out_sqrsum,
        residual,
        post_mix,
        comb_mix,
        layer_input,
    )
    return {
        "kernel": "representative/mhc_pre_big_fuse",
        "operation": "kernel",
        "params": {
            "num_tokens": num_tokens,
            "hidden": hidden_size,
            "threads": threads,
            "hidden_block": hidden_block,
            "pass_config": pass_config,
        },
        "time_us": t_us,
        "bandwidth_gbs": bytes_rw / t_us / 1e3,
        "extras": {"bytes_rw": bytes_rw},
    }


def build_cases() -> list[CaseSpec]:
    return [
        CaseSpec("mhc_pre_big_fuse_16_4096", "mhc_raw", (16, 4096)),
        CaseSpec("mhc_pre_big_fuse_512_1280", "mhc_raw", (512, 1280)),
        CaseSpec("mhc_pre_big_fuse_2048_4096", "mhc_raw", (2048, 4096)),
        CaseSpec("mhc_pre_big_fuse_8192_4096", "mhc_raw", (8192, 4096)),
        CaseSpec("representative_per_token_cast", "per_token", ()),
        CaseSpec("representative_per_block_cast", "per_block", ()),
        CaseSpec("representative_swiglu_forward_and_per_token_cast", "swiglu", ()),
        CaseSpec("representative_topk_gate", "topk_gate", ()),
        CaseSpec("representative_topk_group_72_12", "topk_group", (12, 6)),
        CaseSpec("representative_topk_group_256_8", "topk_group", (8, 32)),
        CaseSpec("representative_topk_group_256_16", "topk_group", (16, 16)),
        CaseSpec("representative_expand_to_fused", "expand", ()),
        CaseSpec("representative_reduce_fused", "reduce", ()),
        CaseSpec("representative_mhc_pre_big_fuse_2048_4096", "mhc_rep", (2048, 4096)),
        CaseSpec("representative_mhc_pre_big_fuse_8192_4096", "mhc_rep", (8192, 4096)),
    ]


def build_case_map() -> dict[str, CaseSpec]:
    return {case.name: case for case in build_cases()}


def run_case(case: CaseSpec, ops, device: str) -> dict[str, Any]:
    if case.runner == "mhc_raw":
        return benchmark_mhc_pre_big_fuse_case(*case.args, device, ops)
    if case.runner == "per_token":
        return benchmark_representative_per_token_cast(ops, device)
    if case.runner == "per_block":
        return benchmark_representative_per_block_cast(ops, device)
    if case.runner == "swiglu":
        return benchmark_representative_swiglu(ops, device)
    if case.runner == "topk_gate":
        return benchmark_representative_topk_gate(ops, device)
    if case.runner == "topk_group":
        return benchmark_representative_topk_group(ops, device, *case.args)
    if case.runner == "expand":
        return benchmark_representative_expand(ops, device)
    if case.runner == "reduce":
        return benchmark_representative_reduce(ops, device)
    if case.runner == "mhc_rep":
        return benchmark_representative_mhc(*case.args, device, ops)
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
    else:
        aggregate["bandwidth_gbs"] = statistics.median(record["bandwidth_gbs"] for record in sample_records)
    return aggregate


def run_case_samples(case: CaseSpec, ops, device: str, samples: int) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    sample_records = []
    for sample_index in range(1, samples + 1):
        record = run_case(case, ops, device)
        sample_records.append(record)
        if samples > 1:
            print(f"  {style(f'sample {sample_index}/{samples}', TermStyle.dim)} time={record['time_us']:.2f} us")
    return aggregate_sample_records(sample_records), sample_records


def parse_args(
    default_case_names: list[str] | None,
    description: str,
    argv: list[str] | None = None,
) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument(
        "--baseline",
        default=str(benchmark_root() / BASELINE_FILENAME),
        help="Path to the JSONL baseline file.",
    )
    parser.add_argument("--output", help="Optional JSONL output path for current benchmark records.")
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Allowed slowdown ratio margin. 0.05 means current/baseline <= 1.05.",
    )
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
    ops = import_ops()
    runtime_device = ops.get_runtime_device_type()
    device = runtime_device if runtime_device in {"musa", "cuda"} else get_test_device()
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

    cases = [case_map[name] for name in selected_names]
    if not cases:
        raise ValueError("no benchmark cases selected")

    build_type_detail = build_type or "unknown"
    if build_type_source is not None:
        build_type_detail = f"{build_type_detail} ({build_type_source})"
    print_banner(
        title,
        (f"device={device} cases={len(cases)} samples={args.samples} threshold={args.threshold:.4f} build_type={build_type_detail}"),
    )

    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        print(f"{style(f'[{index:02d}/{len(cases):02d}]', TermStyle.bold, TermStyle.gray)} {style(case.name, TermStyle.bold)}")
        record, sample_records = run_case_samples(case, ops, device, args.samples)
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
        exit_code = 1 if regression_stats["failures"] else 0

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
