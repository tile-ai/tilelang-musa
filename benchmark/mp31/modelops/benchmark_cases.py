from __future__ import annotations

import argparse
import copy
import json
import statistics
import subprocess
import sys
import tempfile
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
    check_regression,
    detect_tilelang_musa_build_type,
    ensure_release_build,
    get_test_device,
    load_baselines,
    print_banner,
    print_json_record,
    print_perf,
    print_sample_stats,
    print_summary,
    style,
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


def build_cases() -> list[CaseSpec]:
    return [
        CaseSpec("example_mha_fwd_bhsd", "example_mha", (2, 28, 8192, 8192, 128, False)),
        CaseSpec("sparse_mla_fwd_pipelined_v1", "sparse_mla_fwd_v1", (896, 4096, 128, 1, 576, 512, 2048, 512)),
        CaseSpec("sparse_mla_fwd_pipelined_v2", "sparse_mla_fwd_v2", (896, 4096, 128, 1, 576, 512, 2048, 640)),
        CaseSpec("sparse_mla_decode_fwd_pipelined_v1", "sparse_mla_decode_v1", (256, 2, 8192, 128, 1, 576, 512, 2048, 512)),
        CaseSpec("sparse_mla_decode_fwd_pipelined_v2", "sparse_mla_decode_v2", (1, 896, 16384, 128, 1, 576, 512, 2048, 640)),
    ]


def default_case_names() -> list[str]:
    return [case.name for case in build_cases()]


def build_case_map() -> dict[str, CaseSpec]:
    return {case.name: case for case in build_cases()}


def _require_record(record: dict[str, Any] | None, case: CaseSpec) -> dict[str, Any]:
    if record is None:
        raise RuntimeError(f"benchmark case did not return a record: {case.name}")
    return record


def run_case(case: CaseSpec, device: str) -> dict[str, Any]:
    if case.runner == "example_mha":
        from .kernels import example_mha_fwd_bhsd

        batch, heads, seq_q, seq_kv, dim, is_causal = case.args
        example_mha_fwd_bhsd.DEVICE = device
        example_mha_fwd_bhsd.TARGET = device
        return _require_record(
            example_mha_fwd_bhsd.main(
                batch=batch,
                heads=heads,
                seq_q=seq_q,
                seq_kv=seq_kv,
                dim=dim,
                is_causal=is_causal,
                tune=False,
                verbose=False,
            ),
            case,
        )
    if case.runner == "sparse_mla_fwd_v1":
        from .kernels import sparse_mla_fwd_pipelined_v1

        seq_len, seq_len_kv, heads, kv_heads, d_qk, d_v, topk, threads = case.args
        return _require_record(
            sparse_mla_fwd_pipelined_v1.test_sparse_mla_fwd_v1(
                S=seq_len,
                SKV=seq_len_kv,
                H=heads,
                HKV=kv_heads,
                DQK=d_qk,
                DV=d_v,
                topk=topk,
                dtype=torch.bfloat16,
                check_correctness=True,
                perf_test=True,
                threads=threads,
            ),
            case,
        )
    if case.runner == "sparse_mla_fwd_v2":
        from .kernels import sparse_mla_fwd_pipelined_v2

        seq_len, seq_len_kv, heads, kv_heads, d_qk, d_v, topk, threads = case.args
        return _require_record(
            sparse_mla_fwd_pipelined_v2.test_sparse_mla_fwd_v2(
                S=seq_len,
                SKV=seq_len_kv,
                H=heads,
                HKV=kv_heads,
                DQK=d_qk,
                DV=d_v,
                topk=topk,
                dtype=torch.bfloat16,
                check_correctness=True,
                perf_test=True,
                threads=threads,
            ),
            case,
        )
    if case.runner == "sparse_mla_decode_v1":
        from .kernels import sparse_mla_decode_fwd_pipelined_v1

        batch, seq_len, seq_len_kv, heads, kv_heads, d_qk, d_v, topk, threads = case.args
        return _require_record(
            sparse_mla_decode_fwd_pipelined_v1.test_sparse_mla_fwd(
                B=batch,
                S=seq_len,
                SKV=seq_len_kv,
                H=heads,
                HKV=kv_heads,
                DQK=d_qk,
                DV=d_v,
                topk=topk,
                dtype=torch.bfloat16,
                check_correctness=True,
                perf_test=True,
                threads=threads,
            ),
            case,
        )
    if case.runner == "sparse_mla_decode_v2":
        from .kernels import sparse_mla_decode_fwd_pipelined_v2

        batch, seq_len, seq_len_kv, heads, kv_heads, d_qk, d_v, topk, threads = case.args
        return _require_record(
            sparse_mla_decode_fwd_pipelined_v2.test_sparse_mla_fwd(
                B=batch,
                S=seq_len,
                SKV=seq_len_kv,
                H=heads,
                HKV=kv_heads,
                DQK=d_qk,
                DV=d_v,
                topk=topk,
                dtype=torch.bfloat16,
                check_correctness=True,
                perf_test=True,
                threads=threads,
            ),
            case,
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
        "--check-regression",
        action="store_true",
        help="Compare current records against the baseline file.",
    )
    parser.add_argument(
        "--threshold",
        type=float,
        default=DEFAULT_THRESHOLD,
        help="Allowed slowdown ratio margin. 0.05 means current/baseline <= 1.05.",
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


def _select_cases(args: argparse.Namespace, default_case_names: list[str] | None) -> list[CaseSpec]:
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
    return cases


def run_cases(
    *,
    title: str,
    default_case_names: list[str] | None = None,
    description: str,
    argv: list[str] | None = None,
    print_final_summary: bool = True,
    print_run_header: bool = True,
    print_case_header: bool = True,
) -> BenchmarkRunResult:
    start_time = time.perf_counter()
    args = parse_args(default_case_names, description, argv)
    if args.samples < 1:
        raise ValueError("--samples must be >= 1")

    ensure_release_build(strict=not args.allow_non_release_build)
    device = get_test_device()
    if device != "musa":
        raise RuntimeError("MUSA is required for mp31/modelops")
    build_type, build_type_source = detect_tilelang_musa_build_type()

    cases = _select_cases(args, default_case_names)

    build_type_detail = build_type or "unknown"
    if build_type_source is not None:
        build_type_detail = f"{build_type_detail} ({build_type_source})"
    if print_run_header:
        print_banner(
            title,
            f"source=modelops device={device} cases={len(cases)} samples={args.samples} "
            f"threshold={args.threshold:.4f} build_type={build_type_detail}",
        )

    records: list[dict[str, Any]] = []
    for index, case in enumerate(cases, start=1):
        if print_case_header:
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


def _case_runner_path() -> Path:
    return Path(__file__).resolve().with_name("case_runner.py")


def _load_output_records(output_path: Path) -> list[dict[str, Any]]:
    with output_path.open() as f:
        return [json.loads(line) for line in f if line.strip()]


def _run_isolated_case(
    *,
    case: CaseSpec,
    args: argparse.Namespace,
    output_path: Path,
) -> int:
    command = [
        sys.executable,
        str(_case_runner_path()),
        "--cases",
        case.name,
        "--samples",
        str(args.samples),
        "--output",
        str(output_path),
    ]
    if args.allow_non_release_build:
        command.append("--allow-non-release-build")

    completed = subprocess.Popen(
        command,
        cwd=benchmark_root().parent.parent,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    assert completed.stdout is not None
    for line in completed.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
    returncode = completed.wait()
    completed.stdout.close()
    return returncode


def run_cases_isolated(
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
    if device != "musa":
        raise RuntimeError("MUSA is required for mp31/modelops")
    build_type, build_type_source = detect_tilelang_musa_build_type()
    cases = _select_cases(args, default_case_names)

    build_type_detail = build_type or "unknown"
    if build_type_source is not None:
        build_type_detail = f"{build_type_detail} ({build_type_source})"
    print_banner(
        title,
        f"source=modelops device={device} cases={len(cases)} samples={args.samples} "
        f"threshold={args.threshold:.4f} build_type={build_type_detail} isolated=true",
    )

    records: list[dict[str, Any]] = []
    exit_code = 0
    with tempfile.TemporaryDirectory(prefix="modelops-bench-") as temp_dir:
        temp_root = Path(temp_dir)
        for index, case in enumerate(cases, start=1):
            print(f"{style(f'[{index:02d}/{len(cases):02d}]', TermStyle.bold, TermStyle.gray)} {style(case.name, TermStyle.bold)}")
            output_path = temp_root / f"{case.name}.jsonl"
            returncode = _run_isolated_case(case=case, args=args, output_path=output_path)
            if returncode != 0:
                exit_code = 1
                print(f"{style('[FAIL]', TermStyle.bold, TermStyle.red)} isolated case failed: {case.name}")
                continue
            case_records = _load_output_records(output_path)
            if len(case_records) != 1:
                exit_code = 1
                print(f"{style('[FAIL]', TermStyle.bold, TermStyle.red)} expected one record for {case.name}, got {len(case_records)}")
                continue
            records.extend(case_records)

    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w") as f:
            for record in records:
                f.write(json.dumps(record, sort_keys=True) + "\n")
        print(f"{style('[WRITE]', TermStyle.bold, TermStyle.blue)} saved {len(records)} records to {output_path}")

    regression_stats: dict[str, Any] | None = None
    if args.check_regression and records:
        print_banner("Regression Check", f"baseline={args.baseline}")
        regression_stats = check_regression(records, load_baselines(Path(args.baseline)), args.threshold)
        if regression_stats["failures"] or regression_stats["missing"]:
            exit_code = 1

    if print_final_summary:
        print_summary(len(records), regression_stats, time.perf_counter() - start_time)
        if regression_stats is None and exit_code == 0:
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
