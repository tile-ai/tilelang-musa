from __future__ import annotations

import argparse
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


@dataclass(frozen=True)
class SourceRun:
    name: str
    records: list[dict[str, Any]]
    regression_stats: dict[str, Any] | None
    exit_code: int


def _ensure_repo_root_on_path() -> None:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists():
            root_text = str(candidate)
            if sys.path[0] != root_text:
                try:
                    sys.path.remove(root_text)
                except ValueError:
                    pass
                sys.path.insert(0, root_text)
            return
    raise RuntimeError(f"unable to determine repository root from {current}")


def _format_duration(seconds: float) -> str:
    if seconds < 60.0:
        return f"{seconds:.2f}s"
    minutes, remaining_seconds = divmod(seconds, 60.0)
    if minutes < 60.0:
        return f"{int(minutes)}m {remaining_seconds:.2f}s"
    hours, remaining_minutes = divmod(minutes, 60.0)
    return f"{int(hours)}h {int(remaining_minutes)}m {remaining_seconds:.2f}s"


def _combined_regression_stats(source_runs: list[SourceRun]) -> dict[str, Any] | None:
    stats = [run.regression_stats for run in source_runs if run.regression_stats is not None]
    if not stats:
        return None
    return {
        "failures": sum(int(item["failures"]) for item in stats),
        "passed": sum(int(item["passed"]) for item in stats),
        "missing": sum(int(item["missing"]) for item in stats),
        "max_ratio": max(float(item["max_ratio"]) for item in stats),
        "threshold": max(float(item["threshold"]) for item in stats),
    }


def _print_summary(source_runs: list[SourceRun], elapsed_seconds: float) -> None:
    total_cases = sum(len(run.records) for run in source_runs)
    regression_stats = _combined_regression_stats(source_runs)
    status = "[PASS]" if all(run.exit_code == 0 for run in source_runs) else "[FAIL]"

    print("=" * 80)
    print("MP31 Summary")
    print("=" * 80)
    print(f"  status         {status}")
    print(f"  sources        {len(source_runs)}")
    print(f"  cases          {total_cases}")
    print(f"  total_time     {_format_duration(elapsed_seconds)} ({elapsed_seconds:.2f}s)")
    for run in source_runs:
        source_status = "[PASS]" if run.exit_code == 0 else "[FAIL]"
        print(f"  {run.name:<14} {source_status} cases={len(run.records)}")

    if regression_stats is None:
        return

    print(f"  passed         {regression_stats['passed']}")
    print(f"  failed         {regression_stats['failures']}")
    print(f"  missing        {regression_stats['missing']}")
    print(f"  max_ratio      {regression_stats['max_ratio']:.4f}")
    print(f"  threshold      {1.0 + regression_stats['threshold']:.4f}")


def main() -> int:
    _ensure_repo_root_on_path()

    parser = argparse.ArgumentParser(description="Run all MP31 benchmarks from TileKernels and MATE sources.")
    parser.add_argument(
        "--source",
        choices=("all", "tilekernels", "mate"),
        default="all",
        help="Benchmark source to run.",
    )
    parser.add_argument(
        "--check-regression",
        action="store_true",
        help="Compare current records against each selected source's baseline file.",
    )
    args, benchmark_args = parser.parse_known_args()

    forwarded = list(benchmark_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.check_regression:
        forwarded = ["--check-regression", *forwarded]

    source_runs: list[SourceRun] = []
    start_time = time.perf_counter()

    if args.source in {"all", "tilekernels"}:
        from benchmark.mp31.tilekernels.benchmark_cases import run_cases as run_tilekernels_cases

        result = run_tilekernels_cases(
            title="TileKernels Benchmark",
            default_case_names=None,
            description="Run MP31 TileKernels benchmarks.",
            argv=forwarded,
            print_final_summary=False,
        )
        source_runs.append(
            SourceRun(
                name="tilekernels",
                records=result.records,
                regression_stats=result.regression_stats,
                exit_code=result.exit_code,
            )
        )

    if args.source in {"all", "mate"}:
        from benchmark.mp31.mate.benchmark_cases import default_case_names, run_cases as run_mate_cases

        result = run_mate_cases(
            title="MATE Benchmark",
            default_case_names=default_case_names(),
            description="Run MP31 MATE-origin benchmarks.",
            argv=forwarded,
            print_final_summary=False,
        )
        source_runs.append(
            SourceRun(
                name="mate",
                records=result.records,
                regression_stats=result.regression_stats,
                exit_code=result.exit_code,
            )
        )

    elapsed_seconds = time.perf_counter() - start_time
    _print_summary(source_runs, elapsed_seconds)
    return 1 if any(run.exit_code != 0 for run in source_runs) else 0


if __name__ == "__main__":
    raise SystemExit(main())
