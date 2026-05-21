from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any

import torch


BASELINE_FILENAME = "baselines/mate.jsonl"
DEFAULT_THRESHOLD = 0.05


class TermStyle:
    reset = "\033[0m"
    bold = "\033[1m"
    dim = "\033[2m"
    red = "\033[31m"
    green = "\033[32m"
    yellow = "\033[33m"
    blue = "\033[34m"
    magenta = "\033[35m"
    cyan = "\033[36m"
    gray = "\033[90m"


def benchmark_root() -> Path:
    return Path(__file__).resolve().parent


def repo_root() -> Path:
    current = benchmark_root()
    for candidate in (current, *current.parents):
        if (candidate / ".git").exists():
            return candidate
    raise RuntimeError(f"unable to determine repository root from {current}")


def ensure_repo_root_on_path() -> Path:
    root = repo_root()
    root_text = str(root)
    if sys.path[0] != root_text:
        try:
            sys.path.remove(root_text)
        except ValueError:
            pass
        sys.path.insert(0, root_text)
    return root


ensure_repo_root_on_path()


def use_color() -> bool:
    if os.getenv("NO_COLOR"):
        return False
    return sys.stdout.isatty()


def style(text: str, *styles: str) -> str:
    if not use_color() or not styles:
        return text
    return f"{''.join(styles)}{text}{TermStyle.reset}"


def print_banner(title: str, detail: str | None = None) -> None:
    line = style("=" * 80, TermStyle.gray)
    print(line)
    print(style(title, TermStyle.bold, TermStyle.cyan))
    if detail:
        print(style(detail, TermStyle.dim))
    print(line)


def get_test_device() -> str:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa"
    if torch.cuda.is_available():
        return "cuda"
    raise RuntimeError("Neither MUSA nor CUDA is available")


def count_bytes(*tensors: torch.Tensor | None) -> int:
    total = 0
    for tensor in tensors:
        if tensor is not None:
            total += tensor.numel() * tensor.element_size()
    return total


def benchmark_timer(fn, warmup: int = 25, rep: int = 100) -> float:
    from tilelang.profiler import do_bench

    ms = do_bench(fn, warmup=warmup, rep=rep)
    return ms * 1e3


def format_params(params: dict[str, Any]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def ratio_style(ratio: float, threshold: float) -> str:
    if ratio > 1.0 + threshold:
        return TermStyle.red
    if ratio > 1.0:
        return TermStyle.yellow
    return TermStyle.green


def print_perf(record: dict[str, Any]) -> None:
    extras = record.get("extras", {})
    operation = f"({record['operation']})"
    time_text = f"{record['time_us']:.2f} us"
    print(
        f"{style('[PERF]', TermStyle.bold, TermStyle.blue)} "
        f"{style(record['kernel'], TermStyle.bold)} "
        f"{style(operation, TermStyle.dim)}"
    )
    print(f"  {style('time', TermStyle.cyan):<14} {style(time_text, TermStyle.bold)}")
    if "tflops" in extras:
        print(f"  {style('tflops', TermStyle.cyan):<14} {extras['tflops']:.4f}")
    print(f"  {style('bandwidth', TermStyle.cyan):<14} {record['bandwidth_gbs']:.4f} GB/s")
    if "bytes_rw" in extras:
        print(f"  {style('bytes_rw', TermStyle.cyan):<14} {extras['bytes_rw']}")
    print(f"  {style('params', TermStyle.cyan):<14} {format_params(record['params'])}")


def print_json_record(record: dict[str, Any]) -> None:
    print(
        f"{style('[JSON]', TermStyle.bold, TermStyle.magenta)} "
        f"{style(json.dumps(record, sort_keys=True), TermStyle.dim)}"
    )


def record_key(record: dict[str, Any]) -> tuple[str, str, str]:
    params_json = json.dumps(record["params"], sort_keys=True, separators=(",", ":"))
    return record["kernel"], record["operation"], params_json


def load_baselines(path: Path) -> dict[tuple[str, str, str], dict[str, Any]]:
    baselines: dict[tuple[str, str, str], dict[str, Any]] = {}
    with path.open() as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            record = json.loads(line)
            baselines[record_key(record)] = record
    return baselines


def print_summary(
    total_cases: int,
    regression_stats: dict[str, Any] | None = None,
) -> None:
    print_banner("Summary")
    print(f"  {style('cases', TermStyle.cyan):<14} {style(str(total_cases), TermStyle.bold)}")
    if regression_stats is None:
        return

    failures = int(regression_stats["failures"])
    missing = int(regression_stats["missing"])
    passed = int(regression_stats["passed"])
    max_ratio = float(regression_stats["max_ratio"])
    threshold = float(regression_stats["threshold"])
    status = "[PASS]" if failures == 0 else "[FAIL]"
    status_style = TermStyle.green if failures == 0 else TermStyle.red
    print(f"  {style('regression', TermStyle.cyan):<14} {style(status, TermStyle.bold, status_style)}")
    print(f"  {style('passed', TermStyle.cyan):<14} {style(str(passed), TermStyle.green)}")
    print(f"  {style('failed', TermStyle.cyan):<14} {style(str(failures), TermStyle.red if failures else TermStyle.green)}")
    print(f"  {style('missing', TermStyle.cyan):<14} {style(str(missing), TermStyle.yellow if missing else TermStyle.green)}")
    print(f"  {style('max_ratio', TermStyle.cyan):<14} {style(f'{max_ratio:.4f}', ratio_style(max_ratio, threshold), TermStyle.bold)}")
    print(f"  {style('threshold', TermStyle.cyan):<14} {1.0 + threshold:.4f}")


def check_regression(
    records: list[dict[str, Any]],
    baselines: dict[tuple[str, str, str], dict[str, Any]],
    threshold: float,
) -> dict[str, Any]:
    failures = 0
    passed = 0
    missing = 0
    max_ratio = 0.0
    for record in records:
        baseline = baselines.get(record_key(record))
        if baseline is None:
            missing += 1
            print(
                f"{style('[WARN]', TermStyle.bold, TermStyle.yellow)} "
                f"missing baseline for {record['kernel']} {format_params(record['params'])}"
            )
            continue
        ratio = record["time_us"] / baseline["time_us"]
        max_ratio = max(max_ratio, ratio)
        ratio_text = style(f"{ratio:.4f}", ratio_style(ratio, threshold), TermStyle.bold)
        op_text = f"({record['operation']})"
        case_failed = ratio > 1.0 + threshold
        status = "[FAIL]" if case_failed else "[PASS]"
        status_style = TermStyle.red if case_failed else TermStyle.green
        print(
            f"{style(status, TermStyle.bold, status_style)} "
            f"{record['kernel']} {style(op_text, TermStyle.dim)}: "
            f"current={record['time_us']:.2f} us, "
            f"baseline={baseline['time_us']:.2f} us, "
            f"ratio={ratio_text}"
        )
        if case_failed:
            failures += 1
        else:
            passed += 1
    return {
        "failures": failures,
        "passed": passed,
        "missing": missing,
        "max_ratio": max_ratio,
        "threshold": threshold,
    }


def _parse_cmake_build_type(cache_path: Path) -> str | None:
    if not cache_path.is_file():
        return None
    with cache_path.open() as f:
        for line in f:
            if line.startswith("CMAKE_BUILD_TYPE:STRING="):
                value = line.strip().split("=", 1)[1]
                return value or None
    return None


def detect_tilelang_musa_build_type() -> tuple[str | None, Path | None]:
    candidates: list[Path] = []
    try:
        import tilelang

        lib_path = Path(tilelang._LIB_PATH).resolve()
        candidates.extend(
            [
                lib_path.parents[2] / "CMakeCache.txt",
                lib_path.parents[1] / "CMakeCache.txt",
            ]
        )
    except Exception:
        pass

    candidates.append(repo_root() / "build" / "CMakeCache.txt")

    seen: set[Path] = set()
    for candidate in candidates:
        candidate = candidate.resolve()
        if candidate in seen:
            continue
        seen.add(candidate)
        build_type = _parse_cmake_build_type(candidate)
        if build_type is not None:
            return build_type, candidate
    return None, None


def ensure_release_build(strict: bool = True) -> tuple[str | None, Path | None]:
    build_type, source = detect_tilelang_musa_build_type()
    if not strict:
        return build_type, source
    if build_type != "Release":
        location = f" ({source})" if source is not None else ""
        detected = build_type if build_type is not None else "unknown"
        raise RuntimeError(
            "Benchmarks require tilelang-musa to be compiled with "
            f"CMAKE_BUILD_TYPE=Release; detected {detected}{location}. "
            "Use --allow-non-release-build to bypass this check."
        )
    return build_type, source
