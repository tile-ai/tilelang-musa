from __future__ import annotations

import argparse
import json
import os
import re
import shlex
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path

import pytest


@dataclass(frozen=True)
class RunnerResult:
    returncode: int
    failed_cases: frozenset[str]


_RUNNER_RESULT: RunnerResult | None = None


def _raw_runner_args() -> list[str]:
    return shlex.split(os.environ.get("MP31_RUNNER_ARGS", "--allow-non-release-build"))


def _split_runner_args() -> tuple[str, bool, set[str] | None]:
    args = _raw_runner_args()

    source_parser = argparse.ArgumentParser(add_help=False)
    source_parser.add_argument("--source", choices=("all", "tilekernels", "mate"), default="all")
    source_args, remaining = source_parser.parse_known_args(args)

    regression_parser = argparse.ArgumentParser(add_help=False)
    regression_parser.add_argument("--check-regression", action="store_true")
    regression_args, remaining = regression_parser.parse_known_args(remaining)

    case_parser = argparse.ArgumentParser(add_help=False)
    case_parser.add_argument("--cases", nargs="*")
    case_args, _ = case_parser.parse_known_args(remaining)

    selected_cases = set(case_args.cases) if case_args.cases else None
    return source_args.source, regression_args.check_regression, selected_cases


def _case_names() -> list[tuple[str, str]]:
    source, check_regression, selected_cases = _split_runner_args()
    cases: list[tuple[str, str]] = []

    if source in {"all", "tilekernels"}:
        from benchmark.mp31.tilekernels.benchmark_cases import build_cases

        cases.extend(("tilekernels", case.name) for case in build_cases())

    if source in {"all", "mate"}:
        from benchmark.mp31.mate.benchmark_cases import default_case_names, regression_supported_case_names

        mate_case_names = regression_supported_case_names() if check_regression and selected_cases is None else default_case_names()
        cases.extend(("mate", name) for name in mate_case_names)

    if selected_cases is not None:
        known_cases = {case_name for _, case_name in cases}
        unknown_cases = selected_cases - known_cases
        if unknown_cases:
            raise ValueError(f"unknown benchmark case(s): {', '.join(sorted(unknown_cases))}")
        cases = [(case_source, case_name) for case_source, case_name in cases if case_name in selected_cases]

    return cases


def _parse_failed_cases(output: str) -> frozenset[str]:
    failed_cases: set[str] = set()
    current_case: str | None = None
    label_to_case: dict[str, str] = {}
    case_line_pattern = re.compile(r"^\[\d+/\d+\]\s+(.+)$")

    for line in output.splitlines():
        case_match = case_line_pattern.match(line)
        if case_match:
            current_case = case_match.group(1).strip()
            continue
        if line.startswith("[JSON]") and current_case is not None:
            json_start = line.find("{")
            if json_start >= 0:
                record = json.loads(line[json_start:])
                label_to_case[_format_record_label(record)] = current_case
            continue
        if line.startswith("[FAIL]"):
            label = _failed_record_label(line)
            if label is not None and label in label_to_case:
                failed_cases.add(label_to_case[label])
            elif current_case is not None:
                failed_cases.add(current_case)
        elif line.startswith("[WARN] missing baseline for "):
            label = line.removeprefix("[WARN] missing baseline for ")
            if label in label_to_case:
                failed_cases.add(label_to_case[label])

    return frozenset(failed_cases)


def _format_params(params: dict[str, object]) -> str:
    return ", ".join(f"{key}={value}" for key, value in sorted(params.items()))


def _format_record_label(record: dict[str, object]) -> str:
    return f"{record['kernel']} ({record['operation']}) [{_format_params(record['params'])}]"


def _failed_record_label(line: str) -> str | None:
    prefix = "[FAIL] "
    if not line.startswith(prefix):
        return None
    detail = line[len(prefix) :]
    marker = ": current="
    if marker not in detail:
        return None
    return detail.split(marker, 1)[0]


def _run_runner_once() -> RunnerResult:
    global _RUNNER_RESULT
    if _RUNNER_RESULT is not None:
        return _RUNNER_RESULT

    test_path = Path(__file__).resolve()
    repo_root = test_path.parents[2]
    runner_path = test_path.with_name("runner.py")
    command = [sys.executable, str(runner_path), *_raw_runner_args()]

    completed = subprocess.Popen(
        command,
        cwd=repo_root,
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    output_lines: list[str] = []
    assert completed.stdout is not None
    for line in completed.stdout:
        sys.stdout.write(line)
        sys.stdout.flush()
        output_lines.append(line)
    returncode = completed.wait()
    completed.stdout.close()
    output = "".join(output_lines)

    _RUNNER_RESULT = RunnerResult(
        returncode=returncode,
        failed_cases=_parse_failed_cases(output),
    )
    return _RUNNER_RESULT


def pytest_generate_tests(metafunc):
    if "mp31_case" not in metafunc.fixturenames:
        return

    cases = _case_names()
    metafunc.parametrize("mp31_case", [pytest.param(case, id=f"{case[0]}::{case[1]}") for case in cases])


def test_mp31_benchmark_case(mp31_case, capfd):
    _, case_name = mp31_case
    with capfd.disabled():
        runner_result = _run_runner_once()

    if runner_result.returncode != 0:
        if runner_result.failed_cases:
            if case_name in runner_result.failed_cases:
                pytest.fail(f"MP31 benchmark case failed: {case_name}")
            return
        pytest.fail("MP31 benchmark runner failed before case-specific failures could be identified")
