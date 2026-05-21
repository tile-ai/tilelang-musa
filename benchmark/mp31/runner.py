from __future__ import annotations

import argparse
import subprocess
import sys
from pathlib import Path


def _run(script: Path, extra_args: list[str]) -> int:
    cmd = [sys.executable, str(script), *extra_args]
    print(f"[RUN] {' '.join(cmd)}")
    repo_root = script
    for candidate in (script.parent, *script.parents):
        if (candidate / ".git").exists():
            repo_root = candidate
            break
    return subprocess.call(cmd, cwd=repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(
        description="Run all MP31 benchmarks from TileKernels and MATE sources."
    )
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

    mp31_root = Path(__file__).resolve().parent
    scripts: list[Path] = []
    if args.source in {"all", "tilekernels"}:
        scripts.append(mp31_root / "tilekernels" / "tilekernels_benchmark.py")
    if args.source in {"all", "mate"}:
        scripts.append(mp31_root / "mate" / "mate_benchmark.py")

    forwarded = list(benchmark_args)
    if forwarded and forwarded[0] == "--":
        forwarded = forwarded[1:]
    if args.check_regression:
        forwarded = ["--check-regression", *forwarded]

    for script in scripts:
        rc = _run(script, forwarded)
        if rc != 0:
            return rc
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
