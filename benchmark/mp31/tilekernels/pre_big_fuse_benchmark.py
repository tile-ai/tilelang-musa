from __future__ import annotations

import sys
from pathlib import Path


def _ensure_repo_root_on_path() -> None:
    current = Path(__file__).resolve()
    for candidate in current.parents:
        if (candidate / ".git").exists():
            root_text = str(candidate)
            if root_text not in sys.path:
                sys.path.insert(0, root_text)
            return
    raise RuntimeError(f"unable to determine repository root from {current}")


def main() -> int:
    _ensure_repo_root_on_path()

    from benchmark.mp31.tilekernels.benchmark_cases import run_cases_main

    return run_cases_main(
        title="MHC Pre-Big-Fuse Benchmark",
        default_case_names=[
            "mhc_pre_big_fuse_16_4096",
            "mhc_pre_big_fuse_512_1280",
            "mhc_pre_big_fuse_2048_4096",
            "mhc_pre_big_fuse_8192_4096",
            "representative_mhc_pre_big_fuse_2048_4096",
            "representative_mhc_pre_big_fuse_8192_4096",
        ],
        description="Run the standalone MHC pre-big-fuse benchmarks.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
