from __future__ import annotations

import sys
from pathlib import Path

benchmark_root = Path(__file__).resolve().parents[1]
if str(benchmark_root) not in sys.path:
    sys.path.insert(0, str(benchmark_root))

from tilekernels.benchmark_cases import run_cases_main


def main() -> int:
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
