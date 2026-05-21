from __future__ import annotations

import sys
from pathlib import Path

benchmark_root = Path(__file__).resolve().parents[1]
if str(benchmark_root) not in sys.path:
    sys.path.insert(0, str(benchmark_root))

from tilekernels.benchmark_cases import run_cases_main


def main() -> int:
    return run_cases_main(
        title="Per-Block Cast Benchmark",
        default_case_names=["representative_per_block_cast"],
        description="Run the standalone per-block cast benchmark.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
