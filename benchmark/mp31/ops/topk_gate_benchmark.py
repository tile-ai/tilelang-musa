from __future__ import annotations

import sys
from pathlib import Path

benchmark_root = Path(__file__).resolve().parents[1]
if str(benchmark_root) not in sys.path:
    sys.path.insert(0, str(benchmark_root))

from ops.benchmark_cases import run_cases_main


def main() -> int:
    return run_cases_main(
        title="TopK Gate Benchmark",
        default_case_names=["representative_topk_gate"],
        description="Run the standalone top-k gate benchmark.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
