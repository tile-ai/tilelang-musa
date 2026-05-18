from __future__ import annotations

import sys
from pathlib import Path

benchmark_root = Path(__file__).resolve().parents[1]
if str(benchmark_root) not in sys.path:
    sys.path.insert(0, str(benchmark_root))

from ops.benchmark_cases import run_cases_main


def main() -> int:
    return run_cases_main(
        title="TopK Group Benchmark",
        default_case_names=[
            "representative_topk_group_72_12",
            "representative_topk_group_256_8",
            "representative_topk_group_256_16",
        ],
        description="Run the standalone top-k sum and top-k group index benchmarks.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
