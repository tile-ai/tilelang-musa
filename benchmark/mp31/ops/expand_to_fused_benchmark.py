from __future__ import annotations

import sys
from pathlib import Path

benchmark_root = Path(__file__).resolve().parents[1]
if str(benchmark_root) not in sys.path:
    sys.path.insert(0, str(benchmark_root))

from ops.benchmark_cases import run_cases_main


def main() -> int:
    return run_cases_main(
        title="Expand-To-Fused Benchmark",
        default_case_names=["representative_expand_to_fused"],
        description="Run the standalone expand-to-fused benchmark.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
