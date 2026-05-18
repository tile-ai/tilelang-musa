from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    benchmark_root = Path(__file__).resolve().parent
    if str(benchmark_root) not in sys.path:
        sys.path.insert(0, str(benchmark_root))

    from ops.benchmark_cases import run_cases_main

    return run_cases_main(
        title="TileLang MUSA Representative Benchmark",
        default_case_names=None,
        description="Run the full MP31 representative benchmark suite.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
