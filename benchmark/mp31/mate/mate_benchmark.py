from __future__ import annotations

import sys
from pathlib import Path


def main() -> int:
    benchmark_root = Path(__file__).resolve().parent
    mp31_root = benchmark_root.parent
    if str(mp31_root) not in sys.path:
        sys.path.insert(0, str(mp31_root))

    from mate.benchmark_cases import default_case_names, run_cases_main

    return run_cases_main(
        title="TileLang MUSA MATE-Origin Benchmark",
        default_case_names=default_case_names(),
        description="Run the MP31 TileLang benchmarks migrated from MATE.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
