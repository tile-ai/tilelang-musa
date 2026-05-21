from __future__ import annotations

import sys
from pathlib import Path

benchmark_root = Path(__file__).resolve().parents[1]
mp31_root = benchmark_root.parent
if str(mp31_root) not in sys.path:
    sys.path.insert(0, str(mp31_root))

from mate.benchmark_cases import run_cases_main


def main() -> int:
    return run_cases_main(
        title="MATE Sparse MLA Decode Direct TileLang Benchmark",
        default_case_names=[
            "sparse_mla_decode_v32_temp_aligned_bf16",
        ],
        description="Run direct TileLang Sparse MLA decode benchmarks migrated from MATE.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
