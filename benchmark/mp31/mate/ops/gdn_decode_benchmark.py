from __future__ import annotations

import sys
from pathlib import Path

benchmark_root = Path(__file__).resolve().parents[1]
mp31_root = benchmark_root.parent
if str(mp31_root) not in sys.path:
    sys.path.insert(0, str(mp31_root))

from mate.benchmark_cases import build_gdn_decode_cases, run_cases_main


def main() -> int:
    return run_cases_main(
        title="MATE GDN Decode Benchmark",
        default_case_names=[case.name for case in build_gdn_decode_cases()],
        description="Run the standalone GDN decode benchmark migrated from MATE.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
