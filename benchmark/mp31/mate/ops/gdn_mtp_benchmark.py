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

    from benchmark.mp31.mate.benchmark_cases import gdn_mtp_case_names, run_cases_main

    return run_cases_main(
        title="MATE GDN MTP Benchmark",
        default_case_names=gdn_mtp_case_names(),
        description="Run the standalone GDN MTP benchmark migrated from MATE.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
