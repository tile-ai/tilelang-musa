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

    from benchmark.mp31.tilekernels.benchmark_cases import run_cases_main

    return run_cases_main(
        title="TopK Gate Benchmark",
        default_case_names=["representative_topk_gate"],
        description="Run the standalone top-k gate benchmark.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
