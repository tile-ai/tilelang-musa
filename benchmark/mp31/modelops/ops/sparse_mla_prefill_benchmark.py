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

    from benchmark.mp31.modelops.benchmark_cases import run_cases_main

    return run_cases_main(
        title="ModelOps Sparse MLA Prefill Benchmark",
        default_case_names=[
            "sparse_mla_fwd_pipelined_v1",
            "sparse_mla_fwd_pipelined_v2",
        ],
        description="Run the ModelOps sparse MLA prefill benchmarks.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
