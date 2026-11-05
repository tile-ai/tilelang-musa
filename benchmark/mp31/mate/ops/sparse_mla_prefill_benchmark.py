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

    from benchmark.mp31.mate.benchmark_cases import run_cases_main

    return run_cases_main(
        title="MATE Sparse MLA Prefill Direct TileLang Benchmark",
        default_case_names=[
            "sparse_mla_prefill_v32_temp_aligned_bf16",
            "sparse_mla_prefill_model1_small_extra_bf16",
            "sparse_mla_prefill_model1_extra_bf16",
        ],
        description="Run direct TileLang Sparse MLA prefill benchmarks migrated from MATE.",
    )


if __name__ == "__main__":
    raise SystemExit(main())
