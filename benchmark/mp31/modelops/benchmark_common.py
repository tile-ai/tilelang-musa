from __future__ import annotations

from benchmark.mp31.tilekernels.benchmark_common import (
    DEFAULT_THRESHOLD,
    TermStyle,
    benchmark_root,
    check_regression,
    detect_tilelang_musa_build_type,
    ensure_release_build,
    get_test_device,
    load_baselines,
    print_banner,
    print_json_record,
    print_perf,
    print_sample_stats,
    print_summary,
    style,
)


BASELINE_FILENAME = "baselines/modelops.jsonl"
