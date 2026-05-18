# TileLang MUSA MP31 Benchmark Bundle

## Migration Source

This directory carries the 15 TileKernels representative baseline records into
`tilelang-musa` for MP31/S5000-side tracking.

The benchmark runners use only local implementations under
`benchmark/mp31/ops`. They are migrated from tilekernels repository (`git@sh-code.mthreads.com:tianyi.xu/tilekernels.git`
) but do not depend on it.

Baseline source:

- The current `tilekernels_baselines.jsonl` values are **release** benchmark
  measurements from `release_v0.1.8_musa.3`.
- Baseline records keep the compact JSONL schema:
  `kernel`, `operation`, `params`, `time_us`, `bandwidth_gbs`, and
  `extras.bytes_rw`.
- `time_us` is rounded to 2 decimal places and `bandwidth_gbs` is rounded to 4
  decimal places for readability.

Historical source snapshots of tilekernels repository:

- `847f94dfee761c7683e0ff6e56de200cbc95d519` (`2026-05-11`): original
  representative benchmark suite and initial 15-record baseline set.
- `fee56c1aeb1ee7c283b95e939fe8ff753fecd40d` (`2026-05-12`): later optimization
  snapshot used before the release baseline refresh.

## Tool Layout

- `tilekernels_baselines.jsonl`: merged 15-record
  baseline file.
- `tilekernels_benchmark.py`: aggregate benchmark entrypoint for the whole
  15-case suite.
- `ops/benchmark_common.py`, `ops/benchmark_cases.py`: shared benchmark
  framework, case registry, output formatting, and regression checking.
- `ops/*_benchmark.py`: standalone per-operator benchmark entrypoints.
- `ops/quant/`, `ops/moe/`, `ops/mhc/`: local operator implementations used by
  both the aggregate runner and standalone runners.

## Benchmark Guardrails

These benchmarks are intended to be compared against a **release** baseline, so
the runner enforces the same assumption by default:

- `tilelang-musa` must be compiled with `CMAKE_BUILD_TYPE=Release`.
- If the current build is not `Release`, the benchmark exits with an error
  before running any case.
- The runner prints the detected `build_type` in the benchmark banner so the
  active build configuration is visible in logs.
- Use `--allow-non-release-build` only when you intentionally want to bypass
  this check for debugging or local investigation. Results produced in that mode
  should not be used to refresh performance baselines.

## Usage

Aggregate example:

```bash
cd tilelang_musa
python benchmark/mp31/tilekernels_benchmark.py
```

Bypass the release-build check explicitly:

```bash
python benchmark/mp31/tilekernels_benchmark.py \
  --allow-non-release-build
```

Median-sampled baseline refresh:

```bash
python /root/tilelang_musa/benchmark/mp31/tilekernels_benchmark.py \
  --samples 5 \
  --output /tmp/tilekernels_baselines.jsonl
```

`--samples N` runs each case N independent times and uses the median `time_us`
as the emitted `[PERF]`, `[JSON]`, `--output`, and regression-check value.
`bandwidth_gbs` is recomputed from `extras.bytes_rw / median_time_us / 1e3`.
The runner also prints `n`, `median`, `mean`, `min`, and `max` for each case when
`N > 1`; use these to spot unstable cases before accepting a refreshed
baseline.

Standalone examples:

```bash
python /root/tilelang_musa/benchmark/mp31/ops/per_token_cast_benchmark.py

python /root/tilelang_musa/benchmark/mp31/ops/topk_sum_and_topk_group_idx_benchmark.py \
  --check-regression

python /root/tilelang_musa/benchmark/mp31/ops/pre_big_fuse_benchmark.py \
  --cases mhc_pre_big_fuse_2048_4096 representative_mhc_pre_big_fuse_2048_4096
```

CLI Notes:

- All entrypoints support `--output`, `--check-regression`, `--baseline`,
  `--threshold`, `--samples`, `--cases`, and `--allow-non-release-build`.
- For single-op entrypoints, `--cases` can be used to narrow to a subset of the
  cases owned by that operator family.
- Prefer `--samples 5` or higher when refreshing `tilekernels_baselines.jsonl`.
  Median is used instead of mean so an occasional slow run does not shift the
  baseline.

## Workflow

The four components form a complete automated performance detection pipeline, following the steps below:

1. **Run Benchmarks**: Execute all predefined performance test cases to generate real-time current performance metrics.
2. **Compare with Baseline**: Match the newly obtained benchmark results against the stable historical baseline data.
3. **Detect Regression (Performance Degradation):** Identify performance degradation if the current metrics are slower or worse than the baseline.
4. **Trigger Guardrail Check**: Judge the severity of the regression. If the performance decline exceeds the preset threshold (5% slowdown), trigger a check failure to alert and prevent problematic code merging.
