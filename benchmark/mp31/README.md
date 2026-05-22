# TileLang MUSA MP31 Benchmark Bundle

## Migration Source

This directory carries the 15 TileKernels representative baseline records into
`tilelang-musa` for MP31/S5000-side tracking.

The benchmark runners use only local implementations under
`benchmark/mp31/tilekernels`. They are migrated from tilekernels repository (`git@sh-code.mthreads.com:tianyi.xu/tilekernels.git`
) but do not depend on it.

Baseline source:

- The current `baselines/tilekernels.jsonl` and `baselines/mate.jsonl` values
  are **release** benchmark measurements from `release_v0.1.8_musa.3`,
  `c3ed1bd5272c916a042df7877cdc4b587fb1006a`.
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

MATE-origin TileLang benchmarks are also kept in this bundle under
`benchmark/mp31/mate`. These cases are migrated from the MATE repository's
TileLang-backed benchmark paths, but the runners use local kernel copies and
host-side helpers only; they must not import the MATE repository at runtime.

## Tool Layout

- `baselines/tilekernels.jsonl`: merged 15-record TileKernels baseline file.
- `baselines/mate.jsonl`: MATE-origin benchmark baseline file.
- `tilekernels/benchmark_common.py`, `tilekernels/benchmark_cases.py`: shared benchmark
  framework, case registry, output formatting, and regression checking.
- `tilekernels/*_benchmark.py`: standalone per-operator benchmark entrypoints.
- `tilekernels/quant/`, `tilekernels/moe/`, `tilekernels/mhc/`: local operator implementations used by
  both the aggregate runner and standalone runners.
- `mate/ops/*_benchmark.py`: standalone MATE-origin per-operator entrypoints.
- `mate/kernels/`: local TileLang kernels and minimal host-side helpers
  migrated from MATE. These files intentionally avoid MATE package imports.
- `runner.py`: aggregate entrypoint that can execute TileKernels, MATE-origin,
  or both benchmark groups and print one combined summary.

## AI Coding Guide

When using an AI coding agent to modify benchmark code, ask it to read
`benchmark/AGENTS.md` before making changes. That file is the benchmark-local
agent guide: it summarizes the MP31 source layout, important migration history,
baseline rules, common commands, and constraints that are easy to miss when an
agent only inspects one file at a time.

Recommended prompt pattern:

```text
Please read benchmark/AGENTS.md first, then update the MP31 benchmark case ...
```

Use this README as the human-facing command reference and migration note. Use
`benchmark/AGENTS.md` as the AI-facing working contract for coding tasks under
`benchmark/`. If benchmark commands, default cases, baseline provenance, source
layout, or migration constraints change, update both files when the change
affects both humans and agents.

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

Aggregate examples:

```bash
cd tilelang_musa
python benchmark/mp31/runner.py --source tilekernels
```

Bypass the release-build check explicitly:

```bash
python benchmark/mp31/runner.py --source tilekernels \
  --allow-non-release-build
```

Median-sampled baseline refresh:

```bash
python /root/tilelang_musa/benchmark/mp31/runner.py \
  --source tilekernels \
  --samples 5 \
  --output /tmp/tilekernels.jsonl
```

`--samples N` runs each case N independent times and uses the median `time_us`
as the emitted `[PERF]`, `[JSON]`, `--output`, and regression-check value.
`bandwidth_gbs` is recomputed from `extras.bytes_rw / median_time_us / 1e3`.
The runner also prints `n`, `median`, `mean`, `min`, and `max` for each case when
`N > 1`; use these to spot unstable cases before accepting a refreshed
baseline.

Standalone examples:

```bash
python /root/tilelang_musa/benchmark/mp31/tilekernels/per_token_cast_benchmark.py

python /root/tilelang_musa/benchmark/mp31/tilekernels/topk_sum_and_topk_group_idx_benchmark.py \
  --check-regression

python /root/tilelang_musa/benchmark/mp31/tilekernels/pre_big_fuse_benchmark.py \
  --cases mhc_pre_big_fuse_2048_4096 representative_mhc_pre_big_fuse_2048_4096
```

Run all MP31 benchmark sources:

```bash
python /root/tilelang_musa/benchmark/mp31/runner.py \
  --source all \
  --allow-non-release-build
```

The aggregate runner emits one final `MP31 Benchmark Summary` section with the
combined status, case count, and total wall-clock time for all selected
benchmark sources.

Run only MATE-origin benchmarks:

```bash
python /root/tilelang_musa/benchmark/mp31/runner.py \
  --source mate \
  --allow-non-release-build
```

Run MATE-origin standalone benchmarks:

```bash
python /root/tilelang_musa/benchmark/mp31/mate/ops/gdn_decode_benchmark.py \
  --allow-non-release-build

python /root/tilelang_musa/benchmark/mp31/mate/ops/gdn_mtp_benchmark.py \
  --allow-non-release-build

python /root/tilelang_musa/benchmark/mp31/mate/ops/gdn_prefill_benchmark.py \
  --allow-non-release-build

python /root/tilelang_musa/benchmark/mp31/mate/ops/sparse_mla_prefill_benchmark.py \
  --allow-non-release-build

python /root/tilelang_musa/benchmark/mp31/mate/ops/sparse_mla_decode_benchmark.py \
  --allow-non-release-build
```

CLI Notes:

- The aggregate runner and standalone entrypoints support `--output`,
  `--check-regression`, `--baseline`,
  `--threshold`, `--samples`, `--cases`, and `--allow-non-release-build`.
- For single-op entrypoints, `--cases` can be used to narrow to a subset of the
  cases owned by that operator family.
- MATE-origin `--check-regression` is currently covered only for GDN decode,
  GDN MTP, and GDN prefill cases. The aggregate runner automatically skips
  unsupported Sparse MLA cases when `--check-regression` is used without an
  explicit `--cases` list. If unsupported cases are explicitly requested with
  `--check-regression`, the runner exits with a clear error instead of treating
  missing baselines as a pass.
- Prefer `--samples 5` or higher when refreshing baseline JSONL files.
  Median is used instead of mean so an occasional slow run does not shift the
  baseline.

## MATE Benchmark Migration Notes

The MATE-origin benchmarks are direct TileLang microbenchmarks. They do not run
MATE's public Python APIs, logging decorators, FlashMLA wrappers, or testing
utilities. Inputs, outputs, metadata, and optional buffers are constructed by
the local benchmark harness in `mate/benchmark_cases.py`.

Migrated GDN cases:

- `gdn_decode`: direct `gated_deltanet_decode_fp32_vk` TileLang backend.
- `gdn_mtp`: direct `gated_deltanet_mtp_fp32_vk_smem` TileLang backend.
- `gdn_prefill`: local three-kernel pipeline:
  `chunk_local_cumsum`, `kkt_solve`, and `fused_chunk_gdn_prefill`.

Migrated Sparse MLA cases:

- `sparse_mla_prefill_v32`: direct V3.2 prefill TileLang interface.
- `sparse_mla_prefill_model1`: direct Model1 prefill TileLang interface with
  extra-KV inputs.
- `sparse_mla_decode_v32`: direct V3.2 scheduled decode TileLang interface.

The default MATE aggregate runner includes all migrated Sparse MLA cases,
including the large Model1 extra-KV case
`sparse_mla_prefill_model1_extra_bf16`.

The Sparse MLA decode migration uses a local scheduled metadata generator
equivalent to MATE's temp `get_mla_metadata_pytorch` helper for the direct V3.2
decode case. This covers the migrated direct benchmark path without depending
on MATE's temp metadata utilities or FlashMLA.

When adding more MATE-origin cases, keep these rules:

- Do not import `mate`, `flash_mla`, `sparse_mla_test_utils`, or files from a
  checked-out MATE repository.
- Keep source-specific code under `benchmark/mp31/mate`; keep TileKernels-origin
  code under `benchmark/mp31/tilekernels`.
- Prefer direct TileLang kernel factories or TileLang interfaces. Wrapper-level
  behavior should only be reimplemented when it is required to construct the
  kernel ABI.
- Avoid `from __future__ import annotations` in files that define TileLang
  `T.prim_func` signatures, because TileLang evaluates annotations when
  building TIR.

## Workflow

The four components form a complete automated performance detection pipeline, following the steps below:

1. **Run Benchmarks**: Execute all predefined performance test cases to generate real-time current performance metrics.
2. **Compare with Baseline**: Match the newly obtained benchmark results against the stable historical baseline data.
3. **Detect Regression (Performance Degradation):** Identify performance degradation if the current metrics are slower or worse than the baseline.
4. **Trigger Guardrail Check**: Judge the severity of the regression. If the performance decline exceeds the preset threshold (5% slowdown), trigger a check failure to alert and prevent problematic code merging.
