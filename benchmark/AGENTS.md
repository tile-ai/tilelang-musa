# Benchmark Agent Guide

This file is for AI coding agents working under `benchmark/`. It is a local
project guide, not a reusable skill: keep broad TileLang MUSA workflows in
external skills, and keep benchmark-specific source layout, commands, and
guardrails here.

## Scope

- The active benchmark suite lives under `benchmark/mp31`.
- The MP31 bundle contains two independent source groups:
  - `mp31/tilekernels`: TileKernels-origin representative operator benchmarks.
  - `mp31/mate`: MATE-origin TileLang microbenchmarks with local kernel copies.
- The top-level source runner is `benchmark/mp31/runner.py`.
- Baselines are stored in `benchmark/mp31/baselines/`.

## Source Layout

- `mp31/README.md` is the user-facing overview and command reference.
- `mp31/baselines/tilekernels.jsonl` contains the 15-record TileKernels
  release baseline.
- `mp31/baselines/mate.jsonl` contains the MATE-origin release baseline.
- `mp31/tilekernels/benchmark_cases.py` owns the TileKernels case registry,
  per-case setup, median sampling, JSONL output, and regression checks.
- `mp31/tilekernels/benchmark_common.py` owns shared TileKernels benchmark
  helpers, release-build detection, formatting, and baseline loading.
- `mp31/tilekernels/*_benchmark.py` files are standalone per-operator entrypoints.
- `mp31/tilekernels/quant`, `mp31/tilekernels/moe`, and `mp31/tilekernels/mhc`
  contain the local operator implementations used by the benchmarks.
- `mp31/mate/benchmark_cases.py` owns the MATE-origin case registry and local
  input/metadata construction.
- `mp31/mate/benchmark_common.py` mirrors the benchmark harness helpers for
  the MATE-origin source group.
- `mp31/mate/ops/*_benchmark.py` files are standalone MATE-origin entrypoints.
- `mp31/mate/kernels` contains local TileLang kernels and minimal host helpers
  migrated from MATE.

## Important History

- Commit `4e5baa7b1378f96095097650f4b05d0be9ba609e` introduced the MP31
  TileKernels benchmark bundle and removed the older benchmark directories.
- Commit `23de3600952f53d13724c57cb56cf4ff28b24d79` renamed the MP31 benchmark
  package for clarity.
- Later changes added the MATE-origin benchmark source group, unified
  `runner.py`, optional regression checking, explicit slow-case opt-in, and
  reorganized baseline artifacts under `mp31/baselines/`.

## Running Benchmarks

Run from the repository root unless a script intentionally sets its own cwd.

```bash
python benchmark/mp31/runner.py --source all
python benchmark/mp31/runner.py --source tilekernels
python benchmark/mp31/runner.py --source mate
```

The top-level runner prints one final combined `MP31 Benchmark Summary` with the
status, case count, regression totals when enabled, and total wall-clock time
across all selected benchmark sources.

Useful shared options:

- `--allow-non-release-build`: bypass the default `CMAKE_BUILD_TYPE=Release`
  guard for debugging only.
- `--check-regression`: compare current results with the selected source
  baseline.
- `--baseline PATH`: use a non-default JSONL baseline.
- `--threshold FLOAT`: allowed slowdown margin; default is `0.05`.
- `--samples N`: run each case N times and use median `time_us`.
- `--cases NAME ...`: run a selected subset of case names.
- `--output PATH`: write current records as compact JSONL.

MATE-origin regression checks currently have baseline coverage only for GDN
decode, GDN MTP, and GDN prefill. The aggregate runner filters MATE to those
GDN cases when `--check-regression` is used without explicit `--cases`;
explicit unsupported cases should fail with a clear error instead of silently
passing with missing baselines.

Prefer `--samples 5` or higher when refreshing baseline files. The harness uses
median time for output and regression checks, then recomputes bandwidth from the
median time when `extras.bytes_rw` is present.

## Release Baseline Rules

- Regression baselines are release measurements from
  `release_v0.1.8_musa.3`, commit
  `c3ed1bd5272c916a042df7877cdc4b587fb1006a`.
- Do not refresh baseline JSONL files from a non-Release build.
- The benchmark harness rejects non-Release builds by default. Use
  `--allow-non-release-build` only for local investigation.
- Keep baseline records in the compact JSONL schema:
  `kernel`, `operation`, `params`, `time_us`, `bandwidth_gbs`, and optional
  `extras` fields such as `bytes_rw`, `flops`, or `tflops`.
- Keep `time_us` rounded to 2 decimal places and `bandwidth_gbs` rounded to 4
  decimal places when committing refreshed baselines.

## Adding Or Changing Cases

- Put TileKernels-origin work under `mp31/tilekernels`; put MATE-origin work
  under `mp31/mate`.
- Register new cases in the relevant `benchmark_cases.py`.
- Add or update standalone `*_benchmark.py` entrypoints only when a natural
  per-operator entrypoint exists.
- Keep case names stable because regression lookup keys include
  `kernel`, `operation`, and JSON-serialized `params`.
- Make `params` complete enough to distinguish shapes and kernel configs.
- Compute `extras.bytes_rw` when bandwidth is meaningful; compute FLOP metadata
  for math-heavy MATE-origin cases when available.
- Seed random tensors in case setup for reproducibility.
- Validate correctness separately when changing kernel behavior; these scripts
  primarily measure performance.

## MATE-Origin Constraints

- Do not import `mate`, `flash_mla`, `sparse_mla_test_utils`, or files from a
  checked-out MATE repository at runtime.
- Keep MATE-origin benchmarks as direct TileLang microbenchmarks using local
  kernel copies and local host-side helpers.
- Reimplement only the wrapper behavior needed to construct the kernel ABI.
- Extend local sparse metadata generators instead of depending on MATE temp
  metadata utilities.
- Avoid `from __future__ import annotations` in files that define TileLang
  `T.prim_func` signatures, because TileLang evaluates annotations while
  building TIR.
- The default MATE aggregate runner intentionally skips compile-sensitive cases
  listed in `SLOW_CASE_NAMES`; run them explicitly with `--cases` when
  debugging or profiling those shapes.

## Agent Workflow

1. Read `benchmark/mp31/README.md` before editing benchmark behavior.
2. Inspect the relevant source group's `benchmark_cases.py` and
   `benchmark_common.py`.
3. Make the smallest source-local change that preserves the TileKernels/MATE
   boundary.
4. Run a narrow case first with `--allow-non-release-build` if the local build
   is not Release.
5. Run regression checks from a Release build before updating committed
   baselines.
6. Update `benchmark/mp31/README.md` when commands, default cases, baseline
   provenance, or migration constraints change.
