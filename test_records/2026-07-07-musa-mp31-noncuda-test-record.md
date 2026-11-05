# MUSA MP31 Non-CUDA Test Record

Date: 2026-07-07 19:49 CST
Repo: `/host/code/tilelang`
Branch: `0160_tirx`
HEAD: `96f43ebd`
Base develop commit noted by requester: `a3e2291d`

## Environment

- PyTorch: `2.9.1`
- MUSA available: `True`
- Device count: `8`
- Device name: `MTT S5000`
- Device capability: `(3, 1)`
- Parallel test execution used `MUSA_VISIBLE_DEVICES=0..7`.

## Scope

Included screenshot/local paths:

- `testing/test_conftest_fixture_probe.py`
- `testing/musa/common`
- `testing/musa/mp31/basic`
- `testing/musa/mp31/dsa`
- `testing/musa/mp31/extension`
- `testing/musa/mp31/moe`
- `testing/musa/mp31/vectorize`
- `testing/python/analysis`
- `testing/python/arith`
- `testing/python/components`
- `testing/python/debug`
- `testing/python/fastmath`
- `testing/python/issue`
- `testing/python/jit`
- `testing/python/kernel`
- `testing/python/language`
- `testing/python/layout`
- `testing/python/math`
- `testing/python/profiler`
- `testing/python/runtime`
- `testing/python/target`
- `testing/python/transform`
- `testing/python/utils`

Excluded by request:

- `testing/python/cuda`
- `testing/musa/mp31/sqmma`
- `testing/musa/mp31/wmma`
- `benchmark/mp31`
- Other backend-only test directories not shown in scope.

Note: Generic `testing/python/*` directories may contain CUDA-targeted unit tests internally. The top-level `testing/python/cuda` directory was excluded.

## Collection

Command:

```bash
pytest --collect-only -q \
  testing/test_conftest_fixture_probe.py \
  testing/musa/common \
  testing/musa/mp31/basic \
  testing/musa/mp31/dsa \
  testing/musa/mp31/extension \
  testing/musa/mp31/moe \
  testing/musa/mp31/vectorize \
  testing/python/analysis \
  testing/python/arith \
  testing/python/components \
  testing/python/debug \
  testing/python/fastmath \
  testing/python/issue \
  testing/python/jit \
  testing/python/kernel \
  testing/python/language \
  testing/python/layout \
  testing/python/math \
  testing/python/profiler \
  testing/python/runtime \
  testing/python/target \
  testing/python/transform \
  testing/python/utils
```

Result:

- `1794 tests collected`
- `1 collection error`

Collection blocker:

- `testing/python/transform/test_tilelang_transform_lower_tile_op_layout.py`
- Error: `ValueError: Cannot find global function tl.transform._TestingExpandLayoutToBufferInputShape`

## Local Pytest Results

Overall local result, excluding skipped from the effective denominator:

- Passed: `1623`
- Failed or error: `53`
- Skipped: `133`
- Effective pass rate: `1623 / (1623 + 53) = 96.84%`
- Including skipped: `1623 / (1623 + 53 + 133) = 89.72%`

MUSA/MP31 subset result:

- Passed: `595`
- Failed: `18`
- Skipped: `0`
- Pass rate: `595 / 613 = 97.06%`

## Batch Results

| Batch | Device | Result |
| --- | --- | --- |
| `testing/musa/common testing/musa/mp31/extension testing/musa/mp31/moe testing/musa/mp31/vectorize` | default/0 | `90 passed` |
| `testing/musa/mp31/basic/test_gemm.py` | `MUSA_VISIBLE_DEVICES=1` | `220 passed` |
| `testing/musa/mp31/basic/test_gemm_fma.py` | `MUSA_VISIBLE_DEVICES=2` | `16 failed, 2 passed` |
| `testing/musa/mp31/dsa` | `MUSA_VISIBLE_DEVICES=3` | `120 passed` |
| `testing/musa/mp31/basic/test_gemm_reduce_max.py testing/musa/mp31/basic/test_gemm_reduce_sum.py testing/musa/mp31/basic/test_gemm_trans_b.py testing/musa/mp31/basic/test_lower_shared_barrier.py` | `MUSA_VISIBLE_DEVICES=4` | `2 failed, 5 passed` |
| `testing/musa/mp31/basic/test_musa_packed_x2_intrinsics.py testing/musa/mp31/basic/test_musa_packed_x4_intrinsics.py testing/musa/mp31/basic/test_musa_packed_x8_intrinsics.py testing/musa/mp31/basic/test_parallel_load.py` | `MUSA_VISIBLE_DEVICES=5` | `145 passed` |
| `testing/musa/mp31/basic/test_tilelang_transform_musa_tme_prefetch.py testing/musa/mp31/basic/test_tma_load.py testing/musa/mp31/basic/test_tma_load_store.py` | `MUSA_VISIBLE_DEVICES=6` | `12 passed` |
| `testing/musa/mp31/basic/test_warp_specialize_gemm_barrierpipe_stage2.py` | `MUSA_VISIBLE_DEVICES=7` | `1 passed` |
| `testing/test_conftest_fixture_probe.py testing/python/analysis testing/python/arith` | `MUSA_VISIBLE_DEVICES=0` | `99 passed` |
| `testing/python/components testing/python/debug testing/python/fastmath testing/python/issue` | `MUSA_VISIBLE_DEVICES=2` | `2 failed, 92 passed, 11 skipped` |
| `testing/python/jit testing/python/kernel` | `MUSA_VISIBLE_DEVICES=3` | `9 failed, 25 passed, 23 skipped` |
| `testing/python/language` | `MUSA_VISIBLE_DEVICES=4` | `15 failed, 517 passed, 16 skipped` |
| `testing/python/layout testing/python/math testing/python/profiler testing/python/runtime testing/python/target testing/python/utils` | `MUSA_VISIBLE_DEVICES=5` | `92 passed, 41 skipped` |
| `testing/python/transform` | `MUSA_VISIBLE_DEVICES=6` | `1 collection error` |
| `testing/python/transform`, excluding `test_tilelang_transform_lower_tile_op_layout.py` | `MUSA_VISIBLE_DEVICES=6` | `8 failed, 203 passed, 42 skipped` |

## Failure Summary

### `testing/musa/mp31/basic/test_gemm_fma.py`

Result: `16 failed, 2 passed`

Primary failure signatures:

- `IndexError: Buffer A_shared is 3-dimensional ... but 2 index(es) were provided`
- `InternalError: M must be divisible by 4`
- `InternalError: N must be divisible by 8`

Representative location:

- `tilelang/musa/op/gemm/gemm_fma.py:88`

### `testing/musa/mp31/basic/test_lower_shared_barrier.py`

Result: `2 failed, 5 passed` in the grouped batch.

Failed tests:

- `test_lower_shared_barrier_to_named_barrier`
- `test_lower_shared_barrier_dynamic_index_uses_base_plus_idx`

Failure signature:

- `AttributeError: module 'tilelang.language' has no attribute 'block_attr'. Did you mean: 'sblock_attr'?`

### `testing/python/components testing/python/debug testing/python/fastmath testing/python/issue`

Result: `2 failed, 92 passed, 11 skipped`

Failed tests:

- `testing/python/components/test_tilelang_pass_config_disable_warp_specialized.py::test_gemm_f16f16f16_nn`
- `testing/python/issue/test_tilelang_issue_1697.py::test_gemm_jit_kernel_zero_dim`

Failure signatures:

- Same `gemm_fma.py` 3D-buffer-with-2D-index issue.
- Expected `ValueError` for zero-dim JIT kernel, but no exception was raised.

### `testing/python/jit testing/python/kernel`

Result: `9 failed, 25 passed, 23 skipped`

Failed tests:

- `testing/python/kernel/test_tilelang_kernel_gemm.py::test_gemm_f16f16f16_nn`
- `testing/python/kernel/test_tilelang_kernel_gemm.py::test_gemm_f16f16f16_tn`
- `testing/python/kernel/test_tilelang_kernel_gemm.py::test_gemm_f16f16f16_nt`
- `testing/python/kernel/test_tilelang_kernel_gemm.py::test_pad_aligned_f16f16f16_nn`
- `testing/python/kernel/test_tilelang_kernel_gemm.py::test_pad_f16f16f16_nn`
- `testing/python/kernel/test_tilelang_kernel_gemm.py::test_gemm_f16f16f16_sr`
- `testing/python/kernel/test_tilelang_kernel_gemm.py::test_gemm_f16f16f16_rs`
- `testing/python/kernel/test_tilelang_kernel_gemm_batched.py::test_gemm_f16f16f16_nn`
- `testing/python/kernel/test_tilelang_kernel_gemm_with_stride.py::test_tilelang_kernel_gemm_with_stride`

Failure signatures:

- GEMM numerical mismatch beyond tolerance.
- SQMMA compile-time constraints:
  - `SQMMA doesn't support custom stride for A`
  - `SQMMA doesn't support custom stride for B`
  - `offset_a and offset_b must be zero for SQMMA`

### `testing/python/language`

Result: `15 failed, 517 passed, 16 skipped`

Failed tests:

- `testing/python/language/test_tilelang_language_all_of.py::test_block_sparse_matmul_global`
- `testing/python/language/test_tilelang_language_all_of.py::test_block_sparse_matmul_shared`
- `testing/python/language/test_tilelang_language_all_of.py::test_block_sparse_matmul_local`
- `testing/python/language/test_tilelang_language_any_of.py::test_block_sparse_matmul_global`
- `testing/python/language/test_tilelang_language_any_of.py::test_block_sparse_matmul_shared`
- `testing/python/language/test_tilelang_language_any_of.py::test_block_sparse_matmul_local`
- `testing/python/language/test_tilelang_language_reduce.py::test_reduce[sum-float16-64x128-f2f-t256-b4]`
- `testing/python/language/test_tilelang_language_reduce.py::test_reduce[sum-bfloat16-64x128-f2f-t256-b4]`
- `testing/python/language/test_tilelang_language_reduce.py::test_reduce[max-bfloat16-128x64-s2f-t256-b2]`
- `testing/python/language/test_tilelang_language_reduce.py::test_reduce[min-float16-128x128-f2f-t256-b8]`
- `testing/python/language/test_tilelang_language_tma_copy.py::test_tma_copy_pipeline_2_stages`
- `testing/python/language/test_tilelang_language_tma_copy.py::test_tma_copy_pipeline_3_stages`
- `testing/python/language/test_tilelang_language_tma_copy.py::test_tma_copy_store_pipeline_2_stages`
- `testing/python/language/test_tilelang_language_tma_copy.py::test_tma_copy_store_pipeline_3_stages`
- `testing/python/language/test_tilelang_language_view.py::test_view_symbolic_shape_equivalence`

Failure signatures:

- MUSA codegen internal check failure for block sparse matmul.
- TMA copy/store pipeline compile failure in `mcc`:
  - `fatal error: error in backend: no registers from class available to allocate`
- `T.reshape/view shape check failed`

### `testing/python/transform`

Full directory collection error:

- `testing/python/transform/test_tilelang_transform_lower_tile_op_layout.py`
- `ValueError: Cannot find global function tl.transform._TestingExpandLayoutToBufferInputShape`

After excluding that file:

Result: `8 failed, 203 passed, 42 skipped`

Failed tests:

- `testing/python/transform/test_tilelang_transform_Inject_software_pipeline.py::test_inject_software_pipeline_replays_scalar_let_without_annotation_slot`
- `testing/python/transform/test_tilelang_transform_if_stmt_binding.py::test_if_stmt_binding_keeps_direct_bind_scope`
- `testing/python/transform/test_tilelang_transform_inject_set_max_nreg.py::test_inject_set_max_nreg`
- `testing/python/transform/test_tilelang_transform_lower_shared_barrier.py::test_plan_update_keeps_barrier_init_with_tcgen05_no_tma`
- `testing/python/transform/test_tilelang_transform_plan_update_buffer_allocation_location.py::test_plan_update_keeps_loop_header_local_var_outside_loop_body`
- `testing/python/transform/test_tilelang_transform_vectorize_single_side.py::test_vectorize_single_side_shared_to_global_lowers_to_stg`
- `testing/python/transform/test_tilelang_transform_vectorize_single_side.py::test_vectorize_single_side_shared_to_global_uses_wide_stg`
- `testing/python/transform/test_tilelang_transform_vectorize_single_side.py::test_vectorize_single_side_global_to_shared_lowers_to_ldg`

Failure signatures:

- Old API names still referenced in tests:
  - `T.block`
  - `T.LetStmt`
- Missing registered operator:
  - `tl.annotate_producer_reg_dealloc`
- Target-specific implementation missing for some CUDA-targeted transform tests:
  - `tl.copy requires a target-specific implementation`
  - `tl.fill requires a target-specific implementation`
- Expected MUSA vectorized `ldg/stg` intrinsic not found in lowered IR.

## Screenshot Badge Count

For the screenshot scope, excluding `testing/python/cuda`, the badge count was:

- Green: `1706`
- Red/yellow: `23`
- Gray: `66`
- Total: `1795`

Badge-derived rates:

- Excluding gray: `1706 / (1706 + 23) = 98.67%`
- Including gray: `1706 / 1795 = 95.04%`

This differs from the local pytest result because the screenshot badge statuses and the local run statuses are not identical. The local run found additional failures in generic `testing/python/*` suites.

## Follow-Up Buckets

Most actionable local failures appear to cluster into these buckets:

1. MUSA FMA GEMM fallback does not handle 3D shared layouts with rank-aware indexing.
2. Some tests still reference pre-tirx API names such as `T.block`, `T.block_attr`, and `T.LetStmt`.
3. Generic Python GEMM tests expose MUSA/SQMMA stride, offset, and numerical issues.
4. `testing/python/transform` has one missing test-only global function registration.
5. TMA copy pipeline cases hit an `mcc` register allocation failure on MP31.
6. Some generic transform tests are still CUDA-targeted despite the top-level CUDA directory being excluded.
