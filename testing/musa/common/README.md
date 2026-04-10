# MUSA Common Test Cases

The `common` directory contains architecture-agnostic MUSA test cases that validate fundamental TileLang behaviors, including elementwise ops, local-buffer copy semantics, reduction finalization, manual FMA kernels, and loop unrolling codegen.

## Test Cases (One Line per Case)

- `test_add_global.py`: Verifies vectorized elementwise add directly on global memory.
- `test_add_fragment.py`: Verifies elementwise add through fragment-local accumulation followed by copy-out.
- `test_copy_to_local.py`: Verifies `T.copy` to local buffer lowering avoids buggy thread-partitioned writes and matches numerical reference.
- `test_finalize_reducer.py`: Verifies `T.finalize_reducer` cross-warp lowering uses named barriers correctly and produces correct row-wise sums.
- `test_fma.py`: Verifies a manual shared-memory + local-tile FMA matmul with fused ReLU output.
- `test_reduce_sum.py`: Verifies cross-warp `T.reduce_sum` lowers to `AllReduce` with expected named-barrier initialization and correct results.
- `test_unroll_factor.py`: Verifies `T.unroll(..., unroll_factor=4)` emits `#pragma unroll 4` and preserves numerical correctness.
