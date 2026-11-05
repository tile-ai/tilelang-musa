# MUSA MP31 Test Cases

The `mp31` directory contains tests and operator examples for MUSA MP31 (S5000), covering core kernel features, extension semantics and transform validation, model-level operator implementations, and GEMM shape-legality regression checks.

## basic: Core Kernel Feature Coverage

- `basic/test_add_tma.py`: Elementwise Add + TMA Copy.
- `basic/test_gemm.py`: SQMMA GEMM + TMA Copy.
- `basic/test_gemm_fma.py`: Verifies the FMA GEMM path with SQMMA disabled.
- `basic/test_gemm_reduce_max.py`: SQMMA GEMM + reduce_max.
- `basic/test_gemm_reduce_sum.py`: SQMMA GEMM + reduce_sum.
- `basic/test_gemm_trans_b.py`: Focused validation of the GEMM `transpose_B` parameter behavior.
- `basic/test_lower_shared_barrier.py`: Verifies `LowerSharedBarrier` pass lowering from `shared.barrier` allocation to named-barrier placeholder calls.
- `basic/test_parallel_load.py`: Verifies manual parallel load-to-shared GEMM path with TMA/warp-specialized disabled.
- `basic/test_tma_1d.py`: 1D TMA load/store case that checks both numerical correctness and generated `tl::tma_load` codegen.

## extension: Extension Semantics and Transform Validation

- `extension/test_allow_annotate_buffer_region.py`: Verifies `T.annotate_layout` re-annotation on sub-buffer regions (`allow_buffer_region=True`) in a two-GEMM flow.
- `extension/test_allow_reannotation.py`: Verifies re-annotating the same shared buffer with different SQMMA layouts across two GEMMs (`allow_reannotation=True`).
- `extension/test_force_async_copy.py`: Verifies that scalar/vectorized `force_async_copy=True` emits `cp_async`, and checks wait insertion behavior under thread-storage-sync toggles.
- `extension/test_producer_threads.py`: Verifies `producer_threads` effects on producer/consumer thread partitioning and runtime copy correctness.
- `extension/test_robust_copy.py`: Verifies robust copy / robust async copy semantics (including zero-sized robust regions) and corresponding codegen patterns.
- `extension/test_tilelang_transform_late_vectorize_planner.py`: Verifies LateVectorizePlanner boundaries for when `exp2`/`cast` patterns should and should not be vectorized.
- `extension/test_wait_sqmma.py`: Verifies synchronization semantics of `wg_wait` + `T.wait_wgmma()` and correctness with overlapped independent compute.

## ldsm_sqmma / tme_sqmma

- `ldsm_sqmma/`: Primarily validates GEMM shape legality on the **LDSM + SQMMA** path, covering combinations across `AB/ABt/AtB`, `fp16/fp8/tf32`, `stage0/stage3`, and `basic/splitM/splitN/splitK`.
- `tme_sqmma/`: Primarily validates GEMM shape legality on the **TME + SQMMA** path, currently covering representative split combinations for `AB` and `AtB` variants.

## dsa: Model-Level DSA/MLA Cases (One Line per Case)

- `dsa/act_quant_kernel.py`: Group-wise activation quantization to FP8 with per-group scale output.
- `dsa/fp8_gemm_kernel.py`: FP8xFP8 GEMM with scale-based dequantization, outputting BF16/FP32.
- `dsa/fp8_index_kernel.py`: FP8 QK index-score kernel (`GEMM + ReLU + reduction`).
- `dsa/quant.py`: KV-cache quantize/dequantize utilities and layout conversion helpers.
- `dsa/sparse_mla_fwd_pipelined_v1.py`: Sparse MLA prefill forward (pipelined v1).
- `dsa/sparse_mla_fwd_pipelined_v2.py`: Sparse MLA prefill forward (pipelined v2).
- `dsa/sparse_mla_fwd_sglang_v1.py`: Sparse MLA prefill forward in sglang-style interface (v1).
- `dsa/sparse_mla_fwd_sglang_v2.py`: Sparse MLA prefill forward in sglang-style interface (v2).
- `dsa/sparse_mla_decode_fwd_pipelined_v1.py`: Sparse MLA decode forward (pipelined v1).
- `dsa/sparse_mla_decode_fwd_pipelined_v2.py`: Sparse MLA decode forward (pipelined v2).
- `dsa/sparse_mla_decode_fwd_scheduled_v2.py`: Sparse MLA decode forward with tile scheduler (scheduled v2).
- `dsa/dsa_decode_v1.py`: End-to-end DSA decode script with TileLang-vs-Torch reference comparison.
- `dsa/kernel.py`: Combined DSA operator entry for quantization, indexing, and sparse attention.
- `dsa/compare.py`: Accuracy comparison helpers (`allclose`, cosine diff, bitwise).
- `dsa/test_sparse_mla_fwd_pipelined.py::test_dsa_decode`: End-to-end correctness/stability for pipelined decode (v2) under random top-k sparse indices.
- `dsa/test_sparse_mla_fwd_pipelined.py::test_dsa_decode_scheduled`: End-to-end correctness/stability for scheduled decode (v2) with tile-scheduler metadata.
- `dsa/test_sparse_mla_fwd_pipelined.py::test_dsa_prefill`: End-to-end prefill (v2) consistency with reference implementation in autoregressive sparse-index scenarios.

## flash_attention: Model-Level Attention Cases (One Line per Case)

- `flash_attention/example_dual_gemm_bhsd.py`: Dual-GEMM attention skeleton (`QK^T` then multiply by `V`) without softmax.
- `flash_attention/example_mha_fwd_bhsd.py`: FlashAttention-style forward with online softmax, causal mask, and tiled pipeline.

## linear_attention: Model-Level Linear-Attention Cases (One Line per Case)

- `linear_attention/example_linear_attn_fwd.py`: Fused chunked linear-attention forward with intra-/inter-chunk contributions and final-state accumulation.
- `linear_attention/example_retention_fwd.py`: Chunked retention forward validating recurrence with head-dependent decay.

## mhc: Model-Level mHC Cases (One Line per Case)

- `mhc/mhc_pre.py`: Fused mHC pre-stage operators (norm, mix split, sinkhorn, residual-mix preprocessing).
- `mhc/mhc_post.py`: Fused mHC post-stage operator combining residual paths and post-mix to produce output.

## nsa: Model-Level NSA Cases (One Line per Case)

- `nsa/act_quant_kernel.py`: Activation group-wise FP8 quantization for the NSA path.
- `nsa/fp8_index_kernel.py`: FP8 index-scoring kernel for the NSA path.
- `nsa/sparse_attention_fwd_kernel.py`: NSA sparse-attention forward kernel with top-k indexed KV gather.
- `nsa/kernel.py`: NSA combined entry that integrates quantization, indexing, and sparse-attention invocation.
