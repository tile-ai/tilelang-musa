# TileLang MUSA Programming Guide

## Force Async Copy
```python
## Example 1
for v in T.vectorized(4):
    T.copy(src_global[v], dst_shared[v], force_async_copy=True)

## Example 2
with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
    dst_shared = T.alloc_shared([128], T.float16)
    tile_start = bx * 128
    T.copy(
        src_global[tile_start : tile_start + 128],
        dst_shared[0:128],
        force_async_copy=True,
    )
```
- Function: `force_async_copy=True` explicitly requests lowering the corresponding `global -> shared` `T.copy` into `async copy`. This is only a request; successful lowering requires the following conditions.
  - The source `buffer` must be `global`, the destination `buffer` must be `shared`, and the source and destination `dtype` must match.
  - The number of bytes copied by a single `async copy` must be 4, 8, or 16 bytes. This can be formed with a vectorized `copy` through `T.vectorized(...)`, or with a contiguous `BufferRegion` slice of the corresponding width.
- Note:
  - By default, users do not need to manually insert `commit/wait`. When both `TL_DISABLE_WARP_SPECIALIZED=True` and `TL_DISABLE_THREAD_STORAGE_SYNC=True` are set, users need to insert `T.ptx_commit_group()` and `T.ptx_wait_group(N)` before consuming the `shared buffer` to manage `async copy` synchronization semantics.

## Robust Copy
```python
## Example
robust_desc = T.make_robust_desc(T.address_of(src_global[1]), 8)
T.copy(src_global[tid], dst_shared[tid], src_robust_desc=robust_desc)
```
- Function: `src_robust_desc` specifies the valid byte range of the source address for `T.copy`, so MUSA lowering can use `robust load` / `robust async copy` when the source side may be out of bounds. The `descriptor` is created with `T.make_robust_desc(addr, size_bytes)`, where `addr` is usually obtained with `T.address_of(...)`, and `size_bytes` is the number of valid bytes starting from that address.
- Note:
  - `src_robust_desc` only describes the valid range on the source side. The source `buffer` must be `global`, and the `descriptor` must be created by `T.make_robust_desc(...)`.
  - `src_robust_desc` can be combined with `force_async_copy=True`, for example `T.copy(src_region, dst_shared, force_async_copy=True, src_robust_desc=robust_desc)`.

## Manual TME barrier
```python
## Example 1
barrier = T.alloc_barrier(128)
T.copy(src[0], tile, barrier=barrier)
T.barrier_arrive(barrier)
T.barrier_wait(barrier, 0)

## Example 2
with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
    tile = T.alloc_shared((block_n,), T.float32)
    barriers = T.alloc_barrier([128, 128])
    barrier = barriers[bx % 2]
    T.copy(src[bx * block_n], tile, barrier=barrier)
    T.barrier_arrive(barrier)
    T.barrier_wait(barrier, 0)
```
- Function: `T.copy(..., barrier=barrier)` explicitly specifies the `shared barrier` used by a `TME copy` and gives users control over `barrier arrive/wait`. A typical flow is to allocate a `barrier` with `T.alloc_barrier(arrive_count)`, issue the `copy` with `T.copy(src, dst_shared, barrier=barrier)`, and then synchronize with `T.barrier_arrive(barrier)` and `T.barrier_wait(barrier, parity)`.
- Note:
  - `barrier` must come from `T.alloc_barrier(...)`. After using `T.copy(..., barrier=barrier)`, users must call the matching `T.barrier_arrive(...)` and `T.barrier_wait(...)` before consuming the destination `shared buffer`.
  - `barrier` only takes effect when that `T.copy` can be lowered into a `TME copy`. Normal `scalar copy` or `SIMT copy` will not use this `barrier`.

## TME cache policy hints
```python
## Example 1: use MUSA inner/outer cache policy
T.copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    inner_cache_policy="cache_none",
    outer_cache_policy="cache_persist",
)

## Example 2: use NV-compatible eviction policy
T.copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    eviction_policy="evict_first",
)

## Example 3: specify cache policy for descriptor TME store
T.copy(
    A_shared,
    A_global[0:block_m, 0:block_n],
    inner_cache_policy="cache_once",
    outer_cache_policy="cache_normal",
)

## Example 4: specify cache policy for T.tma_copy
barrier = T.alloc_barrier(128)
T.tma_copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    barrier=barrier,
    inner_cache_policy="cache_once",
    outer_cache_policy="cache_normal",
)
```
- Function: `T.copy(...)` and `T.tma_copy(...)` support specifying cache policy hints for descriptor-based MUSA `TME load`; `T.copy(...)` also supports cache policy hints for descriptor-based `TME store`. Users can set MUSA inner/outer cache behavior separately with `inner_cache_policy` / `outer_cache_policy`, or use `eviction_policy` for NV-compatible paired hints.
  - Valid values for `inner_cache_policy` and `outer_cache_policy` are `"cache_none"`, `"cache_once"`, `"cache_normal"`, and `"cache_persist"`. When omitted, they default to `"cache_normal"`.
  - Valid values for `eviction_policy` are `"evict_normal"`, `"evict_first"`, and `"evict_last"`, which map to paired inner/outer settings of `"cache_normal"`, `"cache_once"`, and `"cache_persist"` respectively.
- Note:
  - `eviction_policy` cannot be combined with `inner_cache_policy` or `outer_cache_policy`. Use `inner_cache_policy` / `outer_cache_policy` directly when MUSA inner/outer cache behavior needs to be controlled separately.
  - The current MUSA backend supports explicit cache policy hints only for descriptor-based `TME load/store`; `1D TME load/store`, `tma_store_add`, and `tma_load_im2col` currently only support the default `"cache_normal"`.

## T.gemm gemm_ss
```python
## Example
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_N, block_K), dtype)
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

T.copy(A_global[by * block_M, ko * block_K], A_shared)
T.copy(B_global[bx * block_N, ko * block_K], B_shared)
T.gemm(
    A_shared,
    B_shared,
    C_local,
    transpose_B=True,
    policy=T.GemmWarpPolicy.FullRow,
)
```
- Function: `gemm_ss` is the `T.gemm(...)` form where both `A` and `B` operands are `shared buffers`. The MUSA backend extends the lowering path of `T.gemm(A_shared, B_shared, ...)`. For legal `shapes`, `dtypes`, and thread layouts, it uses the priority `SQMMA -> WMMA -> FMA`: if `SQMMA` can handle the current `T.gemm`, it is preferred; otherwise MUSA tries `WMMA`, and finally falls back to `FMA`.
  - `policy` controls `GEMM` `warp` partitioning. Common values include `T.GemmWarpPolicy.Square`, `T.GemmWarpPolicy.FullRow`, and `T.GemmWarpPolicy.FullCol`.
  - `transpose_A` / `transpose_B` describe the logical transpose relation of `GEMM` operands, and also affect the `k_major` value used by manual `SQMMA layout` annotation.
- Note:
  - When `tilelang.PassConfigKey.TL_DISABLE_SQMMA: True` is set, `gemm_ss` will not be lowered to `SQMMA`, and instead follows the `WMMA -> FMA` priority.
  - When `tilelang.PassConfigKey.TL_DISABLE_PH1_WMMA: True` is set, `gemm_ss` will not be lowered to `WMMA`, and instead follows the `SQMMA -> FMA` priority.
  - When both `tilelang.PassConfigKey.TL_DISABLE_SQMMA: True` and `tilelang.PassConfigKey.TL_DISABLE_PH1_WMMA: True` are set, `gemm_ss` will only be lowered to `FMA`.

## T.gemm gemm_ss wg_wait
```python
## Example
T.gemm(A_shared, B_shared, C_local, wg_wait=-1)
T.wait_sqmma()
```
- Function: `wg_wait` controls wait behavior for MUSA `SQMMA GEMM`. With the default `wg_wait=0`, the compiler guarantees `SQMMA` synchronization. `wg_wait=-1` means `SQMMA` is issued without an immediate wait, and users must explicitly call `T.wait_sqmma()` later.
- Note:
  - `wg_wait` only takes effect when the current `T.gemm` is lowered to `SQMMA`.
  - When using `wg_wait=-1`, users must call `T.wait_sqmma()` before reading, copying, or continuing to compute with that `accumulator`.
  - `T.wait_sqmma()` is an alias of `T.wait_wgmma()`; they are equivalent.

## T.gemm gemm_rr
```python
## Example
A_fragment = T.alloc_fragment((block_M, block_K), dtype)
B_fragment = T.alloc_fragment((block_N, block_K), dtype)
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

T.copy(A_global[by * block_M, ko * block_K], A_fragment)
T.copy(B_global[bx * block_N, ko * block_K], B_fragment)
T.gemm(A_fragment, B_fragment, C_local, transpose_B=True)
```
- Function: `gemm_rr` is the `T.gemm(...)` form where both `A` and `B` operands are `local fragments`. Users still call `T.gemm(A, B, C, ...)`; the compiler recognizes `gemm_rr` automatically from the `scope` of `A_fragment` and `B_fragment`. On MUSA, `gemm_rr` follows the `WMMA -> FMA` lowering priority.
  - `transpose_A` / `transpose_B` still describe the logical transpose relation of `GEMM` operands.
  - `policy` still controls `GEMM` `warp` partitioning. Common values include `T.GemmWarpPolicy.Square`, `T.GemmWarpPolicy.FullRow`, and `T.GemmWarpPolicy.FullCol`.
- Note:
  - `gemm_rr` requires both `A` and `B` to be fragments allocated by `T.alloc_fragment(...)`, and `C` must also be a `local fragment`.
  - MUSA `SQMMA` does not support `gemm_rr`. If `PH1 WMMA` can handle `gemm_rr`, it is preferred; otherwise `gemm_rr` falls back to `FMA`.
  - When `tilelang.PassConfigKey.TL_DISABLE_PH1_WMMA: True` is set, `gemm_rr` will not be lowered to `WMMA` and will be lowered directly to `FMA`.

## make_sqmma_swizzled_layout
```python
## Example 1: annotate a whole shared buffer
KV_shared = T.alloc_shared((block_m, block_k), T.float16)
T.annotate_layout(
    {KV_shared: tilelang.layout.make_sqmma_swizzled_layout(KV_shared, k_major=True)}
)

## Example 2: annotate a BufferRegion
KV_shared = T.alloc_shared((2, block_m, block_k), T.float16)
V_region = KV_shared[1, :, :]
T.annotate_layout(
    {V_region: tilelang.layout.make_sqmma_swizzled_layout(V_region, continuity=64, k_major=False)},
    allow_buffer_region=True,
)
```
- Function: `tilelang.layout.make_sqmma_swizzled_layout(buffer, continuity=None, k_major=True)` generates a `swizzled layout` for MUSA `SQMMA shared operands`, and binds it to the corresponding `shared buffer` or `BufferRegion` through `T.annotate_layout(...)`.
  - `buffer` represents an `SQMMA shared operand A/B`. It can be a `shared buffer` or a `BufferRegion`. When using a `BufferRegion` as the `key` of `T.annotate_layout(...)`, set `allow_buffer_region=True`.
  - `k_major` specifies whether the operand should use a `K-major` `swizzled layout`. It must match the transpose relation of the later `T.gemm(...)`. In general, `A` operands use `k_major=not transpose_A`, and `B` operands use `k_major=transpose_B`.
  - `continuity` specifies the contiguous dimension length. By default, it is the last dimension of `buffer` or `BufferRegion`, namely the last dimension of `A` or `B` in `gemm(A, B)`. `gemm(A, B)` may be split into multiple `sqmma(A_i, B_i)` instructions. When `k_major=False`, `continuity` should be explicitly set to the last dimension of `A_i` or `B_i`.
- Note:
  - This API only creates a `layout` description. It does not allocate `shared memory` and does not move or reorder data. It takes effect only after being passed to `T.annotate_layout(...)`.
  - `BufferRegion` must be at least 2D. If the `region` has more than 2 dimensions, all leading dimensions must have size 1; the actual `swizzle` applies to the last two dimensions.

## Annotate Layout allow_buffer_region
```python
## Example 1: annotate a single BufferRegion
KV_shared = T.alloc_shared((2, block_m, block_k), T.float16)
K_region = KV_shared[0, :, :]
T.annotate_layout(
    {K_region: tilelang.layout.make_sqmma_swizzled_layout(K_region, k_major=True)},
    allow_buffer_region=True,
)

## Example 2: annotate a 2D slice region
V_shared = T.alloc_shared((block_m, block_n), T.float16)
V_region = V_shared[:, 0:block_n]
T.annotate_layout(
    {V_region: tilelang.layout.make_sqmma_swizzled_layout(V_region, continuity=64, k_major=False)},
    allow_buffer_region=True,
)
```
- Function: `allow_buffer_region=True` allows the `key` of `T.annotate_layout(...)` to be a `BufferRegion`, for example `KV_shared[0, :, :]`. A typical use case is splitting a higher-dimensional `shared buffer` into multiple `2D operand regions` and annotating `SQMMA swizzled layouts` for those regions.
- Note:
  - The default is `allow_buffer_region=False`. In that mode, `T.annotate_layout({KV_shared[0, :, :]: layout})` will raise an error.

## Annotate Layout allow_reannotation
```python
## Example: use different layouts for the same shared buffer in different stages
KV_shared = T.alloc_shared((block_m, block_k), T.float16)

T.annotate_layout(
    {KV_shared: tilelang.layout.make_sqmma_swizzled_layout(KV_shared, k_major=True)},
    allow_reannotation=True,
)
T.copy(K_global, KV_shared)
T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True)

T.annotate_layout(
    {KV_shared: tilelang.layout.make_sqmma_swizzled_layout(KV_shared, continuity=64, k_major=False)},
    allow_reannotation=True,
)
T.copy(V_global, KV_shared)
T.gemm(S_shared, KV_shared, acc_o)
```
- Function: `allow_reannotation=True` allows the same `buffer` to be annotated by `T.annotate_layout(...)` multiple times. A typical use case is a MUSA multi-`GEMM` kernel that reuses the same `shared buffer`: an earlier stage uses the `layout` for a `K tile`, while a later stage reannotates it to the `layout` needed by a `V tile` or another operand.
- Note:
  - Set `allow_reannotation=True` only when the same `buffer` needs repeated `layout annotations`. A single annotation does not need it.
  - Reannotation takes effect in code order. The new `layout` applies to `copy` / `GEMM` and other statements after that `T.annotate_layout(...)`.
  - `allow_reannotation=True` only permits updating `layout annotations`. It does not automatically reorder existing data in the `shared buffer`, so it should usually be placed before the next write or the next compute stage.
  - `allow_reannotation` and `allow_buffer_region` can be used together.

## Safety-related PassConfig
```python
## Example
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_COPY_PREDICATION: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION: True,
}
kernel = tilelang.compile(func, target="musa", pass_configs=pass_configs)
```
- Function:
  - `TL_DISABLE_SAFE_MEMORY_ACCESS` disables the generic `global memory` out-of-bounds protection added by later compiler passes. It mainly affects normal `global memory load/store` and some accesses already lowered to `async copy` that the compiler still considers potentially out of bounds.
  - `TL_DISABLE_SAFE_COPY_PREDICATION` disables the safety predicates generated during `T.copy` lowering. It mainly affects `T.copy` on `tail tiles` / `boundary tiles`: `source` out-of-bounds reads are filled with 0, and `destination` writes only happen within the valid range.
  - `TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION` disables the final safety predicate attached to MUSA `robust async copy`. It only affects compiler-generated `predicated robust async copy`, making final `emission` drop the `predicate` on that `robust async copy`.
- Note:
  - `TL_DISABLE_SAFE_MEMORY_ACCESS` does not disable the safety predicates generated during `T.copy` lowering.
  - `TL_DISABLE_SAFE_COPY_PREDICATION` does not disable the generic `global memory` safety protection added by later compiler passes.
  - `TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION` does not affect normal `T.copy`, normal `global memory load/store`, or normal `robust load`.
  - All three options default to `False`. Enable them only when access ranges are already guaranteed by `launch geometry`, user `guards`, or `robust descriptors`.

## TL_DISABLE_INDEX_TYPE_PROMOTION
```python
## Example
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
}
kernel = tilelang.compile(func, target="musa", pass_configs=pass_configs)
```
- Function: `TL_DISABLE_INDEX_TYPE_PROMOTION=True` disables automatic type promotion for `index` expressions. By default, TileLang promotes `index` expressions that may overflow the current integer width to a wider integer type in passes such as `FlattenBuffer` and `ConfigIndexBitwidth`, for example from `int32` to `int64`.
- Note:
  - This option defaults to `False`, meaning the compiler is allowed to perform `index type promotion` by default.
  - If the access range, `stride`, or `flattened offset` may exceed the `int32` range, enabling this option can cause `index` overflow and incorrect memory access. In that case, users need to perform `int64` type promotion manually to keep the access safe.
