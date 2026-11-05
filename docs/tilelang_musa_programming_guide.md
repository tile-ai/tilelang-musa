# TileLang MUSA Programming Guide

## Force Async Copy
```python
## 示例 1
for v in T.vectorized(4):
    T.copy(src_global[v], dst_shared[v], force_async_copy=True)

## 示例 2
with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
    dst_shared = T.alloc_shared([128], T.float16)
    tile_start = bx * 128
    T.copy(
        src_global[tile_start : tile_start + 128],
        dst_shared[0:128],
        force_async_copy=True,
    )
```
- 功能：`force_async_copy=True` 用于显式请求将对应的 `global -> shared` `T.copy` lowering 成 `async copy`。这只是一个显式请求，能否 lowering 成功需要满足以下条件。
  - 源 `buffer` 必须是 `global`，目标 `buffer` 必须是 `shared`，并且源和目标 `dtype` 必须保持一致。
  - 单条 `async copy` 拷贝的字节数必须是 4、8 或 16 字节；可以通过 `T.vectorized(...)` 写法形成向量化 `copy`，也可以通过连续的 `BufferRegion` slice 写法形成对应宽度的 `copy`。
- 注意：
  - 默认情况下用户不需要手动插入 `commit/wait`，但同时设置 `TL_DISABLE_WARP_SPECIALIZED=True` 和 `TL_DISABLE_THREAD_STORAGE_SYNC=True` 时，用户需要在消费 `shared buffer` 前自己插入 `T.ptx_commit_group()` 和 `T.ptx_wait_group(N)` 来管理 `async copy` 的同步语义。

## Robust Copy
```python
## 示例
robust_desc = T.make_robust_desc(T.address_of(src_global[1]), 8)
T.copy(src_global[tid], dst_shared[tid], src_robust_desc=robust_desc)
```
- 功能：`src_robust_desc` 用于给 `T.copy` 指定源地址的有效字节范围，使 MUSA lowering 在源侧可能越界的 `copy` 场景下使用 `robust load` / `robust async copy`。`descriptor` 通过 `T.make_robust_desc(addr, size_bytes)` 创建，其中 `addr` 通常用 `T.address_of(...)` 获取，`size_bytes` 表示从该地址开始的有效字节数。
- 注意：
  - `src_robust_desc` 只描述源侧有效范围，要求源 `buffer` 是 `global`，并且 `descriptor` 必须由 `T.make_robust_desc(...)` 创建。
  - `src_robust_desc` 可以和 `force_async_copy=True` 组合使用，例如 `T.copy(src_region, dst_shared, force_async_copy=True, src_robust_desc=robust_desc)`。

## Manual TME barrier
```python
## 示例 1
barrier = T.alloc_barrier(128)
T.tma_copy(src[0], tile, barrier=barrier)
T.barrier_arrive(barrier)
T.barrier_wait(barrier, 0)

## 示例 2
with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
    tile = T.alloc_shared((block_n,), T.float32)
    barriers = T.alloc_barrier([128, 128])
    barrier = barriers[bx % 2]
    T.tma_copy(src[bx * block_n], tile, barrier=barrier)
    T.barrier_arrive(barrier)
    T.barrier_wait(barrier, 0)
```
- 功能：`T.tma_copy(..., barrier=barrier)` 用于显式指定 `TME/TMA load` 使用的 `shared barrier`，把 `barrier arrive/wait` 的控制交给用户。典型流程是先用 `T.alloc_barrier(arrive_count)` 分配 `barrier`，再调用 `T.tma_copy(src_global, dst_shared, barrier=barrier)` 发起 `copy`，随后用 `T.barrier_arrive(barrier)` 和 `T.barrier_wait(barrier, parity)` 完成同步。
- 注意：
  - `barrier` 必须来自 `T.alloc_barrier(...)`，并且使用 `T.tma_copy(..., barrier=barrier)` 后，必须由用户在消费目标 `shared buffer` 前调用对应的 `T.barrier_arrive(...)` 和 `T.barrier_wait(...)` 来完成同步。
  - `T.tma_copy` 的 load 侧只发起 `expect_tx + tma_load`，不会像普通 `T.copy` 那样自动完成同步；`T.copy` 当前不再接受 `barrier=` 参数。
  - `barrier` 只适用于 `global -> shared` 的 `TME/TMA load` 路径；普通 `scalar copy` 或 `SIMT copy` 应继续使用 `T.copy(...)`。

## TME cache policy hints
```python
## 示例 1：使用 MUSA inner/outer cache policy
T.copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    inner_cache_policy="cache_none",
    outer_cache_policy="cache_persist",
)

## 示例 2：使用 NV-compatible eviction policy
T.copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    eviction_policy="evict_first",
)

## 示例 3：给 descriptor TME store 指定 cache policy
T.copy(
    A_shared,
    A_global[0:block_m, 0:block_n],
    inner_cache_policy="cache_once",
    outer_cache_policy="cache_normal",
)

## 示例 4：T.tma_copy 显式指定 cache policy
barrier = T.alloc_barrier(128)
T.tma_copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    barrier=barrier,
    inner_cache_policy="cache_once",
    outer_cache_policy="cache_normal",
)
```
- 功能：`T.copy(...)` 和 `T.tma_copy(...)` 支持给 MUSA descriptor 形式的 `TME load` 指定 cache policy hint；`T.copy(...)` 也支持给 descriptor 形式的 `TME store` 指定 cache policy hint。可以通过 `inner_cache_policy` / `outer_cache_policy` 分别指定 MUSA inner/outer cache 行为，也可以通过 `eviction_policy` 使用 NV-compatible 的成对 hint。
  - `inner_cache_policy` 和 `outer_cache_policy` 可选值为 `"cache_none"`、`"cache_once"`、`"cache_normal"`、`"cache_persist"`，省略时默认使用 `"cache_normal"`。
  - `eviction_policy` 可选值为 `"evict_normal"`、`"evict_first"`、`"evict_last"`，分别映射为 `"cache_normal"`、`"cache_once"`、`"cache_persist"` 的 inner/outer 成对设置。
- 注意：
  - `eviction_policy` 不能和 `inner_cache_policy` 或 `outer_cache_policy` 同时设置；如果需要 MUSA inner/outer 分别控制，应直接使用 `inner_cache_policy` / `outer_cache_policy`。
  - 当前 MUSA 后端显式 cache policy hint 只支持 descriptor 形式的 `TME load/store`；`1D TME load/store`、`tma_store_add` 和 `tma_load_im2col` 目前只支持默认 `"cache_normal"`。

## T.gemm gemm_ss
```python
## 示例
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
- 功能：`gemm_ss` 指 `A/B operand` 都是 `shared buffer` 的 `T.gemm(...)` 形态。MUSA 后端扩展了 `T.gemm(A_shared, B_shared, ...)` 的 lowering 路径，合法 `shape`、`dtype` 和线程布局下会按照 `SQMMA -> WMMA -> FMA` 的优先级进行 lowering：`SQMMA` 能处理当前 `T.gemm` 时优先 lowering 成 `SQMMA`，否则尝试 lowering 成 `WMMA`，最后回退成 `FMA`。
  - `policy` 用于控制 `GEMM` 的 `warp` 划分，常用值包括 `T.GemmWarpPolicy.Square`、`T.GemmWarpPolicy.FullRow` 和 `T.GemmWarpPolicy.FullCol`。
  - `transpose_A` / `transpose_B` 表示 `GEMM operand` 的逻辑转置关系，也会影响手动 `SQMMA layout` 中 `k_major` 的取值。
- 注意：
  - 设置 `tilelang.PassConfigKey.TL_DISABLE_SQMMA: True` 时，`gemm_ss` 不再会 lowering 成 `SQMMA`，而是按照 `WMMA -> FMA` 优先级 lowering。
  - 设置 `tilelang.PassConfigKey.TL_DISABLE_PH1_WMMA: True` 时，`gemm_ss` 不再会 lowering 成 `WMMA`，而是按照 `SQMMA -> FMA` 优先级 lowering。
  - 同时设置 `tilelang.PassConfigKey.TL_DISABLE_SQMMA: True` 以及 `tilelang.PassConfigKey.TL_DISABLE_PH1_WMMA: True` 时，`gemm_ss` 只会被 lowering 成 `FMA` 形式。

## T.gemm gemm_ss wg_wait
```python
## 示例
T.gemm(A_shared, B_shared, C_local, wg_wait=-1)
T.wait_sqmma()
```
- 功能：`wg_wait` 用于控制 MUSA `SQMMA GEMM` 的等待行为。默认 `wg_wait=0` 时，会由编译器保证 `SQMMA` 的同步；`wg_wait=-1` 表示发起 `SQMMA` 后不立即等待，用户必须在后续位置显式调用 `T.wait_sqmma()`。
- 注意：
  - `wg_wait` 只有当前 `T.gemm` 被 lowering 成 `SQMMA` 时才会生效。
  - 使用 `wg_wait=-1` 时，必须在读取、`copy` 或继续计算该 `accumulator` 前调用 `T.wait_sqmma()`。
  - `T.wait_sqmma()` 是 `T.wait_wgmma()` 的别名，二者等价。

## T.gemm gemm_rr
```python
## 示例
A_fragment = T.alloc_fragment((block_M, block_K), dtype)
B_fragment = T.alloc_fragment((block_N, block_K), dtype)
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)

T.copy(A_global[by * block_M, ko * block_K], A_fragment)
T.copy(B_global[bx * block_N, ko * block_K], B_fragment)
T.gemm(A_fragment, B_fragment, C_local, transpose_B=True)
```
- 功能：`gemm_rr` 指 `A/B operand` 都是 `local fragment` 的 `T.gemm(...)` 形态。用户仍然调用 `T.gemm(A, B, C, ...)`，编译器会根据 `A_fragment` 和 `B_fragment` 的 `scope` 自动识别成 `gemm_rr`。MUSA 路径下，`gemm_rr` 会按照 `WMMA -> FMA` 的优先级进行 lowering。
  - `transpose_A` / `transpose_B` 仍然表示 `GEMM operand` 的逻辑转置关系。
  - `policy` 仍然用于控制 `GEMM` 的 `warp` 划分，常用值包括 `T.GemmWarpPolicy.Square`、`T.GemmWarpPolicy.FullRow` 和 `T.GemmWarpPolicy.FullCol`。
- 注意：
  - `gemm_rr` 要求 `A` 和 `B` 都是 `T.alloc_fragment(...)` 分配的 `fragment`，`C` 也必须是 `local fragment`。
  - MUSA `SQMMA` 不支持 `gemm_rr`；`gemm_rr` 能使用 `PH1 WMMA` 时会优先 lowering 成 `WMMA`，否则会回退成 `FMA`。
  - 设置 `tilelang.PassConfigKey.TL_DISABLE_PH1_WMMA: True` 时，`gemm_rr` 不再会 lowering 成 `WMMA`，而是直接 lowering 成 `FMA`。

## make_sqmma_swizzled_layout
```python
## 示例 1：标注整个 shared buffer
KV_shared = T.alloc_shared((block_m, block_k), T.float16)
T.annotate_layout(
    {KV_shared: tilelang.layout.make_sqmma_swizzled_layout(KV_shared, k_major=True)}
)

## 示例 2：标注 BufferRegion
KV_shared = T.alloc_shared((2, block_m, block_k), T.float16)
V_region = KV_shared[1, :, :]
T.annotate_layout(
    {V_region: tilelang.layout.make_sqmma_swizzled_layout(V_region, continuity=64, k_major=False)},
    allow_buffer_region=True,
)
```
- 功能：`tilelang.layout.make_sqmma_swizzled_layout(buffer, continuity=None, k_major=True)` 用于为 MUSA `SQMMA` 的 `shared operand` 生成 `swizzled layout`，并通过 `T.annotate_layout(...)` 绑定到对应的 `shared buffer` 或 `BufferRegion`。
  - `buffer` 表示 `SQMMA shared operand A/B`，可以是一个 `shared buffer`，也可以是 `BufferRegion`。使用 `BufferRegion` 作为 `T.annotate_layout(...)` 的 `key` 时，需要设置 `allow_buffer_region=True`。
  - `k_major` 用于指定该 `operand` 是否按 `K-major` 方式生成 `swizzled layout`，需要和后续 `T.gemm(...)` 的转置关系保持一致。一般来说，`A operand` 使用 `k_major=not transpose_A`，`B operand` 使用 `k_major=transpose_B`。
  - `continuity` 用于指定连续维长度，默认值是 `buffer` 或 `BufferRegion` 的最后一维长度，也就是 `gemm(A, B)` 中 `A` 或 `B` 的最后一维长度。`gemm(A, B)` 可能需要拆分成多个 `sqmma(A_i, B_i)` 指令；`k_major=False` 时，`continuity` 的值应该显式设置成 `A_i` 或 `B_i` 的最后一维长度。
- 注意：
  - 这个接口只生成 `layout` 描述，不会分配 `shared memory`，也不会搬运或重排数据；实际生效需要传给 `T.annotate_layout(...)`。
  - `BufferRegion` 至少需要 2D；如果 `region` 维度超过 2D，前导维度的大小必须都是 1，实际 `swizzle` 作用在最后两个维度上。

## Annotate Layout allow_buffer_region
```python
## 示例 1：标注单个 BufferRegion
KV_shared = T.alloc_shared((2, block_m, block_k), T.float16)
K_region = KV_shared[0, :, :]
T.annotate_layout(
    {K_region: tilelang.layout.make_sqmma_swizzled_layout(K_region, k_major=True)},
    allow_buffer_region=True,
)

## 示例 2：标注 2D slice region
V_shared = T.alloc_shared((block_m, block_n), T.float16)
V_region = V_shared[:, 0:block_n]
T.annotate_layout(
    {V_region: tilelang.layout.make_sqmma_swizzled_layout(V_region, continuity=64, k_major=False)},
    allow_buffer_region=True,
)
```
- 功能：`allow_buffer_region=True` 用于允许 `T.annotate_layout(...)` 的 `key` 使用 `BufferRegion`，例如 `KV_shared[0, :, :]`。典型场景是把一个更高维的 `shared buffer` 切成多个 `2D operand region`，并分别给这些 `region` 标注 `SQMMA swizzled layout`。
- 注意：
  - 默认 `allow_buffer_region=False`，此时 `T.annotate_layout({KV_shared[0, :, :]: layout})` 会报错。

## Annotate Layout allow_reannotation
```python
## 示例：同一个 shared buffer 在不同阶段使用不同 layout
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
- 功能：`allow_reannotation=True` 用于允许同一个 `buffer` 多次调用 `T.annotate_layout(...)`。典型场景是 MUSA 多 `GEMM` kernel 中复用同一块 `shared buffer`：前一个阶段按 `K tile` 的 `layout` 参与 `GEMM`，后一个阶段重新标注成 `V tile` 或其他 `operand` 需要的 `layout`。
- 注意：
  - 只有同一个 `buffer` 需要被重复标注 `layout` 时才需要设置 `allow_reannotation=True`；单次标注不需要。
  - 重新标注按代码顺序生效，新的 `layout` 用于该 `T.annotate_layout(...)` 之后的 `copy` / `GEMM` 等语句。
  - `allow_reannotation=True` 只允许更新 `layout annotation`，不会自动重排 `shared buffer` 中已有数据，通常应放在下一次写入或下一段计算之前。
  - `allow_reannotation` 与 `allow_buffer_region` 可以组合使用。

## 安全访问相关 PassConfig
```python
## 示例
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_COPY_PREDICATION: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION: True,
}
kernel = tilelang.compile(func, target="musa", pass_configs=pass_configs)
```
- 功能：
  - `TL_DISABLE_SAFE_MEMORY_ACCESS` 用于关闭编译器后续自动补上的通用 `global memory` 防越界保护。它主要影响普通 `global memory load/store`，以及部分已经 lower 成 `async copy`、但仍被编译器判断可能越界的访问。
  - `TL_DISABLE_SAFE_COPY_PREDICATION` 用于关闭 `T.copy` lowering 过程中自动生成的安全谓词。它主要影响 `tail tile` / 边界 `tile` 上的 `T.copy`：`source` 越界时自动补 0，`destination` 越界时只在有效范围内写回。
  - `TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION` 用于关闭 MUSA `robust async copy` 最终附带的安全谓词。它只影响编译器生成的 `predicated robust async copy`，使最终 `emission` 去掉这条 `robust async copy` 的 `predicate`。
- 注意：
  - `TL_DISABLE_SAFE_MEMORY_ACCESS` 不会关闭 `T.copy` lowering 过程中自动生成的安全谓词。
  - `TL_DISABLE_SAFE_COPY_PREDICATION` 不会关闭编译器后续自动补上的通用 `global memory` 安全保护。
  - `TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION` 不影响普通 `T.copy`、普通 `global memory load/store`，也不影响普通 `robust load`。
  - 这三个选项默认都是 `False`。打开后只适合在访问范围已经由 `launch geometry`、用户 `guard` 或 `robust descriptor` 明确保证时使用。

## TL_DISABLE_INDEX_TYPE_PROMOTION
```python
## 示例
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
}
kernel = tilelang.compile(func, target="musa", pass_configs=pass_configs)
```
- 功能：`TL_DISABLE_INDEX_TYPE_PROMOTION=True` 用于关闭编译器对 `index` 表达式的自动类型提升。默认情况下，TileLang 会在 `FlattenBuffer`、`ConfigIndexBitwidth` 等 pass 中把可能溢出当前整数位宽的 `index` 表达式提升到更宽的整数类型，例如从 `int32` 提升到 `int64`。
- 注意：
  - 这个选项默认是 `False`，即默认允许编译器做 `index type promotion`。
  - 如果访问范围、`stride` 或 `flattened offset` 可能超过 `int32` 表示范围，开启这个选项可能导致 `index` 溢出和错误访存，这时候需要用户自己做 `int64` 类型提升来保证访问的安全性。
