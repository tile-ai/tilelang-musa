---
title: MUSA 扩展
description: TileLang-MUSA 的 copy、GEMM、layout、atomic 和辅助算子接口
tags: [MUSA, TileLang]
---

## MUSA 扩展

本节集中介绍 `v0.1.12+musa.2` 的 MUSA 扩展。版本差异已在“版本与环境”
章节列出；这里按 copy、GEMM、layout、atomic 和辅助算子组织使用方法。

### 版本差异对照

以下从接口和使用语义角度，对比两个 MUSA SDK 配套正式 release：历史正式版
`v0.1.8+musa.3` 与当前正式版 `v0.1.12+musa.2`。GitHub 开源
release 的差异请以对应 release notes 为准。

| 能力项 | MUSA SDK 配套正式历史版 `v0.1.8+musa.3` | 当前 MUSA SDK 配套正式版 `v0.1.12+musa.2` |
| --- | --- | --- |
| `T.Kernel(producer_threads=...)` | 支持通过 `producer_threads` 指定 warp-specialize 场景下的 producer 线程数 | 保持支持 |
| `T.copy(force_async_copy=True)` | 支持显式请求 global→shared async copy，用于在不使用 TME 的场景下使能 ldlms 路径 | 保持支持，并补充 async copy 触发条件说明 |
| `T.copy(src_robust_desc=desc)` | 支持通过 robust descriptor 描述源地址有效字节范围 | 保持支持，可与 `force_async_copy=True` 组合 |
| 手动 TME barrier | 可通过 `T.copy(..., barrier=barrier)` 指定 barrier | 调整为 `T.tma_copy(..., barrier=barrier)`；load 侧只发起搬运，`T.copy` 不再接受 `barrier=` 参数 |
| TME cache policy hint | 支持 descriptor 形式 TME load/store 的 `inner_cache_policy` / `outer_cache_policy` / `eviction_policy` | 进一步明确 `T.copy(...)` 和 `T.tma_copy(...)` 的适用范围 |
| `T.gemm(...)` lowering | 支持 MP31 `gemm_ss` 按 `SQMMA -> WMMA -> FMA` lowering，以及 `gemm_rr` 的 `WMMA -> FMA` | 完善 MP31 `gemm_ss/gemm_rs/gemm_rr` 选择、QY2 subgroup GEMM，并统一 `TL_DISABLE_SQMMA` / `TL_DISABLE_WMMA` 与 `wg_wait` 控制 |
| SQMMA layout 标注 | 支持 `make_sqmma_swizzled_layout` 配合 `T.annotate_layout` 标注 shared operand layout | 完善 `allow_buffer_region`、`allow_reannotation`、转置 operand layout 和 `continuity` |
| 低精度 GEMM | 主要覆盖 FP16/BF16/FP32、INT8/UINT8，MP31 提供部分 FP8 | 完善 MP31 FP8/TF32、packed vector 和混合 operand 的 dtype、布局与指令选择 |
| Sparse / Reduce / Atomic | 主要使用通用 reduce 与标量 atomic | 增加 MUSA `T.gemm_sp`（QY2 sparse MMA、MP31 sparse FMA）、向量 atomic add，以及 TME atomic add/max/min 与 and/or/xor/inc/dec 归约 |
| Copy view / im2col | 无 MUSA 专用 alias view 和拆分 im2col 文档 | 增加 `T.alias_buffer`，以及面向 NWC/NHWC/NDHWC 的 `T.c2d_im2col_1d/2d/3d` |
| PassConfig | 支持控制 SQMMA/WMMA lowering 和安全访问相关 pass | 增加 `TL_DISABLE_INDEX_TYPE_PROMOTION` 等 index 类型提升控制，并细分 copy/robust copy 安全谓词开关 |
| `accelerated_ops` | `v0.1.8+musa.3` 未单独成文 | 提供 `T.mul_half_float_to_bfloat16_x4`，并支持匹配普通表达式的自动向量化识别 |

### 数据拷贝扩展（Copy）

#### 显式异步拷贝（Async Copy）

`force_async_copy=True` 用于显式请求将 `global -> shared` 的 `T.copy` lowering 成 async copy。使用时源 buffer 必须是 global，目标 buffer 必须是 shared，且源和目标 dtype 保持一致；单条 async copy 的字节数通常需要是 4、8 或 16 字节，可以通过 `T.vectorized(...)` 或连续 `BufferRegion` slice 写法形成对应宽度的 copy。

```python
for v in T.vectorized(4):
    T.copy(src_global[v], dst_shared[v], force_async_copy=True)
```

也可以用连续 slice 一次描述一个 tile：

```python
with T.Kernel(T.ceildiv(N, 128), threads=128) as bx:
    tile = T.alloc_shared((128,), T.float16)
    start = bx * 128
    T.copy(
        src_global[start : start + 128],
        tile[0:128],
        force_async_copy=True,
    )
```

默认 pipeline 会插入所需的提交和等待，使 `T.copy` 保持同步语义；只有在
同时设置 `tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED=True` 和
`tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC=True` 时，才需要在
消费 shared buffer 前手动调用 `T.ptx_commit_group()` 与
`T.ptx_wait_group(N)`。`force_async_copy` 是显式请求而不是无条件保证，
不满足地址空间、dtype 或单条传输宽度约束时编译会失败。

`force_async_copy=True` 只是请求异步 lowering，`T.copy` 仍保持同步可观察
语义。需要完全手动管理异步流水时可使用 `T.async_copy(src, dst)`；它不会
自动插入 `T.ptx_wait_group(...)`，必须在第一次消费 shared buffer 前显式
wait，并根据跨线程消费情况补充 shared-memory 同步。

#### Robust Copy

`src_robust_desc` 用于描述源地址的有效字节范围，适合源侧可能越界的 copy 场景。descriptor 通过 `T.make_robust_desc(addr, size_bytes)` 创建，其中 `addr` 通常由 `T.address_of(...)` 获取。

```python
robust_desc = T.make_robust_desc(T.address_of(src_global[1]), 8)
T.copy(src_global[tid], dst_shared[tid], src_robust_desc=robust_desc)
```

descriptor 的大小也可以在 kernel 中从 device tensor 读取：

```python
with T.Kernel(1, threads=4):
    tid = T.get_thread_binding()
    value = T.alloc_local((1,), T.float32)
    valid_bytes = sizes[0]
    robust_desc = T.make_robust_desc(T.address_of(src[0]), valid_bytes)
    T.copy(src[tid], value, src_robust_desc=robust_desc)
    out[tid] = value[0]
```

例如 `src` 是四个 `float32`，descriptor 从 `src[1]` 开始且大小为 8 bytes
时，读取 `src[1]`、`src[2]` 返回原值，读取范围外的元素返回 0。descriptor
起始地址和大小如果在 kernel 启动前已知，会在 host 构造；如果依赖 device
load 或线程索引，则由编译器放到 device 构造。源 buffer 必须位于 global
memory，且 robust descriptor 可以和 `force_async_copy=True` 组合。

#### 手动 TME barrier

对于能够 lowering 成 MUSA TME load 的搬运，可以手动指定 shared barrier，
并在消费目标 shared buffer 前显式完成 barrier arrive / wait。`v0.1.12+musa.2`
中该能力通过 `T.tma_copy(..., barrier=barrier)` 使用；`T.tma_copy` 的 load
侧只发起数据搬运，不会像普通 `T.copy` 那样自动完成同步，且 `T.copy` 不再
接受 `barrier=` 参数。

对 TME store，`T.tma_copy` 只发起 store 和 arrive，用户可以批量多个 store
后再调用 `T.tma_store_wait()`。

```python
barrier = T.alloc_barrier(128)
T.tma_copy(src[0], tile, barrier=barrier)
T.barrier_arrive(barrier)
T.barrier_wait(barrier, 0)
```

多次搬运可以复用 barrier 数组，并按 tile 或 pipeline stage 选择 barrier：

```python
with T.Kernel(T.ceildiv(N, block_n), threads=128) as bx:
    tile = T.alloc_shared((block_n,), T.float32)
    barriers = T.alloc_barrier([128, 128])
    barrier = barriers[bx % 2]
    T.tma_copy(src[bx * block_n], tile, barrier=barrier)
    T.barrier_arrive(barrier)
    T.barrier_wait(barrier, 0)
```

TME store 可以批量发起，再统一等待：

```python
T.tma_copy(tile_0, output_0)
T.tma_copy(tile_1, output_1)
T.tma_store_wait()
```

`barrier` 必须由 `T.alloc_barrier(...)` 创建。`T.tma_copy` 的 load 侧只
发起 `expect_tx + tma_load`，不会自动等待；在第一次读取目标 shared
buffer 前必须由用户完成对应的 arrive/wait。普通 scalar/SIMT copy 不应
传入该参数。

`T.tma_copy` 还可以用 `leader_scope_threads` 控制 TME leader election 的范围，
例如 `leader_scope_threads=32` 表示每个 warp 选出一个 leader；该值必须是
正整数且为 32 的倍数，省略时使用当前 thread extent。

#### TME 缓存策略提示

descriptor 形式的 MUSA TME load/store 支持 cache policy hint。`inner_cache_policy` 和 `outer_cache_policy` 可选值包括 `"cache_none"`、`"cache_once"`、`"cache_normal"`、`"cache_persist"`；也可以使用兼容写法 `eviction_policy="evict_first"` 等。`eviction_policy` 不能和 `inner_cache_policy` / `outer_cache_policy` 同时设置。

```python
T.copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    inner_cache_policy="cache_once",
    outer_cache_policy="cache_normal",
)

barrier = T.alloc_barrier(128)
T.tma_copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    barrier=barrier,
    inner_cache_policy="cache_once",
    outer_cache_policy="cache_normal",
)
```

`eviction_policy` 的 `evict_normal`、`evict_first`、`evict_last` 分别映射为
`cache_normal`、`cache_once`、`cache_persist` 的 inner/outer 成对策略；省略
hint 时默认使用 `cache_normal`。如果需要分别控制 inner/outer cache，应直接
设置两个 MUSA policy。显式 hint 只对 descriptor 形式的 TME load/store 生效，
普通 scalar copy、1D TME load/store、TME 原子 store 和 im2col 搬运仍使用默认
cache 行为。

```python
# descriptor TME load
T.copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    inner_cache_policy="cache_none",
    outer_cache_policy="cache_persist",
)

# descriptor TME store
T.copy(
    A_shared,
    A_global[0:block_m, 0:block_n],
    inner_cache_policy="cache_once",
    outer_cache_policy="cache_normal",
)

# NV-compatible shorthand
T.copy(
    A_global[0:block_m, 0:block_n],
    A_shared,
    eviction_policy="evict_first",
)
```

### GEMM 扩展

MUSA 后端会根据 operand scope、shape、dtype 和线程布局选择合适的 GEMM lowering 路径。用户始终调用 `T.gemm`，不需要直接选择底层模板：

| operand scope | 形态 | 选择顺序 |
| --- | --- | --- |
| `shared` + `shared` | `gemm_ss` | `SQMMA -> WMMA -> FMA` |
| `fragment` + `shared` | `gemm_rs` | 在满足目标约束时选择对应矩阵指令，否则 FMA |
| `fragment` + `fragment` | `gemm_rr` | `WMMA -> FMA` |

`policy` 控制 warp 划分，常用 `Square`、`FullRow` 和 `FullCol`；
`transpose_A` / `transpose_B` 描述 operand 的逻辑转置，并参与 layout
推导。`C` 必须是 `local.fragment` accumulator。

```python
A_shared = T.alloc_shared((block_M, block_K), dtype)
B_shared = T.alloc_shared((block_K, block_N), dtype)
C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
T.gemm(A_shared, B_shared, C_local, policy=T.GemmWarpPolicy.Square)
```

对于 SQMMA GEMM，可以通过 `wg_wait=-1` 表示发起 SQMMA 后不立即等待。使用该模式时，必须在读取、copy 或继续计算 accumulator 前调用 `T.wait_sqmma()`；`T.wait_sqmma()` 与 `T.wait_wgmma()` 等价。

```python
T.gemm(A_shared, B_shared, C_local, wg_wait=-1)
T.wait_sqmma()
```

`wg_wait=0` 是默认同步行为；`wg_wait=-1` 只表示发起 SQMMA 后暂不等待，
并不改变结果语义。使用后必须在读取、copy 或继续计算 accumulator 前调用
`T.wait_sqmma()`。若设置 `TL_DISABLE_SQMMA=True`，则 shared/shared
路径会跳过 SQMMA；若同时设置 `TL_DISABLE_WMMA=True`，最终使用 FMA。

`gemm_rr` 可以直接使用两个 fragment：

```python
A_fragment = T.alloc_fragment((block_M, block_K), T.float16)
B_fragment = T.alloc_fragment((block_K, block_N), T.float16)
C_local = T.alloc_fragment((block_M, block_N), T.float32)
T.clear(C_local)
T.gemm(A_fragment, B_fragment, C_local)
```

`gemm_rr` 不使用 SQMMA；如果 WMMA 不满足形状或 dtype 约束，会自动回退
到 FMA。建议先查看 `kernel.get_kernel_source()`，确认实际选择的路径。

### SQMMA Layout 标注

`tilelang.layout.make_sqmma_swizzled_layout(buffer, continuity=None, k_major=True)` 用于为 MUSA SQMMA shared operand 生成 swizzled layout，并通过 `T.annotate_layout(...)` 绑定到 shared buffer 或 `BufferRegion`。该接口只生成 layout 描述，不会分配 shared memory，也不会自动搬运或重排 shared memory 中已有数据。

```python
KV_shared = T.alloc_shared((block_m, block_k), T.float16)
T.annotate_layout({
    KV_shared: tilelang.layout.make_sqmma_swizzled_layout(KV_shared, k_major=True)
})
```

当 operand 是转置访问或被拆成多个 instruction tile 时，可以显式设置
`k_major` 和 `continuity`：

```python
KV_shared = T.alloc_shared((2, block_m, block_k), T.float16)
V_region = KV_shared[1, :, :]
T.annotate_layout(
    {
        V_region: tilelang.layout.make_sqmma_swizzled_layout(
            V_region,
            continuity=64,
            k_major=False,
        )
    },
    allow_buffer_region=True,
)
```

`k_major` 应与 `T.gemm` 的转置关系一致：A operand 通常使用
`k_major=not transpose_A`，B operand 通常使用 `k_major=transpose_B`。
`continuity` 用于描述连续维长度，尤其是一个 GEMM tile 需要拆成多个
SQMMA instruction tile 时。

使用 `BufferRegion` 时需要 `allow_buffer_region=True`；region 至少需要
2D，超过 2D 时前导维度必须都是 1。

当同一个 shared buffer 在不同阶段需要重新标注 layout 时，可以使用
`allow_reannotation=True`：

```python
T.annotate_layout(
    {KV_shared: tilelang.layout.make_sqmma_swizzled_layout(KV_shared, k_major=True)},
    allow_reannotation=True,
)
T.gemm(Q_shared, KV_shared, acc_s, transpose_B=True)

T.annotate_layout(
    {KV_shared: tilelang.layout.make_sqmma_swizzled_layout(KV_shared, continuity=64, k_major=False)},
    allow_reannotation=True,
)
T.gemm(S_shared, KV_shared, acc_o)
```

`allow_buffer_region=True` 只允许 layout key 使用 `BufferRegion`，不会改变
buffer rank 或物理数据；`allow_reannotation=True` 也只更新后续语句使用
的 layout annotation，不会自动重排 shared memory 中已有的数据。

### PassConfig 常用选项

以下选项适合在用户已明确保证访问范围或希望控制 lowering 路径时使用：

| 配置项 | 说明 |
| --- | --- |
| `tilelang.PassConfigKey.TL_DISABLE_SQMMA` | 禁用 gemm_ss lowering 到 SQMMA |
| `tilelang.PassConfigKey.TL_DISABLE_WMMA` | 禁用 MUSA WMMA lowering |
| `tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS` | 关闭后续自动补上的通用 global memory 防越界保护 |
| `tilelang.PassConfigKey.TL_DISABLE_SAFE_COPY_PREDICATION` | 关闭 `T.copy` lowering 过程中自动生成的安全谓词 |
| `tilelang.PassConfigKey.TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION` | 关闭 MUSA robust async copy 最终附带的安全谓词 |
| `tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION` | 关闭 index 表达式自动提升到更宽整数类型 |

```python
pass_configs = {
    tilelang.PassConfigKey.TL_DISABLE_SQMMA: True,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_COPY_PREDICATION: True,
}
kernel = tilelang.compile(program, target="musa", pass_configs=pass_configs)
```

安全访问相关开关的边界如下：

- `TL_DISABLE_SAFE_MEMORY_ACCESS` 关闭后续针对普通 global load/store 的通用防越界保护；
- `TL_DISABLE_SAFE_COPY_PREDICATION` 关闭 `T.copy` 在 tail tile 上生成的安全谓词；
- `TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION` 只关闭 robust async copy 的最终谓词；
- `TL_DISABLE_INDEX_TYPE_PROMOTION` 阻止 `FlattenBuffer` 等 pass 将可能溢出的 index 自动提升到更宽整数类型。

前三个安全访问开关互相独立：关闭通用 safe-memory pass 不会关闭 `T.copy`
谓词，关闭 `T.copy` 谓词也不会关闭通用 global load/store 保护；robust-copy
开关只影响编译器生成的 robust async-copy 谓词。`TL_DISABLE_INDEX_TYPE_PROMOTION`
也不会自动把表达式转换成 `int64`，访问可能超过 `int32` 范围时需要用户自行
提升 index 类型。

这些开关默认均为 `False`。如果 kernel 依赖 launch geometry、显式 guard
或 robust descriptor 保证访问范围，可以关闭不需要的保护以减少代码；否则
应保持默认值。需要切换 GEMM 路径时，优先只设置一个 lowering 开关，便于
比较生成代码和结果差异。

:::caution
关闭安全访问相关 PassConfig 后，如果访问范围、tail tile 或 flattened offset 没有由用户 guard、launch geometry 或 robust descriptor 保证，可能导致越界访问或错误结果。
:::

### 加速算子（Accelerated Ops）

`accelerated_ops` 用于暴露 MUSA target 上可以高效执行的计算 pattern，定位更接近“显式请求使用某类硬件友好的 fused/accelerated 计算形式”。例如 `T.mul_half_float_to_bfloat16_x4(x, y)` 表示将 `float16x4 * float32x4` 的乘法结果转换为 `bfloat16x4`。普通表达式 `T.Cast("bfloat16", A[i] * B[i])` 在满足 pattern 和类型约束时，也可能由编译器自动识别并优化生成对应操作。

该接口在 `v0.1.12+musa.2` 中作为 MUSA accelerated ops 能力提供。显式调用时需要传入 4-lane vector 表达式，推荐使用 slice 或 `T.Ramp` 写法；`lhs` 必须是 `float16x4`，`rhs` 必须是 `float32x4`，返回值为 `bfloat16x4`。

```python
C[offset : offset + 4] = T.mul_half_float_to_bfloat16_x4(
    A[offset : offset + 4],
    B[offset : offset + 4],
)
```

等价的 `T.Ramp` 写法如下：

```python
lanes = T.Ramp(offset, 1, 4)
C[lanes] = T.mul_half_float_to_bfloat16_x4(A[lanes], B[lanes])
```

满足相同 pattern 时，普通逐元素表达式也可能被编译器自动识别：

```python
for i in T.Parallel(4):
    C[i] = T.Cast("bfloat16", A[i] * B[i])
```

当 `float32` 输入是标量 broadcast 时，也可以匹配自动优化 pattern：

```python
for i in T.Parallel(16):
    C[i] = T.Cast("bfloat16", A[i] * Scale[0])
```

显式调用要求 `lhs` 为 `float16x4`、`rhs` 为 `float32x4`，返回值为
`bfloat16x4`；slice 或 `T.Ramp` 的 lane 数不是 4 时会导致编译失败。

### TME 原子操作与归约

普通的 `T.atomic_add` 支持标量写法；连续的 `T.atomic_add` 位于
`T.Parallel` 中时，编译器会尝试自动向量化：

```python
# 标量 atomic add：所有线程累加到同一个元素。
with T.Kernel(threads=128):
    tx = T.get_thread_binding()
    T.atomic_add(B[0], A[tx])

# 连续访问：编译器可合并为 x2/x4 atomic。
with T.Kernel(threads=1):
    for i in T.Parallel(4):
        T.atomic_add(B[i], A_local[i])
```

需要明确控制宽度时，可以使用 `T.atomic_addx2(dst, value)` 或
`T.atomic_addx4(dst, value)`。它们从给定起始位置处理连续 2/4 个元素；
调用方必须保证源、目标范围完整且满足目标向量类型的自然对齐。无法满足
向量化条件时会回退为逐元素标量 atomic，保证结果语义不变。

在支持该 lowering 的 MUSA target 上，TileLang-MUSA 可以在 tile-region
原子操作中显式请求 TME 路径：

```python
T.atomic_add(out_global, tile_shared, use_tma=True)
T.atomic_max(out_global, tile_shared, use_tma=True)
T.atomic_min(out_global, tile_shared, use_tma=True)
```

源 tile 通常位于 shared memory，目标 tile 位于 global memory；源、目标
region 的 rank 和 tile extent 必须匹配，目标 backing buffer 可以大于本次
归约 tile。编译器会生成 descriptor 形式的 TME store，并使用
`T.tma_store_wait()` 等待已提交的 store。除 add/max/min 外，
还提供 `T.atomic_and`、`T.atomic_or`、`T.atomic_xor`、`T.atomic_inc` 和
`T.atomic_dec` 的 tile-region 归约；位运算要求 32/64-bit 整数，inc/dec
要求 `uint32`。不满足目标或 dtype 约束时应改用普通标量 atomic 路径。

对于连续的 `T.atomic_add`，编译器可以根据连续性、对齐和 dtype 自动合并
为向量原子操作；也可以用 `T.atomic_addx2` / `T.atomic_addx4` 显式指定
向量宽度。向量原子要求 global 目标和自然对齐，条件不满足时会回退为
逐元素标量操作。

标量 `atomic_add`、`atomic_max` 和 `atomic_min` 还支持可选的
`memory_order` 与 `return_prev` 参数：

```python
prev = T.atomic_add(counter[0], value, memory_order="acq_rel", return_prev=True)
T.atomic_max(counter[0], value, memory_order="relaxed")
```

`memory_order` 可取 `relaxed`、`consume`、`acquire`、`release`、`acq_rel`
或 `seq_cst`。`return_prev=True` 只适用于标量原子路径；tile-region 原子
归约不支持返回旧值。`use_tma=True` 仅对支持的 tile-region lowering 生效，
不应和标量返回值语义混用。

### 别名视图（Alias View）

`T.alias_buffer(src, shape, strides=None, elem_offset=0)` 为已有 `Buffer`
创建不分配新内存的 strided view。alias 与源 buffer 共享 data、dtype 和
storage scope，适合将同一块 global 或 shared storage 解释为 TME 所需的
不同 rank/stride 视图。

```python
src_view = T.alias_buffer(
    src,
    shape=(4, 8),
    strides=(4, 1),
    elem_offset=3,
)

for i, j in T.Parallel(4, 8):
    dst[i, j] = src_view[i, j]
```

也可以把一维 backing storage 解释成 TME 所需的高阶 view：

```python
src_view = T.alias_buffer(
    src,
    shape=(panels, rows, groups, panel_cols),
    strides=(panel_cols, cols, groups * cols, 1),
)
shared_view = T.alias_buffer(
    shared,
    shape=(groups, groups, panel_cols),
    strides=(groups * panel_cols, panel_cols, 1),
)

T.tma_copy(
    src_view[panel, row_base : row_base + groups, 0:groups, 0:panel_cols],
    shared_view[:, :, :],
    barrier=bar,
)
```

`strides` 的单位是元素而不是字节，`elem_offset` 是相对元素偏移。该
接口只改变访问视图，不搬运、重排或复制数据，也不会自动触发 TME；用户
仍需保证 alias 的所有访问都落在 backing storage 范围内，并为实际
shared allocation 保留正确的 layout annotation。

与 `T.reshape` / `T.view` 相比，`T.alias_buffer` 允许自定义 stride、重叠
访问以及与源 buffer 不同的逻辑元素数量；它不会改变 dtype，也不会修改
源 buffer 的 layout annotation。

### 快速整数除法（Fast Integer Division）

`T.fast_div`、`T.fast_mod` 和 `T.fast_divmod` 用于显式选择整数除法和
取余的乘法/移位实现；普通的 `//` 和 `%` 不会被自动替换。

```python
# 只需要商或余数时分别使用 fast_div / fast_mod。
quotient = T.fast_div(dividend, divisor)
remainder = T.fast_mod(dividend, divisor)

# 同时需要两者时使用 fast_divmod，避免重复计算商。
q, r = T.fast_divmod(index, T.int32(7))
```

参数必须是 scalar `int32`，`dividend` 非负且 `divisor` 大于 0。divisor
为编译期常量时，加速参数在编译期生成；divisor 是 kernel 参数或 shape
表达式时，在每次 launch 前由 host 生成；divisor 依赖 device load 或
thread index 时，则在 device 侧生成。用户不需要指定构造位置。

这组接口只在 MUSA target 上可用，不会自动替换已有的 `//` 或 `%`。它
必须成功生成乘法/移位形式；不支持的表达式会在编译期报错，不会静默
回退为普通除法。若同一 divisor 只使用一次，fast division 的收益可能
被加速参数构造成本抵消，应结合生成代码和 profiler 判断。

### 稀疏 GEMM（Sparse GEMM）

`T.gemm_sp` 用于 2:4 sparse GEMM，除 A/B operand 和 C accumulator 外，
还需要传入保存非零元素位置的 metadata buffer。MUSA 路径会根据目标选择
QY2 sparse MMA 或 MP31 sparse FMA；A、metadata、B 通常放在 shared memory，
C 放在 `local.fragment`，metadata 使用自然 row-major 排布且无需额外
layout annotation。

压缩输入可使用 `tilelang.utils.sparse.compress` 生成：

```python
from tilelang.utils.sparse import compress, get_e_factor

A_sparse, metadata = compress(A)  # FP16/BF16 默认使用 int16 metadata
# 对 int8/FP8 等类型，metadata 通常使用 int32。
e_factor = get_e_factor(in_dtype, metadata_dtype)
```

核心调用形式如下（`A_sparse` 只保存每四个元素中的两个非零值）：

```python
A_shared = T.alloc_shared((block_M, block_K // 2), in_dtype)
E_shared = T.alloc_shared((block_M, block_K // e_factor), metadata_dtype)
B_shared = T.alloc_shared((block_K, block_N), in_dtype)
C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)
T.clear(C_frag)

T.copy(A_sparse[by * block_M, ko * (block_K // 2)], A_shared)
T.copy(metadata[by * block_M, ko * (block_K // e_factor)], E_shared)
T.copy(B[ko * block_K, bx * block_N], B_shared)
T.gemm_sp(
    A_shared,
    E_shared,
    B_shared,
    C_frag,
    transpose_A=False,
    transpose_E=False,
    transpose_B=False,
    policy=T.GemmWarpPolicy.Square,
)
```

FP16/BF16 常用 `int16` metadata，E-factor 为 16；INT8/FP8 常用 `int32`
metadata，E-factor 为 32。`T.gemm_sp` 默认是同步接口；
`transpose_A`、`transpose_E`、`transpose_B` 可分别描述压缩矩阵、metadata
和 dense B 的逻辑转置。压缩数据和 metadata 必须保持同一 2:4 分组，K
维也应按对应 sparse tile 对齐。

### Im2col 展开

`T.c2d_im2col_1d`、`T.c2d_im2col_2d` 和 `T.c2d_im2col_3d` 将 NWC、NHWC
或 NDHWC 输入展开到二维 shared tile，输出 tile 形状为 `(block_M, block_K)`。
逻辑上，im2col 矩阵的行数为 `N * product(output_spatial_dims)`，列数为
`product(kernel_spatial_extents) * C`。

```python
# 1D：NWC
T.c2d_im2col_1d(
    img, col, coord_c, coord_w, coord_n, dim_c, dim_nw,
    offset_s, output_w, kernel, stride, dilation, pad,
)

# 2D：NHWC
T.c2d_im2col_2d(
    img, col,
    coord_c, coord_w, coord_h, coord_n,
    dim_c, dim_nhw,
    offset_h, offset_w,
    R, S, output_w, output_h,
    stride_w, stride_h,
    dilation_w, dilation_h,
    pad_w, pad_h,
)

# 3D：NDHWC
T.c2d_im2col_3d(
    img, col, coord_c, coord_w, coord_h, coord_d, coord_n,
    dim_c, dim_nhwd,
    offset_d, offset_h, offset_w,
    T_kernel, R, S, output_w, output_h, output_d,
    stride_w, stride_h, stride_d,
    dilation_w, dilation_h, dilation_d,
    pad_w, pad_h, pad_d,
)
```

MUSA TME 快路径要求已证明的通道搬运宽度满足
`(dim_c * dtype_bytes) % 16 == 0`；目标 layout 为 SQMMA-compatible 时，
lowering 还可能按 SQMMA instruction tile 拆分一次逻辑搬运。无法证明
descriptor/layout 条件时，会回退到逐元素 scalar 实现。fallback 仍按同一
im2col 映射计算，并对越界输入写入 0。

`barrier` 是可选的手动 TME barrier；省略时，快路径可以由
`InjectTmaBarrier` 等同步 pass 自动接入 barrier。显式传入后，必须在消费
shared tile 前完成对应 arrive/wait；fallback 不依赖 TME barrier。

### 语言与同步辅助

`v0.1.12+musa.2` 同步了多项常用的 MUSA 语言接口：

- `T.transpose(src, dst)` 对 2D shared tile 做转置，语义为
  `dst[j, i] = src[i, j]`；
- `T.any_sync`、`T.all_sync`、`T.ballot_sync`、`T.ballot`、`T.activemask`
  提供 warp vote/ballot，predicate 是第一个参数，mask 缺省为 `0xFFFFFFFF`；
  MUSA 的 32-bit ballot 会以 `uint64` 返回；
- `T.match_any_sync(value[, mask])` 返回与当前 lane 值相同的 lane mask，
  `T.match_all_sync(value[, mask])` 在所有 lane 值相同时返回 mask，否则返回 0；
- `T.shfl_sync`、`T.shfl_xor`、`T.shfl_down`、`T.shfl_up` 用于 warp 内
  数据交换，`T.sync_warp`、`T.syncthreads_count/and/or` 用于带谓词同步；
- `T.ieee_add`、`T.ieee_sub`、`T.ieee_mul`、`T.ieee_fmaf` 提供可配置
  舍入的 IEEE 运算，适合对数值边界有明确要求的 kernel。

```python
with T.Kernel(1, threads=128):
    tid = T.get_thread_binding()
    tile_in = T.alloc_shared((16, 16), T.float16)
    tile_out = T.alloc_shared((16, 16), T.float16)
    # tile_in 由前序 copy 填充后，转置到 tile_out。
    T.transpose(tile_in, tile_out)

    active = tid < valid_threads
    active_mask = T.ballot_sync(active)
    if T.any_sync(active):
        value = T.shfl_down(T.float32(tid), 1, mask=active_mask)
```

warp vote/shuffle 的 `mask` 应覆盖实际参与的 lane；跨 warp 或跨线程块的
数据交换仍需使用 shared memory 和 block/grid 级同步。
