---
title: 官方编程接口
description: TileLang-MUSA 官方 DSL、类型系统、控制流和核心算子接口
tags: [MUSA, TileLang]
---

## 官方编程接口

TileLang 是嵌入 Python 的 TIR DSL。kernel 内可以使用 TileLang 提供的
类型、循环、内存和算子接口；host 侧仍然使用普通 Python 组织参数、编译和
运行时调用。本文介绍跨后端通用的官方接口，MUSA 专属的 TME、SQMMA、
robust copy 和硬件 layout 见 [MUSA 扩展](./05_musa_extensions.md)。

:::tip TileLang 官方参考
本页是 MUSA SDK 环境下的使用摘要。完整的跨后端接口、参数签名和最新
示例请直接跳转到 [TileLang 官方文档](https://tilelang.com/)：

- [官方编程指南总览](https://tilelang.com/programming_guides/overview.html)
- [Language Basics](https://tilelang.com/programming_guides/language_basics.html)
- [Instructions Reference](https://tilelang.com/programming_guides/instructions.html)
- [官方 API Reference](https://tilelang.com/autoapi/tilelang/index.html)

TileLang-MUSA 的 `T.gemm`、TME、SQMMA、MUSA target 和架构限制以本目录中
的 MUSA 文档和已安装版本为准；通用 API 的完整签名以上述官方 Reference
为准。
:::

### 导入与 kernel 定义

常用导入方式如下：

```python
import tilelang
import tilelang.language as T
from tilelang import jit
```

使用 `@T.prim_func` 定义 TIR kernel。参数用 `T.Tensor` 或 `T.Buffer` 标注
形状和 dtype；形状可以是常量，也可以是符号表达式。

```python
N = T.dynamic("N", "int32")

@T.prim_func
def add_kernel(
    A: T.Tensor((N,), "float32"),
    B: T.Tensor((N,), "float32"),
    C: T.Tensor((N,), "float32"),
):
    for i in T.serial(N):
        C[i] = A[i] + B[i]
```

需要在类型标注中引用名称时，也可以使用 `T.dyn["N"]`。`T.dynamic(name,
dtype)` 创建可以在 kernel 体内直接使用的 TIR 变量；`T.symbolic` 是旧的
兼容别名，新代码应优先使用 `T.dynamic`。

### JIT、编译和 target

`@tilelang.jit` 适合把 Python 参数、tile size 和 dtype 封装成可调用 kernel；
`tilelang.compile` 适合显式查看 TIR、选择 execution backend 或传入 pass
config。

```python
@tilelang.jit
def vector_add(A, B, block=256, dtype="float32"):
    N = T.dynamic("N", "int32")
    A: T.Tensor((N,), dtype)
    B: T.Tensor((N,), dtype)
    C = T.empty((N,), dtype)

    with T.Kernel(T.ceildiv(N, block), threads=block) as bx:
        for i in T.Parallel(block):
            index = bx * block + i
            C[index] = A[index] + B[index]
    return C

# 也可以显式编译一个 prim_func。
kernel = tilelang.compile(
    add_kernel,
    target="musa",
    execution_backend="cython",
)
```

常用 target 形式包括字符串和配置字典。MUSA 目标可以使用
`target="musa"`；需要指定硬件架构时使用与当前 SDK 匹配的配置。实际可用
架构和 dtype 由安装的 MUSA SDK、设备和 TileLang-MUSA 版本共同决定。

### `T.Kernel` 与线程绑定

`with T.Kernel(...)` 声明 kernel 的 grid 和 block。传入两个 grid 维度时，
上下文会产生 `bx`、`by` 等 block 绑定；`threads` 指定每个 block 的线程数。

```python
with T.Kernel(
    T.ceildiv(N, block),
    T.ceildiv(M, block),
    threads=256,
) as (bx, by):
    tx = T.get_thread_binding()
    row = by * block + tx
    ...
```

大多数 kernel 应优先使用 `T.Parallel`、`T.Pipelined` 等结构化循环，而不是
手写线程索引。需要底层控制时，可以使用 `T.get_thread_binding()`、
`T.get_block_binding()` 和 `T.get_lane_idx()`。

### 类型系统

同一个 dtype 可以用字符串、TileLang dtype 对象或框架 dtype 表示，编译器会
将它们规范化：

```python
T.Tensor((M, N), "float16")
T.Tensor((M, N), T.float16)
T.Tensor((M, N), torch.float16)
```

常用标量类型包括：

| 类别 | 类型 |
| --- | --- |
| 布尔 | `bool` |
| 有符号整数 | `int8`、`int16`、`int32`、`int64` |
| 无符号整数 | `uint8`、`uint16`、`uint32`、`uint64` |
| 浮点 | `float16`、`bfloat16`、`float32`、`float64` |
| 低精度浮点 | `float8_*`、`float6_*`、`float4_*` |

许多基础 dtype 还有 `x2`、`x4`、`x8` 等 packed vector 形式，例如
`float16x4`、`int8x4`。低精度和 packed dtype 是否能使用，取决于目标架构
和具体 lowering；混合精度计算应显式指定 accumulator dtype，例如用
`float16` 输入配合 `float32` fragment accumulator。

### 内存作用域与分配

TileLang 常用的内存作用域如下：

- `global`：kernel 参数默认所在的设备内存；
- `shared.dyn`：block 内共享的片上内存，由 `T.alloc_shared` 分配；
- `local.fragment`：线程/warp 使用的 fragment，由 `T.alloc_fragment` 分配；
- `local.var`：单个标量或小型局部 buffer，由 `T.alloc_var` 分配；
- barrier：异步搬运和 producer/consumer pipeline 使用的 barrier。

```python
A_shared = T.alloc_shared((BM, BK), "float16")
B_shared = T.alloc_shared((BK, BN), "float16")
C_fragment = T.alloc_fragment((BM, BN), "float32")
local_value = T.alloc_var("int32", init=0)
barrier = T.alloc_barrier(128)
T.clear(C_fragment)
```

`T.empty(shape, dtype)` 用于声明输出 tensor；`T.fill(buffer, value)` 和
`T.clear(buffer)` 用于初始化 buffer。不要把 fragment 当作普通 global
tensor 使用，fragment 的布局由目标和 GEMM/reduce lowering 决定。

### 数据访问与搬运

`T.copy(src, dst)` 接受 buffer、buffer load 或 buffer region，通常从目标
buffer 推导搬运范围：

```python
# Global -> Shared
T.copy(A[by * BM, ko * BK], A_shared)
T.copy(B[ko * BK, bx * BN], B_shared)

# Fragment -> Global
T.copy(C_fragment, C[by * BM, bx * BN])
```

普通 `T.copy` 的可观察语义是同步的：语句完成后可以安全读取目标。编译器
可以根据作用域、连续性和目标选择 SIMT、TMA、LDSM 或其他搬运路径，并在
需要时自动插入提交、等待和 shared-memory 同步。

需要完全手动管理 global→shared 异步流水时，使用 `T.async_copy(src, dst)`：

```python
T.async_copy(A[by * BM, ko * BK], A_shared)
# ... 与搬运无关的计算 ...
T.ptx_wait_group(0)
T.gemm(A_shared, B_shared, C_fragment)
```

`T.async_copy` 不自动插入 `T.ptx_wait_group`，必须在第一次消费目标前完成
wait；跨线程生产和消费时还需要 shared-memory 同步。MUSA 上的
`force_async_copy`、TME barrier 和 robust descriptor 见
[MUSA 扩展](./05_musa_extensions.md)。

### 控制流和循环

kernel 内支持标准条件语句、循环和 `break`/`continue`。条件应为 TIR 表达式；
Python 常量条件会在编译期折叠。

```python
if T.all_of(i < M, j < N):
    C[i, j] = A[i, j] + B[i, j]

for i in T.serial(0, N, 2):
    ...

for k in T.unroll(K_TILE):
    ...

for i, j in T.Parallel(M, N):
    C[i, j] = A[i, j] + B[i, j]

for ko in T.Pipelined(T.ceildiv(K, BK), num_stages=3):
    T.copy(A[by * BM, ko * BK], A_shared)
    T.gemm(A_shared, B_shared, C_fragment)
```

循环接口的语义如下：

- `T.serial`：普通串行循环；
- `T.unroll`：请求对小 trip count 循环展开；
- `T.Parallel`：嵌套并行循环，适合 elementwise 和连续搬运；
- `T.Pipelined`：重叠搬运和计算的 software pipeline；
- `T.Persistent`：高级 persistent thread-block 循环，仅适用于专用模板；
- `while`：条件必须是可 lower 到 TIR 的表达式。

Tile size 不能整除问题规模时，可以显式写边界 guard：

```python
for i in T.Parallel(BLOCK):
    index = bx * BLOCK + i
    if index < N:
        C[index] = A[index] + B[index]
```

简单的越界访问也可能由 `LegalizeSafeMemoryAccess` 自动插入 guard；如果
关闭安全访问 pass，必须由 launch geometry 或用户 guard 保证范围安全。

### 计算、归约和视图

核心计算接口包括：

```python
T.gemm(A_shared, B_shared, C_fragment)
T.gemm_sp(A_sparse, metadata, B_shared, C_fragment)

value = T.exp(x) + T.rsqrt(y)
T.reduce_sum(src, dst, dim=0)
T.reduce_max(src, dst, dim=1)
T.reduce_min(src, dst, dim=1)
T.cumsum(src, dst, dim=0)
T.cummax(src, dst, dim=0)
```

`T.gemm` 的 A/B/C 作用域、transpose、warp policy 和 MUSA lowering 由目标
决定；`T.gemm_sp` 需要 2:4 compressed operand 和 metadata。归约通常需要
先分配并初始化 fragment 或 reducer；结果布局必须符合后续 copy 或消费算子
的要求。

`T.reshape` 和 `T.view` 创建共享底层 storage 的逻辑视图，不会自动搬运数据：

```python
view = T.reshape(buffer, (M, N))
view2 = T.view(buffer, shape=(M, N))
```

需要自定义 stride、offset 或 TME rank view 时使用 MUSA 专属的
`T.alias_buffer`，不要把 `reshape/view` 当作 arbitrary strided alias。

### 同步、warp 操作与诊断

高层 pipeline 通常由 pass 自动安排 producer/consumer 顺序。需要显式同步
时可使用：

- `T.sync_threads([barrier_id, arrive_count])`：block-wide barrier；
- `T.sync_warp([mask])`：warp-wide barrier；
- `T.mbarrier_arrive`、`T.mbarrier_wait_parity`：异步搬运 barrier；
- `T.syncthreads_count/and/or`：带 predicate 的 block 同步；
- `T.any_sync`、`T.all_sync`、`T.ballot_sync`：warp vote；
- `T.shfl_sync`、`T.shfl_down`、`T.shfl_up`、`T.shfl_xor`：warp 数据交换。

MUSA 的 warp-match、TME barrier 和完整同步示例见
[MUSA 扩展](./05_musa_extensions.md)。

调试接口包括：

```python
T.print(C_fragment, msg="accumulator:")
T.device_assert(index < N, msg="index out of range")
```

`T.print` 用于 kernel 内查看标量或 buffer；`T.device_assert` 用于在设备端
验证索引和中间结果。更完整的源码、IR 和 MUSA C code 查看方法见
[调试诊断](./07_debug_diagnostics.md)。

### 标注、软件流水线和自动调优

对于普通 GEMM，`T.Pipelined(..., num_stages=...)` 足以让编译器推导
producer/consumer pipeline。需要手动安排 effectful 语句时，可以使用
`stage` 和 `order`：

```python
for ko in T.Pipelined(
    T.ceildiv(K, BK),
    stage=[0, 0, 1],
    order=[0, 1, 2],
):
    T.copy(A[ko * BK], A_shared)
    T.copy(B[ko * BK], B_shared)
    T.gemm(A_shared, B_shared, C_fragment)
```

`stage`/`order` 只应对应 copy、GEMM、reduce、store、atomic 和显式 wait 等
有实际效果的语句；`base = ko * BK` 这类可重放的 scalar bind 不占 annotation
槽位。详见 [Software Pipeline Annotations](https://tilelang.com/programming_guides/software_pipeline.html)。

常用编译标注包括：

- `T.use_swizzle(panel_size=..., enable=True)`：提示 block rasterization；
- `T.annotate_layout({...})`：给 buffer 或 fragment 附加显式 layout；
- `T.annotate_safe_value(value, ...)`：向安全访问分析提供值域信息；
- `T.annotate_l2_hit_ratio(buffer, ratio)`：提供 cache 行为提示；
- `loop_layout=`：给 `T.Parallel` 的嵌套循环附加 Fragment layout。

TileLang 内置 autotuner，可以搜索 tile size、pipeline stage、线程数等
编译期参数：

```python
@tilelang.autotune(
    configs=lambda M, N, K: [
        {"block_M": 64, "block_N": 128, "block_K": 32, "threads": 128},
        {"block_M": 128, "block_N": 128, "block_K": 64, "threads": 256},
    ],
    warmup=25,
    rep=100,
)
@tilelang.jit(out_idx=[-1])
def tuned_kernel(M, N, K, block_M=128, block_N=128, block_K=32, threads=128):
    ...
```

配置项的 key 必须对应 kernel factory 的参数；正式比较前应固定输入、参考
结果、warmup/repeat 和 target。详见 [官方 Autotuning 指南](https://tilelang.com/programming_guides/autotuning.html)。

### Python 兼容边界

TileLang 是 Python 嵌入式 DSL，kernel 内并非所有 Python 语法都可用：

| Python 写法 | 支持情况 | 推荐写法或说明 |
| --- | --- | --- |
| `range(n)`、`range(a, b, step)` | ✅ | 优先写成 `T.serial(...)`，语义更明确 |
| `for x in list`、`enumerate`、`zip` | ❌ | 使用索引循环 |
| `if` / `elif` / `else` | ✅ | 条件应为 TIR 表达式 |
| `while` | ✅ | 条件必须可 lower 到 TIR |
| `break` / `continue` | ✅ | 用于受支持的循环 |
| 多维索引和 slice | ✅ | 生成 BufferLoad/BufferRegion |
| `+`、`-`、`*`、`/`、`%`、增强赋值 | ✅ | 生成设备端算术 |
| `a = b = c` | ❌ | 分开赋值 |
| kernel 内定义普通 Python 函数或 class | ❌ | 使用 `@T.macro` 定义可内联代码块 |
| `len`、`type`、`isinstance` | ❌ | 使用 `buffer.shape` 和编译期信息 |
| `print()` | ⚠️ | kernel 内使用 `T.print()` |

### 不支持语法

TileLang-MUSA 不支持 TileLang 中与 NVIDIA 硬件特性强相关的 IR，例如：

```python
alloc_tmem
alloc_descriptor
alloc_wgmma_desc
alloc_tcgen05_smem_desc
alloc_tcgen05_instr_desc
```

TileLang-MUSA 也不支持以下 IR：

```python
set_max_nreg
inc_max_nreg
dec_max_nreg
no_set_max_nreg
annotate_producer_reg_dealloc
annotate_consumer_reg_alloc
disable_warp_group_reg_alloc
```

不同 MUSA 架构对低精度、warp-specialized pipeline、TME 和矩阵指令的支持
不同。遇到编译失败时，先确认 target arch、dtype、shape、scope 和 pass
config，再查看生成的 TIR/MUSA C 代码。
