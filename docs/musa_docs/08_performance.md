---
title: 性能分析
description: TileLang-MUSA 的正确性基线、Profiler、Autotune 和性能回归分析
tags: [MUSA, TileLang]
---

## 性能分析

性能分析的目标不是只得到一个 latency 数字，而是确认：

- kernel 是否走到了预期的 MUSA lowering；
- 测量是否排除了首次编译、输入分配和同步开销；
- latency、有效带宽、计算吞吐和资源占用是否互相匹配；
- 调整 block、pipeline、layout 或 dtype 后是否真的改善了目标 workload。

建议遵循“正确性 → 生成代码 → 稳定测量 → 受限搜索 → 回归记录”的顺序。

### 1. 建立正确性和测量基线

性能测试前先固定以下条件：

| 项目 | 要求 |
| --- | --- |
| 设备 | 固定 MUSA GPU 和 `MUSA_VISIBLE_DEVICES` |
| 软件 | 记录 torch、torch_musa、TileLang-MUSA、TVM FFI、MUSA SDK 和 driver |
| 输入 | 固定 shape、dtype、stride、随机种子和数据分布 |
| 编译 | 单独完成 `.compile(...)`，不要把首次编译时间计入 latency |
| 校验 | 与 PyTorch/reference 对比，明确 `rtol`/`atol` |
| 测量 | 固定 backend、warmup、repeat、quantiles 和 return mode |

输入 tensor 应在计时前创建并放到 MUSA：

```python
torch.manual_seed(0)
A = torch.randn((M, K), dtype=torch.float16, device="musa")
B = torch.randn((K, N), dtype=torch.float16, device="musa")

kernel = matmul.compile(
    M=M,
    N=N,
    K=K,
    block_M=128,
    block_N=128,
    block_K=32,
)
```

### 2. 使用 TileLang Profiler

`kernel.get_profiler()` 会复用 kernel 的输入生成和调用适配器。MUSA 上建议
优先使用 `event` backend；`cupti`、`cudagraph` 等 backend 是否可用取决于
当前 torch_musa、工具链和 profiler 构建。

```python
profiler = kernel.get_profiler()
latency_ms = profiler.do_bench(
    n_warmup=20,
    n_repeat=100,
    backend="event",
    return_mode="mean",
)
print(f"mean latency: {latency_ms:.3f} ms")
```

### Benchmark 目录中的可复现实例

仓库 benchmark 不是通用 microbenchmark 的替代品，而是带有固定模型 shape、
reference 校验和结果记录的端到端样例。当前 [MP31 modelops benchmark](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/modelops/) 中最适合作为
入门的是 `example_mha_fwd_bhsd`：它包含两个 `T.gemm`、online softmax、
pipeline copy 和 MUSA profiler 输出。

同一 benchmark 树中还可以按场景选择：

| 场景 | Benchmark 入口 |
| --- | --- |
| FlashAttention/MHA | [MP31 modelops](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/modelops/) |
| MLA | [MP31 modelops ops](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/modelops/ops/) |
| GDN / Linear Attention | [MP31 MATE ops](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/mate/ops/) |
| MoE | [MP31 tilekernels/moe](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/tilekernels/moe/) |

```bash
# 从 tilelang_musa 仓库根目录运行
python benchmark/mp31/runner.py \
  --source modelops \
  --cases example_mha_fwd_bhsd \
  --samples 3
```

若当前构建不是 Release，需要显式加入 `--allow-non-release-build`；发布性能
数据应使用 Release build。该 benchmark 的默认参数为 `B=2,H=28,seq_q=8192,
seq_kv=8192,D=128`，kernel 配置为 `block_M=256, block_N=128, threads=512`。
每条记录包含 `time_us`、`bandwidth_gbs`、`flops`、`tflops` 和完整参数，适合
保存成 JSONL 后做回归比较。仓库中的 [modelops baseline](https://github.com/tile-ai/tilelang-musa/blob/v0.1.12%2Bmusa.1/benchmark/mp31/baselines/modelops.jsonl) 保存了对应 source 的基线记录。

需要做回归检查时：

```bash
python benchmark/mp31/runner.py \
  --source modelops \
  --cases example_mha_fwd_bhsd \
  --samples 3 \
  --check-regression \
  --threshold 0.05
```

单独观察 benchmark 的 reference 和 TileLang 结果时，可直接运行：

```bash
python benchmark/mp31/modelops/ops/example_mha_fwd_bhsd_benchmark.py \
  --batch 2 --heads 28 --seq_q 8192 --seq_kv 8192 --dim 128 \
  --verbose
```

这个例子用于说明“融合 Attention 的两个 GEMM”如何测量，不能替代独立 dense
GEMM microbenchmark；独立 GEMM 应使用本页的 `matmul` kernel 固定 M/N/K 后
单独测量。

`do_bench` 的重要参数：

| 参数 | 作用 |
| --- | --- |
| `warmup` | 以毫秒为单位的自动 warmup 预算 |
| `rep` | 以毫秒为单位的自动测量预算 |
| `n_warmup` | 固定 warmup 迭代次数，非零时覆盖自动预算 |
| `n_repeat` | 固定测量迭代次数，非零时覆盖自动预算 |
| `backend` | `event`、`cupti` 或 `cudagraph`，需确认设备支持 |
| `quantiles` | 返回指定分位数，例如 `[0.5, 0.95, 0.99]` |
| `return_mode` | `min`、`max`、`mean` 或 `median` |
| `input_tensors` | 使用已准备好的输入，避免 profiler 重复分配 |

需要观察尾延迟时：

```python
p50, p95, p99 = profiler.do_bench(
    n_warmup=20,
    n_repeat=200,
    backend="event",
    quantiles=[0.5, 0.95, 0.99],
    return_mode="mean",
)
print(f"p50={p50:.3f} ms, p95={p95:.3f} ms, p99={p99:.3f} ms")
```

不同测量模式不要混在同一张表中；报告中应同时记录输入规模和 return mode。

### 3. 从生成代码解释性能

同样的 Python DSL 可能由于 shape、dtype、scope 和 target 不同而选择不同
lowering。每次性能变化都应保存生成 source：

```python
source = kernel.get_kernel_source()
print(source)
```

检查以下信号：

- GEMM 是否出现预期的 SQMMA、WMMA、MMA 或 FMA 路径；
- copy 是否变成 TME、async copy、LDLMS 或普通 global/shared load/store；
- `wg_wait`、mbarrier arrive/wait、pipeline stage 是否和依赖关系匹配；
- block/grid 是否产生足够并行度，是否因为 tail guard 引入大量分支；
- shared memory、register 和 fragment layout 是否导致 occupancy 或 bank conflict
  问题；
- 编译器是否因为不满足对齐、dtype 或 descriptor 条件而走 fallback。

生成代码显示的指令路径只是解释线索，最终仍需用 profiler 和设备计数器验证。

### 4. 选择合适的指标

- **Latency**：单次 kernel 完成时间，适合固定 batch/sequence 的端到端比较；
- **有效带宽**：`读写字节数 / latency`，适合 elementwise、copy、im2col；
- **计算吞吐**：`FLOPs / latency`，适合 GEMM、GEMV 和 attention；
- **尾延迟**：p95/p99，适合服务场景和动态 shape；
- **资源效率**：结合生成代码、编译器报告和设备 profiler 观察 register、
  shared memory、occupancy、cache hit 和执行单元利用率。

例如 GEMM 的近似 TFLOPS：

```python
flops = 2 * M * N * K
tflops = flops / (latency_ms * 1e9)
print(f"throughput: {tflops:.3f} TFLOP/s")
```

不要用理论峰值直接判断 kernel 已经最优；内存受限的 elementwise 或小矩阵
kernel 更适合用带宽、occupancy 和 launch overhead 分析。

### 5. 使用 autotune 搜索配置

先定义一个小而有效的配置空间，再让 autotuner 编译和测量候选：

```python
@tilelang.autotune(
    configs=lambda M, N, K: [
        {"block_M": 64, "block_N": 128, "block_K": 32, "threads": 128},
        {"block_M": 128, "block_N": 128, "block_K": 32, "threads": 256},
        {"block_M": 128, "block_N": 256, "block_K": 64, "threads": 256},
    ],
    warmup=25,
    rep=100,
)
@tilelang.jit
def tuned_matmul(M, N, K, block_M=128, block_N=128, block_K=32, threads=256):
    ...
```

配置 key 必须对应 kernel factory 的参数；候选配置应满足 shared memory、线程数、
fragment shape 和目标指令约束。第一次搜索应固定单个 representative shape，
再为不同 M/N/K 或 sequence length 建立独立配置。

autotune 结果还会受到 cache、设备温度、后台进程和输入生成方式影响。发布
前应在干净环境中重复最佳配置，并把配置、软件版本、设备和 measured latency
一起保存。更多参数见 [官方 Autotuning 指南](https://tilelang.com/programming_guides/autotuning.html)。

### 6. 常见性能问题

| 症状 | 可能原因 | 建议 |
| --- | --- | --- |
| 首次运行很慢 | JIT 编译或 cache miss | 单独统计 compile，确认 cache 命中 |
| latency 波动大 | warmup 不足、后台负载、频率变化 | 固定设备，增加 warmup/repeat，报告分位数 |
| GEMM 没有提升 | block/warp policy 不合适或 fallback 到 FMA | 检查 source 中实际 intrinsic 和 layout |
| copy 带宽低 | 非连续 region、tail guard、未走 TME/async | 检查 stride、对齐、scope 和 descriptor 条件 |
| stage 增加反而变慢 | shared/register 压力或同步开销 | 比较 1/2/3 stage 的资源和 latency |
| 小矩阵变慢 | launch、barrier 或 pipeline 固定开销占比高 | 评估 simpler kernel、GEMV 或融合算子 |
| 结果正确但吞吐下降 | 误把 debug/print/assert 留在 kernel | 移除诊断接口后重新测量 |

### 7. 性能回归记录

每次提交至少保存：

```text
commit / package version:
torch + torch_musa:
MUSA SDK + driver:
device + target:
shape / dtype / stride:
block_M / block_N / block_K / threads / stages:
backend + warmup + repeat + return_mode:
correctness tolerance:
latency mean / median / p95:
generated intrinsic path:
```

性能回归应先确认生成路径一致，再比较数值；如果路径发生变化，先在调试页
记录 pass/source 差异，再决定是否接受 latency 变化。相关生成代码和 IR 方法见
[调试诊断](./07_debug_diagnostics.md)，GEMM、GEMV、attention 和 im2col 的
起始用例见[典型用例](./06_typical_use_cases.md)。
