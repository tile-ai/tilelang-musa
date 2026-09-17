---
title: 典型用例
description: TileLang-MUSA 的 GEMM、GEMV、Attention 和 im2col 用例
tags: [MUSA, TileLang]
---

## 典型用例

本页按“先写出正确 kernel，再逐步使用 MUSA 扩展”的顺序组织示例。示例
默认使用 MUSA tensor；运行前请完成[环境检查](./01_version_environment.md)，
并先阅读[快速开始](./03_quick_start.md)中的编译和校验流程。

### Dense GEMM：从 tile 到 pipeline

下面的 kernel 展示一个常见的 shared-memory GEMM：A/B 从 global 搬到 shared，
在 fragment accumulator 中累加，再写回 global。`T.Pipelined` 让后端有机会
重叠下一轮搬运和当前轮计算。

```python
import tilelang
import tilelang.language as T
import torch
import torch_musa  # noqa: F401


tilelang.disable_cache()


@tilelang.jit
def matmul(A, B, block_M, block_N, block_K,
           dtype="float16", accum_dtype="float"):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(
        T.ceildiv(N, block_N),
        T.ceildiv(M, block_M),
        threads=512,
    ) as (bx, by):
        T.use_swizzle(panel_size=4, order="col")
        A_shared = T.alloc_shared((block_M, block_K), dtype)
        B_shared = T.alloc_shared((block_K, block_N), dtype)
        C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
        T.clear(C_local)

        for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=3):
            T.copy(A[by * block_M, k * block_K], A_shared)
            T.copy(B[k * block_K, bx * block_N], B_shared)
            T.gemm(
                A_shared,
                B_shared,
                C_local,
                policy=T.GemmWarpPolicy.Square,
            )

        T.copy(C_local, C[by * block_M, bx * block_N])
    return C


M = N = K = 1024
kernel = matmul.compile(
    M=M,
    N=N,
    K=K,
    block_M=256,
    block_N=256,
    block_K=64,
)
A = torch.randn((M, K), dtype=torch.float16, device="musa")
B = torch.randn((K, N), dtype=torch.float16, device="musa")
C = kernel(A, B)
ref = torch.mm(A, B)
torch.testing.assert_close(
    C.to(torch.float32),
    ref.to(torch.float32),
    rtol=1.25e-1,
    atol=1.25e-1,
)
```

这个最小 GEMM 假设 M/N/K 都能被对应 block 整除。要支持任意 shape，应为
global→shared copy 和输出 store 增加 tail guard，或使用后端提供的 safe-copy
路径。`policy`、`transpose_A/B`、`wg_wait` 和具体 SQMMA/WMMA lowering 见
[MUSA 扩展](./05_musa_extensions.md)。

### GEMM 的调优顺序

建议一次只改变一个因素，并在每次修改后同时检查结果和生成代码：

1. 先固定 `block_M/N/K`，确认 `T.gemm` 结果正确；
2. 再调整 `threads` 和 `GemmWarpPolicy`，观察寄存器、shared memory 和 occupancy；
3. 使用 `num_stages` 调整搬运/计算重叠，确认 barrier wait 没有落后于第一次消费；
4. 在目标支持时添加 SQMMA/WMMA layout annotation 或低精度 dtype；
5. 最后使用 autotune 搜索有限的配置集合，并保存最佳配置。

### GEMV：矩阵向量乘

GEMV 可以看成 `(1, K) * (K, N)` 的特殊 GEMM。与直接复用 GEMM 不同，GEMV
通常需要让多个线程沿 K 维并行，再用 shared reduction 或 atomic 合并部分和：

```python
# 结构示意：每个 block 负责一段 N，每个线程负责 K 的一个分片。
with T.Kernel(T.ceildiv(N, block_N), threads=(block_N, reduce_threads)) as bx:
    tn = T.get_thread_binding(0)
    tk = T.get_thread_binding(1)
    accum = T.alloc_local((1,), accum_dtype)
    T.clear(accum)

    for ko in T.serial(T.ceildiv(K, block_K)):
        for kk in T.serial(T.ceildiv(block_K, reduce_threads)):
            k = ko * block_K + tk * T.ceildiv(block_K, reduce_threads) + kk
            if k < K:
                accum[0] += x[k].astype(accum_dtype) * A[bx * block_N + tn, k].astype(accum_dtype)

    # 根据 reduce_threads 选择 shared reduction 或 atomic add。
    T.atomic_add(C[bx * block_N + tn], accum[0])
```

GEMV 的关键参数是 `block_N`、`block_K`、K 维并行度和归约方式。小 K 通常
适合每线程直接累加；大 K 可以增加 `reduce_threads`，但要同时考虑 atomic
冲突、shared reduction 开销和向量化读取。仓库中的完整 GEMV 讨论见
[TileLang 官方 GEMV 说明](https://tilelang.com/deeplearning_operators/gemv.html)，MUSA 实现可从 [MUSA GEMV 测试](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/testing/musa/mp31/gemv/)
开始对照。

### Attention、MLA、GDN 和 MoE

Attention 类 kernel 通常由多个阶段组成：Q/K/V 搬运、QK^T GEMM、softmax 或
归约、PV GEMM 和输出 store。建议先按阶段分配 shared/fragment buffer，再
使用 `T.Pipelined` 重叠 K/V 搬运和矩阵计算；不要一开始就同时引入 layout
重标注、异步 barrier 和低精度 scale。

| 场景 | 推荐先看的 benchmark / 源码 | 重点 |
| --- | --- | --- |
| FlashAttention | [FlashAttention benchmark](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/modelops/) | pipeline copy、shared reuse、softmax 归约 |
| MLA / DSA | [MLA benchmark](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/modelops/ops/) | 低精度 GEMM、长序列 tile 和 workspace |
| Linear Attention / GDN | [GDN benchmark](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/mate/ops/) | 累积状态、scan、分块归约 |
| MoE | [MoE benchmark kernels](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/tilekernels/moe/) | GEMM batch、路由后的不规则 tile 和输出合并 |

这些 kernel 对 shape、dtype 和 device capability 很敏感。应先运行仓库测试
中的固定 shape，再把动态 shape、tail tile 和不同 batch 单独加入验证矩阵。

### Benchmark 中的 FlashAttention 示例

仓库的 [MP31 modelops benchmark](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/benchmark/mp31/modelops/) 提供了可以直接复现的 MHA forward benchmark。对应源码入口为：

- [MHA kernel](https://github.com/tile-ai/tilelang-musa/blob/v0.1.12%2Bmusa.1/benchmark/mp31/modelops/kernels/example_mha_fwd_bhsd.py)
- [MHA benchmark wrapper](https://github.com/tile-ai/tilelang-musa/blob/v0.1.12%2Bmusa.1/benchmark/mp31/modelops/ops/example_mha_fwd_bhsd_benchmark.py)

默认 case `example_mha_fwd_bhsd` 使用 `B=2`、`H=28`、`seq_q=8192`、
`seq_kv=8192`、`D=128`、FP16、非 causal attention。kernel 内包含两个主要
GEMM 阶段：

1. `Q @ K^T` 生成 attention scores；
2. online softmax 后使用 `P @ V` 生成输出。

默认 tile 配置是 `block_M=256`、`block_N=128`、`num_stages=1`、
`threads=512`。代码还展示了 `T.gemm(..., transpose_B=True)`、FullRow
warp policy、fragment softmax、pipeline copy 和结果校验，是观察 GEMM 与
Attention 如何组合的完整例子。

从仓库根目录运行单个 case：

```bash
python benchmark/mp31/modelops/ops/example_mha_fwd_bhsd_benchmark.py \
  --batch 2 --heads 28 --seq_q 8192 --seq_kv 8192 --dim 128
```

也可以通过统一 runner 运行同一个 modelops case：

```bash
python benchmark/mp31/runner.py \
  --source modelops \
  --cases example_mha_fwd_bhsd \
  --allow-non-release-build
```

正式性能数据建议使用 Release build，去掉 `--allow-non-release-build`，并在
运行前确认 MUSA 设备、输入显存和 MUSA SDK 版本。该 case 会先和 PyTorch
reference 做 `assert_close`，再记录 latency、TFLOPS 和有效带宽；不要把
它的 FlashAttention 结果直接当成单独 GEMM 的结果。

### 卷积和 im2col

`T.c2d_im2col_1d/2d/3d` 支持 NWC、NHWC、NDHWC 输入，把卷积窗口展开为
二维 shared tile，再交给后续 GEMM。TME 快路径要求通道搬运宽度满足对齐
约束；不能证明 descriptor 或 layout 合法时会回退到 scalar 实现，并对越界
输入补 0。

| 输入布局 | 接口 | 典型场景 |
| --- | --- | --- |
| NWC | `T.c2d_im2col_1d` | 1D convolution |
| NHWC | `T.c2d_im2col_2d` | 2D convolution |
| NDHWC | `T.c2d_im2col_3d` | 3D convolution |

完整参数顺序和 fallback 语义见 [MUSA 扩展](./05_musa_extensions.md)；仓库
[卷积测试目录](https://github.com/tile-ai/tilelang-musa/tree/v0.1.12%2Bmusa.1/testing/musa/mp31/convolution/)。

### 用例选择和验证清单

| 目标 | 起点 | 必须验证 |
| --- | --- | --- |
| 新写一个算子 | Elementwise Add | dtype、tail tile、CPU/PyTorch reference |
| 追求矩阵吞吐 | Dense GEMM | block shape、accumulator、pipeline、误差容限 |
| 推理小矩阵 | GEMV | K 维并行度、归约冲突、向量化读取 |
| 处理卷积输入 | im2col | descriptor 对齐、padding、stride、fallback |
| 调优复杂模型 kernel | Attention/MLA/GDN/MoE | 固定 shape、workspace、同步、端到端结果 |

所有 benchmark 前都应先完成正确性校验；编译时间、warmup、设备同步和输入
分配不应混入 kernel latency。性能测量方法见[性能分析](./08_performance.md)。
