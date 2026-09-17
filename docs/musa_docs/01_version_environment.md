---
title: 版本与环境
description: TileLang-MUSA 的版本对应关系和环境要求
tags: [MUSA, TileLang]
---

## 版本与环境

### 环境要求

运行 TileLang-MUSA 前，请准备与 MUSA SDK、`torch_musa` 和 TileLang-MUSA
版本匹配的 Linux Python 环境：

| 项目 | 要求 |
| --- | --- |
| 操作系统 | Linux；源码构建建议使用 x86_64 开发环境 |
| C 运行时 | glibc 2.28 或更高版本（Ubuntu 20.04 及以上通常满足） |
| GPU | 已安装 MUSA 驱动的 MTT GPU；仓库 README 当前声明支持 S5000、S4000 和 M1000 |
| MUSA SDK | 推荐 5.2.0 及以上版本；兼容 4.3.8 和 5.2.0，需要对应版本的 `mcc`、MUSA runtime 和设备编译工具链 |
| Python | Python 3.10 是当前 `torch_musa`/TileLang-MUSA 文档推荐环境 |
| Python 依赖 | `torch`、`torch_musa`、`tilelang_musa` 和匹配版本的 `apache-tvm-ffi` |

基础 TileLang 编译、单卡 kernel 和正确性示例只需要 MUSA runtime；分布式
或依赖 muDNN 的模型示例再按项目需要安装 MCCL、muDNN 等组件。不要在同一
环境中混用公开 PyPI 的 upstream `tilelang` 和 MUSA 包；同一环境只选择一组
TileLang-MUSA/TVM FFI 版本。

:::note
请先准备好可运行 `torch_musa` 的 Python 环境。`torch_musa` 的安装参见
[Torch Musa](https://docs.mthreads.com/torchmusa/torchmusa-doc-online/introduction)。
具体 Python 包版本见[安装](./02_installation.md)。
:::

### 目标架构与测试范围

TileLang-MUSA 的 target 名称、硬件别名和测试目录不是一一对应的；最终可用
指令由设备、MUSA SDK 和 lowering 条件共同决定：

| target/别名 | 文档和测试中的定位 |
| --- | --- |
| MP22 / QY2 | QY2 MMA、WMMA/FMA、copy 和基础算子路径 |
| MP31 / PH1 | MP31（S5000）核心 GEMM、TME、FlashAttention、DSA 和模型算子路径 |

使用 `target="musa"` 时由后端选择当前设备兼容路径；需要固定架构时，使用
与设备和 SDK 匹配的 target 配置。公共示例不应写死某个具体架构。

### 安装前检查

安装完成后，建议在运行 kernel 前执行以下检查：

```bash
python - <<'PY'
import torch
import torch_musa
import tilelang
import tvm_ffi

print("torch:", torch.__version__)
print("tilelang:", tilelang.__version__)
print("tvm_ffi:", getattr(tvm_ffi, "__version__", "installed"))
print("musa available:", torch.musa.is_available())
print("musa device count:", torch.musa.device_count())
assert torch.musa.is_available()
PY
```

如果检查失败，按顺序确认：`torch_musa` 是否能导入、MUSA 驱动是否可见、
`MUSA_HOME`/`MUSA_PATH` 是否指向 SDK、`MUSA_VISIBLE_DEVICES` 是否屏蔽了
设备，以及安装的 FFI 版本是否与 TileLang-MUSA 配对。

### 环境变量

| 变量 | 用途 |
| --- | --- |
| `MUSA_HOME` / `MUSA_PATH` | 指定 MUSA SDK 根目录 |
| `MUSA_VISIBLE_DEVICES` | 控制进程可见的 MUSA 设备 |
| `TILELANG_TARGET` | 设置默认 target，例如 `musa` |
| `TILELANG_CACHE_DIR` | 设置 kernel cache 根目录 |
| `TILELANG_DISABLE_CACHE` | 禁用 kernel cache，适合调试生成代码 |
| `TILELANG_VERBOSE` | 输出更详细的编译日志 |

更多 dump、IR、编译命令和缓存变量见[调试诊断](./07_debug_diagnostics.md)。

### 版本说明

当前最新 TileLang-MUSA 发布版本为 `v0.1.12+musa.2`，运行环境兼容
MUSA SDK 4.3.8 和 5.2.0，推荐使用 MUSA SDK 5.2.0 及以上版本。该版本
基于 TileLang 0.1.12 适配 MUSA 平台。
本文以
`v0.1.8+musa.3` 作为旧版本基线，汇总 `v0.1.12+musa.2` 中的 MUSA
接口和编译器能力变化。

### 版本渠道

TileLang-MUSA 同时存在 MUSA SDK 配套发行版和开源仓库发行版两个发布渠道。两个渠道的
基础 TileLang 版本可能相同，但 `musa.*` patch 后缀、适配的 MUSA SDK 和
bugfix 集合不一定相同，不能只按 `0.1.x` 主版本号判断二者完全等价。MUSA SDK 配套发行版同样属于正式 release，面向 MUSA SDK Python 包源
发布，并不是开发快照或临时构建产物。

| 渠道 | 版本 | 获取方式 | 定位 |
| --- | --- | --- | --- |
| MUSA SDK 配套正式版 | `v0.1.12+musa.2` | MUSA SDK Python 包源 | 当前 SDK 文档使用的版本，包含相对开源 `v0.1.12+musa.1` 的额外 bugfix |
| MUSA SDK 配套正式版（历史） | `v0.1.8+musa.3` | MUSA SDK Python 包源 | 发布于 MUSA SDK 5.2.0 时代，配套 `apache-tvm-ffi==0.1.9.post3` |
| 开源仓库正式版 | `v0.1.12+musa.1` | [GitHub Release](https://github.com/tile-ai/tilelang-musa/releases) | 与 MUSA SDK 配套版本同一基础 TileLang 版本，可用于阅读源码和对照 API |
| 开源仓库正式版 | `v0.1.13+musa.1` | [GitHub Release](https://github.com/tile-ai/tilelang-musa/releases) | 开源仓库中的后续基础版本，接口和行为应按该 release 说明核对 |

开源仓库地址为
[https://github.com/tile-ai/tilelang-musa](https://github.com/tile-ai/tilelang-musa)。
当 MUSA SDK 配套包不可用时，用户可以借鉴开源仓库 `v0.1.12+musa.1` 的源码、测试和
release 说明；如果需要使用当前 SDK 文档中的修复和适配，应优先安装 MUSA SDK 配套的
`v0.1.12+musa.2`。开源 `v0.1.13+musa.1` 是另一个基础版本，不能直接
当作 MUSA SDK 配套 `v0.1.12+musa.2` 的替代包。

### 版本差异概览

以下表格只比较两个 MUSA SDK 配套正式 release：历史正式版 `v0.1.8+musa.3`
和当前正式版 `v0.1.12+musa.2`。开源仓库中的 `v0.1.12+musa.1`、
`v0.1.13+musa.1` 属于独立发布渠道，不能直接套用下表的 patch 差异。

| 方向 | MUSA SDK 配套正式历史版 `v0.1.8+musa.3` | 当前 MUSA SDK 配套正式版 `v0.1.12+musa.2` |
| --- | --- | --- |
| 基础版本 | 基于 TileLang 0.1.8，MUSA patch 版本为 `musa.3` | 基于 TileLang 0.1.12，MUSA patch 版本为 `musa.2`，并同步新的 backend、TIRx 与 TVM FFI 结构 |
| Python 依赖 | `tilelang-musa==0.1.8+musa.3` + `apache-tvm-ffi==0.1.9.post3` | `tilelang_musa==0.1.12+musa.2` + `apache-tvm-ffi==0.1.11.post1` |
| 目标架构 | 主要覆盖 MP22/QY2 与 MP31/PH1 | 覆盖 MP22/QY2 与 MP31/PH1，并完善两类目标的指令选择、布局推导、同步和 runtime 适配 |
| Copy / TME | 支持 `T.copy`、robust copy、async copy、TME 搬运和 descriptor cache hint | 增加显式 `T.tma_copy`（load 无隐式 wait、store 可批量后显式 wait）、`T.alias_buffer`，以及 1D/2D/3D im2col TME 路径 |
| Barrier / Pipeline | 支持基础 pipeline、barrier 与 producer 线程控制 | 统一 `expect_tx`、arrive/wait、warp-specialized producer-consumer 与 fallback barrier；`T.copy` 保持同步语义，`T.tma_copy` 交由用户管理 load barrier |
| GEMM lowering | MP31 SQMMA、PH1 WMMA/FMA 与 QY2 MMA/FMA 路径 | 完善 MP31 SQMMA/WMMA/FMA lowering 与布局、QY2 subgroup GEMM，并统一 `wg_wait`、转置 operand layout 和指令选择 |
| 低精度与缩放 | 以 FP16/BF16/FP32、INT8/UINT8 和 MP31 FP8 路径为主 | 完善 MP31 FP8/TF32、packed vector 和混合 operand 的 dtype、布局与指令选择 |
| Sparse / Reduce / Atomic | 主要使用通用 reduce 与标量 atomic | 增加 MUSA `T.gemm_sp`（QY2 sparse MMA、MP31 sparse FMA）、向量 atomic add，以及 TME atomic add/max/min 与 and/or/xor/inc/dec 归约 |
| 语言与算子 API | 支持主流 TileLang 语法和 MUSA 扩展接口 | 增加 `T.fast_div`/`T.fast_mod`/`T.fast_divmod`、`T.transpose`、warp vote/sync、packed vector、IEEE math、im2col 等接口 |
| JIT / runtime | 支持 pip 安装、JIT 编译、kernel cache 和 MUSA codegen | 重构 `src/backend/musa`，改进 host/device codegen、TVM FFI、cooperative launch、runtime allocator 兼容性、kernel source/ASM dump 与缓存处理 |
| 测试与 benchmark | 已有 MP22/MP31 GEMM、FA、DSA、TileKernels 测试和 benchmark | 扩展 MP22/MP31 测试，覆盖 TME、SQMMA/WMMA、atomic、im2col、语言/transform、cache、autotune、profiler 与典型模型 benchmark |


### 版本选择和兼容规则

- 当前最新发布版本 `v0.1.12+musa.2` 使用 `apache-tvm-ffi==0.1.11.post1`，
  兼容 MUSA SDK 4.3.8 和 5.2.0，推荐使用 MUSA SDK 5.2.0 及以上版本；
- 历史版本 `v0.1.8+musa.3` 发布于 MUSA SDK 5.2.0 时代，配套
  `apache-tvm-ffi==0.1.9.post3`；
- 不要只升级 `tilelang_musa` 而保留旧版 `apache-tvm-ffi`，也不要把两个
  版本的包安装在同一个 virtualenv 中；
- 修改 C++/backend 源码后，需要重新构建 native extension；只修改 Python
  kernel 时可以使用 editable 安装或关闭 cache 验证新生成代码；
- 版本确认以 `tilelang.__version__` 和 `tvm_ffi.__version__` 为准，不要只
  根据 pip 缓存目录名判断当前环境。

官方 TileLang 的通用安装、target 和编程指南可参考：

- [TileLang 官方文档](https://tilelang.com/)
- [官方编程指南](https://tilelang.com/programming_guides/overview.html)
- [官方 API Reference](https://tilelang.com/autoapi/tilelang/index.html)
