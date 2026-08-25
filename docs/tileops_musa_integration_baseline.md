# TileOPs-MUSA 接入基线与验收路径

本文给出 TileOPs-MUSA 接入 TileLang-MUSA 时的源码边界、固定基线和验收顺序。
它不把“仓库中已有 MUSA 代码”解释成某个算子已经在当前软件栈或 S5000 上通过，
也不允许从 CUDA 架构编号推导摩尔线程硬件能力。

## 结论

TileLang-MUSA 已经不是待初始化的空后端。固定提交
`475bf79063776fc268d01bb8cabde24a92d753f8` 中已有 MUSA target 探测、Pass
Pipeline、设备端 Codegen、`mcc` 编译驱动、Host Wrapper、Runtime Module、MUSA
模板以及 MP22/MP31 测试。因此 TileOPs-MUSA 的正确工作不是再复制一条编译器，
而是把仓库自有 Kernel 以 `target="musa"` 送入这条既有编译链，并逐项补齐固定栈
编译、正确性和性能证据。

## 谁负责什么

| 阶段 | TileLang-MUSA 事实源 | 对 TileOPs-MUSA 的含义 |
| --- | --- | --- |
| Target 探测 | `tilelang/musa/target.py`、`tilelang/backend/__init__.py` | 只在 MUSA 工具链可发现时解析 `musa` target；算子库不自行猜测硬件架构。 |
| IR Lowering | `tilelang/musa/pipeline.py`、`tilelang/engine/lower.py` | TileLang PrimFunc 经过 MUSA 专用 Pass Pipeline，再拆分 Host/Device IR。 |
| 设备源码生成 | `tilelang/musa/codegen.py`、`src/backend/musa/codegen/codegen_musa.cc`、`src/backend/musa/codegen/rt_mod_musa.cc` | 生成 MUSA C；有编译和仅生成源码两条入口，便于保存可审计源码。 |
| Device 编译 | `tilelang/engine/lower.py`、`tilelang/contrib/mcc.py`、`tilelang/jit/adapter/libgen.py` | `tvm_ffi` 生成 `mubin`，`cython` 生成共享库；命令、目标架构和产物必须进入实验记录。 |
| Host ABI 与加载 | `tilelang/jit/adapter/tvm_ffi.py`、`tilelang/jit/adapter/musa_wrapper.py`、`tilelang/jit/adapter/libgen.py` | `tvm_ffi` 和 `cython` 是两种明确的执行后端；算子计时不能漏掉 TileOPs 标准边界内的必要工作。 |
| 目标模板与指令 | `src/tl_templates/musa/`、`src/backend/musa/op/` | copy、reduce、scan、GEMM 等语义由 MUSA 后端实现；不能用 CUDA 字符串替换代替验证。 |
| 测试范围 | `testing/musa/common/`、`testing/musa/mp22/`、`testing/musa/mp31/` | Common 用于后端基本语义，MP22/MP31 用于架构专用路径；TileOPs 仍需运行自己的 Manifest workload。 |

## 实际编译路线

两种执行后端共享前半段：

```text
TileOps Op + TensorSpec + compile-time params
    -> repository-owned TileLang Kernel, @tilelang.jit(target="musa")
    -> MUSA target detection and MUSAPassPipelineBody
    -> Host IR + Device IR
    -> CodeGenTileLangMUSA -> generated MUSA C
```

随后由明确选择的执行后端分流：

```text
tvm_ffi (auto 顺序中的首选)
    -> tilelang_callback_musa_compile -> mcc -> mubin
    -> MUSAModuleCreate + Host RuntimeModule -> callable

cython
    -> TLMUSASourceWrapper 生成 Host launch wrapper
    -> LibraryGenerator -> mcc --shared -> shared library
    -> CythonKernelWrapper -> callable
```

验收记录必须写明 `execution_backend`，两条路径的编译产物和调用边界不能混用。

这条路线与 TileFoundry 自身的 DSL -> HIR -> TIR -> Codegen 路线相互独立。
TileFoundry 可以作为候选生成、编译、验证和测量的 Agent 执行框架，但它不是
TileLang-MUSA IR 或 Runtime 的中间层。

## TileOPs-MUSA 接入合同

每个准备注册的算子必须满足以下条件：

1. Kernel 源码由 TileOPs-MUSA 仓库持有；运行时不得导入
   `tileops.kernels.*` 获取实现。
2. JIT 入口显式写出 `target="musa"`，输入设备只接受 `device.type == "musa"`。
3. Builder 的参数名、默认值、dtype、shape 和 variant 与固定 TileOPs Manifest
   对齐；不支持的区域在导入 TileLang 前明确拒绝。
4. TileLang-MUSA 的提交、MUSA SDK、`mcc`、torch/torch_musa、驱动和设备型号
   固定到同一份实验记录。
5. Builder 只有在全部要求的 Manifest workload 通过正确性后才可注册；历史数据
   或单个 shape 不能使整个 Op 自动升级。

## 从草案到启用的门禁

| 门禁 | 必须保存的证据 | 失败时状态 |
| --- | --- | --- |
| G0 固定基线 | 两个仓库 commit、依赖和环境版本 | `adapter_drafted` |
| G1 编译 | Manifest、执行后端、生成源码、完整 `mcc` 命令、退出状态、源码与 `mubin` 或共享库哈希 | 保持未注册并记录编译阻塞 |
| G2 加载运行 | 目标设备、Runtime/ABI、可调用入口和无 silent fallback 证明 | 保持未注册并记录运行阻塞 |
| G3 正确性 | 每个 workload 的 reference、容差、seed 和原始结构化结果 | 保持未注册并记录失败 workload |
| G4 正式性能 | 同一 workload、同一计时边界、warmup/repeat、MUSA Event 原始结果和最快官方基线 | 未达到阈值则不接受 |
| G5 启用 | 覆盖台账、来源、版本、G1-G4 证据均完整 | 才能加入 `ENABLED_BUILDERS` |

性能门禁中的“平均达到 70%”必须以同一批配对 workload 计算，并同时保留逐
workload 结果。正确性失败、缺测或无法归因的样本不能从平均值中静默删除。

## 当前证据边界

- 本文固定提交的源码结构可以静态核对，但本文没有产生新的 MUSA 编译、运行或
  性能结果。
- `test_records/2026-07-07-musa-mp31-noncuda-test-record.md` 记录的是另一提交
  `96f43ebd` 上的历史 S5000 测试：MUSA/MP31 子集为 595 passed、18 failed。
  它证明当时存在真实硬件测试，不证明本文固定提交或任一 TileOPs Kernel 已通过。
- 在把 TileOPs-MUSA builder 提升到 `builder_registered` 前，必须在计划采用的
  TileLang-MUSA 提交上重新执行 G1-G4。

## 首个可证伪闭环

建议先选一个 Elementwise 单算子、单 shape、单 dtype：固定 Manifest 后保存生成
源码，编译并记录所选执行后端的二进制，在 S5000 上与 Torch/官方实现做正确性和
配对 Event 计时。只有该闭环能稳定复现后，再扩展到同一算子的全部 Manifest
workload，最后扩展到算子族。这样编译器、Runtime、算子语义和性能问题可以被
分别定位。
