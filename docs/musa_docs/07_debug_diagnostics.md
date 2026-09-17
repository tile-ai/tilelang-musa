---
title: 调试诊断
description: TileLang-MUSA 的生成错误、正确性、IR、源码和运行时诊断
tags: [MUSA, TileLang]
---

## 调试诊断

TileLang-MUSA 的问题通常分为三类：

1. **生成问题**：Python DSL、TIR lowering 或 MUSA C 编译失败；
2. **正确性问题**：kernel 能运行，但结果与 PyTorch/reference 不一致；
3. **性能问题**：结果正确，但 latency、带宽或吞吐低于预期。

建议始终保留一个最小可复现 kernel、固定输入和 target，并按“环境 → DSL →
IR → 生成源码 → 运行时”的顺序缩小问题范围。性能问题应在正确性稳定后
交给[性能分析](./08_performance.md)。

### 0. 固定环境和复现条件

调试前记录以下信息：

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
print("device count:", torch.musa.device_count())
PY
```

同时固定 GPU、`MUSA_VISIBLE_DEVICES`、target、输入 shape/dtype、随机种子、
cache 目录和 pass config。首次编译、kernel 运行和 profiler 测量应分开记录。

### 1. 生成问题：先确认失败阶段

先用最小参数调用 `.compile(...)`，不要一开始运行完整模型：

```python
kernel = tilelang.compile(
    program,
    target="musa",
    execution_backend="cython",
    verbose=True,
)
```

根据错误出现的位置判断阶段：

| 现象 | 优先检查 |
| --- | --- |
| Python 解析或类型错误 | `T.Tensor` shape/dtype、循环类型、buffer scope |
| TIR lowering 失败 | `T.copy` region、layout、GEMM shape、pipeline stage |
| MUSA C 编译失败 | 生成 source、target arch、dtype、MUSA SDK/mcc 版本 |
| wrapper/runtime 加载失败 | `torch_musa`、`apache-tvm-ffi`、动态库路径和 cache |
| kernel 启动后立即失败 | block/grid、barrier、shared memory、设备端 assert |

不要直接猜测 C++ pass。TileLang 的 lowering 是逐步变换的，先保存出错
shape、完整 traceback 和生成前的 TIR，再决定是否进入 backend 源码。

### 2. 查看生成的 MUSA 源码和产物

编译成功后，通过 `get_kernel_source()` 查看实际生成的 MUSA C/C++ source：

```python
from pathlib import Path

kernel = tilelang.compile(
    program,
    target="musa",
    execution_backend="cython",
)
source = kernel.get_kernel_source()
print(source)
Path("generated_kernel.mu").write_text(source)
```

重点搜索：

- `tma_load`、`tma_store`、barrier arrive/wait 是否成对出现；
- `tl_gemm`、SQMMA/WMMA/MMA intrinsic 是否是预期路径；
- global/shared/fragment 指针和 offset 是否包含正确的 block 坐标；
- tail tile 是否有 guard，vector lane 是否满足对齐；
- host wrapper 是否把 shape、stride、descriptor 和 stream 传对。

也可以用环境变量保留源码和编译命令：

```bash
TILELANG_DUMP_KERNEL_SOURCE=/tmp/tilelang_kernel.mu python your_script.py
TILELANG_PRINT_DEVICE_COMPILE_COMMAND=1 python your_script.py
TILELANG_MCC_EXTRA_ARGS="-save-temps,-v" python your_script.py
```

### 3. 查看 TIR 和 pass 变化

安装匹配版本的 `apache-tvm-ffi` 后，可以用 TVM instrument 打印 pass 后的 IR：

```python
from tvm.ir.instrument import PrintAfterAll

kernel = tilelang.compile(
    program,
    target="musa",
    instruments=[PrintAfterAll()],
)
```

需要定位“哪个 pass 引入变化”时，启用 Pass Diff：

```bash
# 终端彩色 diff
TILELANG_PASS_DIFF=terminal python your_script.py

# 生成 HTML 报告
TILELANG_PASS_DIFF=html \\
TILELANG_PASS_DIFF_OUTPUT=/tmp/tilelang_pass_diff \\
python your_script.py
```

Pass Diff 的关键变量：

| 变量 | 说明 |
| --- | --- |
| `TILELANG_PASS_DIFF` | `0`、`terminal`、`html` 或 `both` |
| `TILELANG_PASS_DIFF_OUTPUT` | HTML 报告目录，默认 `tmp/pass_diff_output` |
| `TL_LAYOUT_VISUALIZATION_ENABLE` | 开启 fragment/layout 可视化 |
| `TL_LAYOUT_VISUALIZATION_FORMATS` | `txt`、`png`、`pdf`、`svg` 或 `all` |

布局推导、warp-specialize 或 pipeline 结构异常时，可使用 pass visualizer：

```bash
python -m tilelang.tools.pass_visualizer.viewer \\
    path/to/kernel.py \\
    --target musa \\
    --set M=1024 --set N=1024 --set K=1024 \\
    --set block_M=128 --set block_N=128 --set block_K=32 \\
    --out /tmp/tilelang_passes.html
```

### 4. 正确性问题：使用设备端打印和断言

`T.print` 适合打印少量 lane、标量或小 buffer。GPU 线程并发打印会产生大量
输出，应先限制 block、thread 或条件：

```python
with T.Kernel(1, threads=128):
    tid = T.get_thread_binding()
    if tid == 0:
        T.print(value, msg="value:")
```

`T.device_assert` 适合边界和中间状态检查：

```python
with T.Kernel(T.ceildiv(N, 256), threads=256) as bx:
    tid = T.get_thread_binding()
    index = bx * 256 + tid
    T.device_assert(index < N, msg="index out of range")
```

设备端断言失败后，先用同步方式重新运行并保留第一条错误信息；不要继续
复用可能已经损坏的 shared buffer 或输出 tensor。

### 5. 对照 reference 和缩小 shape

正确性测试建议包含：

```python
torch.manual_seed(0)
output = kernel(*inputs)
expected = reference(*inputs)
torch.testing.assert_close(
    output.to(torch.float32),
    expected.to(torch.float32),
    rtol=rtol,
    atol=atol,
)
```

排查时依次缩小 M/N/K、block、stage 和线程数：

1. 先用一个 block、一个 stage、整除的 shape；
2. 再加入 tail tile、非连续 stride、transpose 和多 block；
3. 最后恢复 TME、SQMMA、robust copy、atomic 和 warp-specialize。

如果关闭某个 pass config 后结果改变，应把它视为定位线索，而不是修复方案。
关闭 safe-memory、safe-copy predication 或 index promotion 可能隐藏越界问题。

### 6. 调试日志和 cache

常用编译变量如下：

| 变量 | 用途 |
| --- | --- |
| `MUSA_HOME` / `MUSA_PATH` | 指定 MUSA SDK 路径 |
| `TILELANG_CACHE_DIR` | 指定 kernel cache 根目录 |
| `TILELANG_DISABLE_CACHE=1` | 禁用 cache，确认是否命中旧产物 |
| `TILELANG_PRINT_ON_COMPILATION=1` | 打印编译过程中的 kernel 信息 |
| `TILELANG_VERBOSE=1` | 开启 verbose 输出 |
| `TILELANG_REPLACE_MUSAC` | 用指定 MUSA source 替换生成 source |

修改 kernel 或 backend 后，如果行为与预期不一致，先清理或切换到新的
`TILELANG_CACHE_DIR`，再比较生成 source；不要只根据 Python traceback 判断
旧 cache 是否被复用。

TileLang 的 TVM logging 可以通过日志级别控制：开发 compiler/pass 时再开启
debug 级别，普通 kernel 调试优先使用 source dump、Pass Diff 和 `T.print`，
避免日志掩盖真正的设备错误。

### 7. 自动缩小最小复现

复杂 kernel 可以使用 AutoDD 自动删除无关 AST 片段：

```bash
python -m tilelang.autodd \\
    buggy_kernel.py \\
    --err-msg "Dimension mismatch" \\
    --backend subproc \\
    --timeout 120 \\
    -o minimized.py
```

`--backend subproc` 更稳定但更慢；`--err-msg` 应使用稳定且唯一的错误片段。
最小复现生成后，仍需手工确认它保留了相同 target、shape、dtype 和 pass config。

### 8. 常见排障路径

| 问题 | 建议顺序 |
| --- | --- |
| 编译失败 | 最小 shape → `verbose` → source/TIR → Pass Diff → 检查 target/dtype |
| 结果错误 | 固定 seed → `T.print`/`device_assert` → 关闭异步路径 → 缩小 tile |
| 随机错误 | 检查 barrier arrive/wait、pipeline stage、atomic 和 cache；用同步路径复现 |
| 只有某个 shape 错 | 检查 tail guard、stride、transpose、vector width 和 layout annotation |
| 性能突然下降 | 比较生成 intrinsic、shared memory/register、stage 和 cache；再运行 profiler |
| 重启后行为改变 | 检查 package/SDK/driver 版本和 `TILELANG_CACHE_DIR` |

完整的官方调试工具说明可参考 [Debugging Tile Language Programs](https://tilelang.com/tutorials/debug_tools_for_tilelang.html)。
