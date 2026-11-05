# TileLang Env 使用手册

本文汇总 TileLang 常用环境变量，覆盖安装路径、缓存、编译调试、MUSA debug、autotune 和测试输出。

## 基本规则

布尔变量通常接受以下 truthy 值：

```bash
1 true yes on
```

通常接受以下 falsy 值：

```bash
0 false no off
```

示例：

```bash
TILELANG_DISABLE_CACHE=1 python your_script.py
TILELANG_PRINT_DEVICE_COMPILE_COMMAND=1 python your_script.py
```

## 安装路径与依赖路径

| 变量 | 默认值/自动探测 | 用途 |
| --- | --- | --- |
| `CUDA_HOME` | `CUDA_PATH`、`nvcc` 路径、`/usr/local/cuda` 等 | CUDA 安装目录。 |
| `CUDA_PATH` | 无 | `CUDA_HOME` 的备选输入。 |
| `MUSA_HOME` | `MUSA_PATH`、`mcc` 路径、`/usr/local/musa` | MUSA 安装目录。 |
| `MUSA_PATH` | 无 | `MUSA_HOME` 的备选输入。 |
| `ROCM_PATH` | `ROCM_HOME`、`hipcc` 路径、`/opt/rocm` | ROCm 安装目录。 |
| `ROCM_HOME` | 无 | `ROCM_PATH` 的备选输入。 |
| `TL_CUTLASS_PATH` | `3rdparty/cutlass/include` | CUTLASS include 目录。 |
| `TL_MUTLASS_PATH` | `3rdparty/mutlass/include` | MUTLASS include 目录。 |
| `TL_COMPOSABLE_KERNEL_PATH` | `3rdparty/composable_kernel/include` | Composable Kernel include 目录。 |
| `TL_TEMPLATE_PATH` | `src` | TileLang C++/CUDA/MUSA 模板目录。 |
| `TVM_IMPORT_PYTHON_PATH` | `3rdparty/tvm/python` | TVM Python 包路径。会被 prepend 到 `PYTHONPATH` 和 `sys.path`。 |
| `TVM_LIBRARY_PATH` | TileLang build lib 路径 | TVM/TileLang 动态库搜索路径。 |
| `SKIP_LOADING_TILELANG_SO` | `0` | 设为 truthy 时跳过加载 TileLang 动态库。通常只用于轻量导入或调试。 |

示例：

```bash
export MUSA_HOME=/usr/local/musa
export TL_MUTLASS_PATH=/path/to/mutlass/include
python your_script.py
```

## 缓存与临时文件

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TILELANG_CACHE_DIR` | `~/.tilelang/cache` | Kernel cache 根目录。 |
| `TILELANG_TMP_DIR` | `$TILELANG_CACHE_DIR/tmp` | TileLang 临时文件目录。 |
| `TILELANG_DISABLE_CACHE` | `0` | 全局禁用 kernel cache。 |
| `TILELANG_CLEAR_CACHE` | `0` | 启动时清理 cache。已废弃，不建议新增依赖。 |
| `TILELANG_CLEANUP_TEMP_FILES` | `0` | 编译后清理临时编译文件。默认保留，方便 debug。 |

以下 debug 变量只要启用，也会强制禁用 kernel cache，避免复用旧产物：

```text
TILELANG_DUMP_KERNEL_SOURCE
TILELANG_PRINT_ASM
TILELANG_DUMP_ASM
TILELANG_REPLACE_ASM
```

## 编译输出与通用调试

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TILELANG_PRINT_ON_COMPILATION` | `1` | 编译时打印 kernel 名称。 |
| `TILELANG_PRINT_DEVICE_COMPILE_COMMAND` | `0` | 打印 `nvcc`/`mcc` 等设备编译命令。 |
| `TILELANG_REPLACE_MUSAC` | 空 | 指向一个源码文件，用它替换生成的 MUSA C 源码再编译 wrapper library。 |
| `TILELANG_OPT_LEVEL` | 空 | MUSA wrapper library 编译时启用额外优化选项。当前设值后会追加 `-Od3`。 |
| `TILELANG_MCC_EXTRA_ARGS` | 空 | 追加到 MUSA wrapper library 的 `mcc` 参数，多个参数用逗号分隔。 |

示例：

```bash
TILELANG_PRINT_DEVICE_COMPILE_COMMAND=1 python your_script.py
TILELANG_MCC_EXTRA_ARGS="-save-temps,-v" python your_script.py
TILELANG_REPLACE_MUSAC=/tmp/edited_kernel.mu python your_script.py
```

## MUSA Kernel Source 与 ASM Debug

这些变量作用在 `tilelang.contrib.mcc.compile_musa` 路径，主要用于调试 MUSA device kernel。

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TILELANG_DUMP_KERNEL_SOURCE` | 空 | Dump 生成的 MUSA kernel source，并同时 dump 对应的 `mcc` 编译命令和 `mcc -S` 汇编生成命令。 |
| `TILELANG_PRINT_ASM` | 空 | 额外用 `mcc -S` 生成 `.s`，并打印到 stdout。 |
| `TILELANG_DUMP_ASM` | 空 | 用 `musaasm -M` 从 fatbin dump 可编辑 assembly。 |
| `TILELANG_REPLACE_ASM` | 空 | 用指定 assembly 重新生成 device object，并替换 fatbin 中的 device bundle。 |

`TILELANG_DUMP_KERNEL_SOURCE` 支持三种用法：

```bash
# dump 到当前目录，自动编号为 ./tilelang_musa_kernel_0000.mu、./tilelang_musa_kernel_0001.mu ...
TILELANG_DUMP_KERNEL_SOURCE=1 python your_script.py

# 以指定文件 stem 自动编号，例如 /tmp/kernel_0000.mu、/tmp/kernel_0001.mu ...
TILELANG_DUMP_KERNEL_SOURCE=/tmp/kernel.mu python your_script.py

# dump 到指定目录，自动编号为 /tmp/tilelang_kernels/tilelang_musa_kernel_0000.mu ...
TILELANG_DUMP_KERNEL_SOURCE=/tmp/tilelang_kernels/ python your_script.py
```

`.cmd` 文件包含两段：

```bash
# compile
mcc --cuda-device-only ... -o ./tilelang_musa_kernel_0000.mubin ./tilelang_musa_kernel_0000.mu

# asm
mcc --cuda-device-only ... -S -o ./tilelang_musa_kernel_0000.s ./tilelang_musa_kernel_0000.mu
```

如果 `TILELANG_DUMP_KERNEL_SOURCE` 指定为 `/tmp/kernel.mu`，则 `.cmd` 中会使用 `/tmp/kernel_0000.mu`、`/tmp/kernel_0000.mubin` 和 `/tmp/kernel_0000.s`。如果这些文件已存在，会继续递增到 `_0001`。

`TILELANG_DUMP_ASM` 和 `TILELANG_REPLACE_ASM` 的路径规则：

```bash
# dump 到当前目录 ./tilelang_musaasm.asm
TILELANG_DUMP_ASM=1 python your_script.py

# dump 到指定文件
TILELANG_DUMP_ASM=/tmp/tilelang_musaasm.asm python your_script.py

# 使用编辑后的 asm 替换 fatbin device bundle
TILELANG_REPLACE_ASM=/tmp/tilelang_musaasm.asm python your_script.py
```

典型调试流程：

```bash
TILELANG_DUMP_KERNEL_SOURCE=/tmp/kernel.mu \
TILELANG_PRINT_ASM=1 \
TILELANG_DUMP_ASM=/tmp/kernel.asm \
python your_script.py
```

然后编辑 `/tmp/kernel.asm`，再运行：

```bash
TILELANG_REPLACE_ASM=/tmp/kernel.asm python your_script.py
```

## 默认 JIT 参数

这些变量影响 `tilelang.jit` 和 kernel cache 中未显式传入的默认参数。

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TILELANG_TARGET` | `auto` | 默认编译 target，例如 `cuda`、`musa`、`hip`、`llvm`。 |
| `TILELANG_EXECUTION_BACKEND` | `auto` | 默认 execution backend。 |
| `TILELANG_VERBOSE` | `0` | 默认 verbose 编译输出。 |

示例：

```bash
TILELANG_TARGET=musa TILELANG_VERBOSE=1 python your_script.py
```

## Kernel 选择

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TILELANG_USE_GEMM_V1` | `0` | 设为 truthy 时强制使用 GEMM v1；默认使用 GEMM v2。 |

## Auto-Tuning

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TILELANG_AUTO_TUNING_DISABLE_CACHE` | `0` | 禁用 autotune cache。 |
| `TILELANG_AUTO_TUNING_CPU_UTILITIES` | `0.9` | autotune 可使用的 CPU 比例。 |
| `TILELANG_AUTO_TUNING_CPU_COUNTS` | `-1` | autotune CPU 数量；`-1` 表示自动。 |
| `TILELANG_AUTO_TUNING_MAX_CPU_COUNT` | `-1` | autotune CPU 数量上限；`-1` 表示不限制。 |

示例：

```bash
TILELANG_AUTO_TUNING_DISABLE_CACHE=1 \
TILELANG_AUTO_TUNING_CPU_COUNTS=16 \
python tune.py
```

## 编译器选择与宿主编译

| 变量 | 默认值/自动探测 | 用途 |
| --- | --- | --- |
| `CXX` | 系统 C++ 编译器 | 指定 host C++ 编译器。 |
| `CC` | 系统 C 编译器 | `CXX` 未设置时作为备选。 |

## 测试与性能回归

| 变量 | 默认值 | 用途 |
| --- | --- | --- |
| `TL_PERF_REGRESSION_FORMAT` | `text` | 设为 `json` 时，性能回归测试输出单行 JSON marker，方便父进程收集。 |

示例：

```bash
TL_PERF_REGRESSION_FORMAT=json python -m pytest testing/...
```

## Pass Config 相关输出目录

以下不是环境变量，而是 TVM pass config。它们通常通过 Python API 传入，用来把生成的 device kernel 源码和编译产物固定输出到指定目录。

| 配置键 | 用途 |
| --- | --- |
| `cuda.kernels_output_dir` | CUDA kernel source 和目标产物输出目录。 |
| `musa.kernels_output_dir` | MUSA kernel source 和目标产物输出目录。 |
| `tl.enable_dump_ir` | 开启 lowering pass IR dump。 |
| `tl.dump_ir_path` | IR dump 输出目录。 |

示例：

```python
import tilelang

kernel = tilelang.compile(
    func,
    target="musa",
    pass_configs={
        "musa.kernels_output_dir": "/tmp/tilelang_musa_kernels",
        "tl.enable_dump_ir": True,
        "tl.dump_ir_path": "/tmp/tilelang_ir",
    },
)
```
