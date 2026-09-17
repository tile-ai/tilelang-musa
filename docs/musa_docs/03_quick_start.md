---
title: 快速开始
description: 使用 TileLang-MUSA 编写、编译、运行和校验第一个 kernel
tags: [MUSA, TileLang]
---

## 快速开始

本页用一个逐元素加法 kernel 展示完整流程：检查 MUSA 环境、定义 DSL
kernel、JIT 编译、查看生成的 MUSA C 代码、运行并与 PyTorch 结果校验。
如果还没有安装对应版本，请先完成[安装](./02_installation.md)，并确认
`torch_musa` 与 MUSA SDK 版本匹配。

### 1. 检查运行环境

在执行 TileLang-MUSA 代码前，先确认 Python 能加载 `torch_musa`、设备可用，
以及 TileLang-MUSA 和 TVM FFI 版本可读取：

```bash
python - <<'PY'
import torch
import torch_musa  # 注册 torch.musa 后端
import tilelang
import tvm_ffi

print("torch:", torch.__version__)
print("tilelang:", tilelang.__version__)
print("tvm_ffi:", getattr(tvm_ffi, "__version__", "installed"))
print("musa available:", torch.musa.is_available())
print("musa device count:", torch.musa.device_count())
assert torch.musa.is_available(), "MUSA device is not available"
PY
```

如果 `torch.musa` 不存在或设备不可用，先检查 `torch_musa` 是否安装、
`MUSA_HOME` 是否指向当前 SDK，以及驱动和运行时版本是否匹配。不要在
CPU-only Python 环境中直接运行下面的 `device="musa"` 示例。

### 2. 编写 Elementwise Add

新建 `elementwise_add.py`：

```python
import tilelang
import tilelang.language as T
import torch
import torch_musa  # noqa: F401  # 注册 MUSA 后端

# 调试或反复修改 kernel 时可以暂时关闭缓存。
tilelang.disable_cache()


@tilelang.jit
def elementwise_add(A, B, num_per_thread=8, threads=256, dtype="float32"):
    N = T.const("N")
    A: T.Tensor[(N,), dtype]
    B: T.Tensor[(N,), dtype]
    C = T.empty((N,), dtype)

    # 每个 block 处理 threads * num_per_thread 个元素。
    with T.Kernel(T.ceildiv(N, threads * num_per_thread), threads=threads) as bx:
        for i, j in T.Parallel(threads, num_per_thread):
            index = (bx * threads + i) * num_per_thread + j
            # 这个 guard 让 N 不是 tile size 整数倍时仍然安全。
            if index < N:
                C[index] = A[index] + B[index]
    return C


def main():
    N = 4097  # 故意使用 tail tile，验证边界 guard
    torch.manual_seed(0)

    a = torch.randn(N, dtype=torch.float32, device="musa")
    b = torch.randn(N, dtype=torch.float32, device="musa")

    # compile 会触发首次 lowering 和设备代码编译。
    kernel = elementwise_add.compile(N=N)
    print(kernel.get_kernel_source())

    c = kernel(a, b)
    expected = a + b
    torch.testing.assert_close(c, expected, rtol=1e-5, atol=1e-5)
    print("correctness: pass")


if __name__ == "__main__":
    main()
```

运行：

```bash
python elementwise_add.py
```

这个例子中 `N` 是编译期绑定的符号参数，`elementwise_add.compile(N=N)`
把它绑定为实际问题规模。`T.Kernel` 的第一个参数是 grid size，
`threads=256` 是每个 block 的线程数；`T.Parallel(threads, num_per_thread)`
生成两个嵌套的并行迭代空间。

### 3. 编译、运行和结果校验

`@tilelang.jit` 返回的是可编译的 kernel factory。第一次调用
`.compile(...)` 时会完成 TIR lowering、MUSA C 代码生成和 wrapper 编译；后续
相同参数可以复用 kernel cache。`kernel(a, b)` 才会向 MUSA 设备发起运行。

推荐按以下顺序定位问题：

1. 先运行环境检查，确认 `torch.musa.is_available()` 为 `True`；
2. 再调用 `.compile(...)`，确认 DSL 和 target lowering 可以通过；
3. 用 `get_kernel_source()` 查看生成的 MUSA C 代码；
4. 最后运行 kernel，并用 `torch.testing.assert_close` 比较参考结果。

如果需要明确指定 target 或 execution backend，可改用显式编译：

```python
kernel = tilelang.compile(
    elementwise_add.get_tir(N=N),
    target="musa",
    execution_backend="cython",
)
```

`@tilelang.jit`、`get_tir` 和 `tilelang.compile` 的参数形式可能随版本变化；
完整通用 API 参见 [TileLang 官方 API Reference](https://tilelang.com/autoapi/tilelang/index.html)。

### 4. 测量 kernel 延迟

正确性通过后再测量性能。编译时间不应计入 kernel latency；使用 profiler
时可以显式指定 warmup 和 repeat：

```python
profiler = kernel.get_profiler()
latency_ms = profiler.do_bench(
    n_warmup=10,
    n_repeat=100,
    backend="event",
    return_mode="mean",
)
print(f"latency: {latency_ms:.3f} ms")
```

`n_warmup` 和 `n_repeat` 是迭代次数；`warmup` 和 `rep` 是 profiler 的时间
预算参数。不同 backend、输入 shape 和缓存状态的结果不能直接比较，应固定
输入、设备、warmup/repeat 和校验方式。更完整的 benchmark 方法见
[性能分析](./08_performance.md)。

### 5. 从简单 kernel 继续学习

- [官方编程接口](./04_official_programming_interface.md)：类型、控制流、内存和核心算子；
- [MUSA 扩展](./05_musa_extensions.md)：TME、SQMMA、robust copy、layout 和 MUSA 指令；
- [典型用例](./06_typical_use_cases.md)：GEMM、FlashAttention、GEMV、MoE 和卷积；
- [调试诊断](./07_debug_diagnostics.md)：查看 IR、MUSA C 源码、设备端打印和断言；
- [TileLang 官方编程指南](https://tilelang.com/programming_guides/overview.html)：跨后端 DSL 接口和最新示例。

:::caution
示例使用 `N=4097` 验证 tail tile。删除 `if index < N` 或关闭安全访问相关
PassConfig 后，只有在 launch geometry 已经严格保证所有访问不越界时才安全。
:::
