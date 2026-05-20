# TileLang MUSA Accelerated Ops Guide

`accelerated_ops` 用于暴露 MUSA target 上可以高效执行的计算 pattern。它和普通 `T.Cast`、`T.gemm` 这类 DSL 接口不同，定位更接近“显式请求使用某类硬件友好的 fused/accelerated 计算形式”。

## mul_half_float_to_bfloat16_x4
```python
## 显式调用

## 示例 1：slice 写法
with T.Kernel(1, threads=1):
    C[offset : offset + 4] = T.mul_half_float_to_bfloat16_x4(
        A[offset : offset + 4],
        B[offset : offset + 4],
    )

## 示例 2：T.Ramp 写法
with T.Kernel(1, threads=1):
    lanes = T.Ramp(offset, 1, 4)
    C[lanes] = T.mul_half_float_to_bfloat16_x4(A[lanes], B[lanes])
```

```python
## 编译器优化

## 示例 1：普通写法
with T.Kernel(1, threads=1):
    for i in T.Parallel(4):
        C[i] = T.Cast("bfloat16", A[i] * B[i])

## 示例 2：float32 scalar broadcast
with T.Kernel(1, threads=1):
    for i in T.Parallel(16):
        C[i] = T.Cast("bfloat16", A[i] * Scale[0])
```

- 功能：`mul_half_float_to_bfloat16_x4(x, y)` 表示把 `x(float16x4) * y(float32x4)` 的乘法结果转换成 `bfloat16x4` 并返回。
  - 显式 `T.mul_half_float_to_bfloat16_x4` 调用时，用户需要传入 4-lane vector 表达式。推荐使用 slice 写法，例如 `A[offset : offset + 4]` 和 `B[offset : offset + 4]`；也可以使用等价的 `T.Ramp(offset, 1, 4)` 写法。
  - 普通表达式 `T.Cast("bfloat16", A[i] * B[i])` 在满足 pattern 和类型约束时，也会由编译器自动识别并优化生成 `mul_half_float_to_bfloat16_x4` 操作。
- 注意：
  - 显式调用的 `lhs` 必须是 `float16x4`，`rhs` 必须是 `float32x4`，返回值必须是 `bfloat16x4`；否则编译器会报错。
