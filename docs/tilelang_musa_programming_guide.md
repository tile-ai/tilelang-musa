# Tilelang MUSA Programming Guide

Tilelang MUSA 支持几乎所有 Tilelang 官方语法, 除了 NV 特定的一些特性, 使用 Tilelang DSL 编写的算子, 可以通过 Tilelang MUSA 无缝运行到 MUSA 平台.

通常只需要做以下步骤:
1. 把 `@tilelang.jit(target="musa")` 修改成 `@tilelang.jit(target="musa")` 或者 `@tilelang.jit`
2. 使用 torch_musa 包, 创建 tensor 时指定 `device="musa"`

以下是 Tilelang MUSA 新增的一些扩展特性, 更适配 Moore Thread GPU 特性

## force_async_copy

MUSA 路径下，`T.copy(...)` 支持参数 `force_async_copy=True`

用于在该 copy 点请求 async copy lowering, 以生成 `cp_async` 相关代码路径

```python
T.copy(src[v], src_shared[v], force_async_copy=True)
```

在 CUDA 习惯中，用户通常更多依赖编译器自动策略; MUSA 在调优场景下，`force_async_copy` 是更常见的显式控制项

## make_robust_desc & src_robust_desc

MUSA 路径新增 robust copy 描述符：

- `robust_desc = T.make_robust_desc(addr, size_bytes)`
- `T.copy(..., src_robust_desc=robust_desc)`

用于描述源地址的有效字节范围，在越界读取场景下提供更稳健的加载语义。

用法

```python
robust_desc = T.make_robust_desc(T.address_of(src[1]), 8)
T.copy(src[tid], dst[tid], src_robust_desc=robust_desc)
```

也可与异步 copy 组合:

```python
T.copy(src, src_shared, force_async_copy=True, src_robust_desc=robust_desc)
```

## producer_threads

`T.Kernel(...)` 支持参数 `producer_threads: int | None`

用于在 warp-specialized 场景中显式控制 producer 线程数量

用法

```python
with T.Kernel(1, threads=128, producer_threads=32):
    ...
```

设置 `producer_threads` 会改变 producer/consumer 线程划分，从而影响下游线程组织与性能特征。

## MUSA 特定 PassConfig 选项

除通用选项外，MUSA 侧可关注：

- `tilelang.PassConfigKey.TL_DISABLE_SQMMA`: 禁用 SQMMA
- `tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST`: 尝试做更多 vectorzie
- `tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST`: Reduce 操作中做更多 vectorize
