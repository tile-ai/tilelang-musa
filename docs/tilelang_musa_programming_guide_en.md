# Tilelang MUSA Programming Guide

Tilelang MUSA supports almost all official Tilelang syntax, except for a small set of NVIDIA-specific features. Operators written with the Tilelang DSL can run on the MUSA platform through Tilelang MUSA with minimal migration effort.

In most cases, you only need to:
1. Set the decorator to `@tilelang.jit(target="musa")`, or use `@tilelang.jit`.
2. Use `torch_musa`, and create tensors with `device="musa"`.

The sections below describe the key extensions added in Tilelang MUSA to better match Moore Threads GPU characteristics.

## force_async_copy

On the MUSA path, `T.copy(...)` supports the argument `force_async_copy=True`.

This requests async copy lowering at that copy site, so the generated code can take the `cp_async`-related path.

```python
T.copy(src[v], src_shared[v], force_async_copy=True)
```

In typical CUDA workflows, users more often rely on compiler-selected strategies. In MUSA tuning workflows, `force_async_copy` is a more commonly used explicit control.

## make_robust_desc & src_robust_desc

The MUSA path adds a robust-copy descriptor:

- `robust_desc = T.make_robust_desc(addr, size_bytes)`
- `T.copy(..., src_robust_desc=robust_desc)`

This describes the valid byte range of the source address and provides more robust load semantics for out-of-bounds read scenarios.

Usage:

```python
robust_desc = T.make_robust_desc(T.address_of(src[1]), 8)
T.copy(src[tid], dst[tid], src_robust_desc=robust_desc)
```

It can also be combined with async copy:

```python
T.copy(src, src_shared, force_async_copy=True, src_robust_desc=robust_desc)
```

## producer_threads

`T.Kernel(...)` supports the parameter `producer_threads: int | None`.

It is used to explicitly control the number of producer threads in warp-specialized scenarios.

Usage:

```python
with T.Kernel(1, threads=128, producer_threads=32):
    ...
```

Setting `producer_threads` changes the producer/consumer thread partitioning, which can affect downstream thread organization and performance characteristics.

## MUSA-specific PassConfig options

In addition to common options, these MUSA-side options are especially relevant:

- `tilelang.PassConfigKey.TL_DISABLE_SQMMA`: disable SQMMA
- `tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST`: attempt more vectorization
- `tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST`: apply more vectorization in reduce operations
