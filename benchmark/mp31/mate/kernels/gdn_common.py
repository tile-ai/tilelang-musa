import functools
from collections import OrderedDict
from collections.abc import Callable
from typing import Any

import torch
import tilelang
import tilelang.language as T


def tensor_cache(fn: Callable[..., Any]) -> Callable[..., Any]:
    cache: OrderedDict[
        tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]],
        tuple[tuple[Any, ...], dict[str, Any], Any],
    ] = OrderedDict()
    cache_size = 256

    def get_id(x: Any) -> Any:
        if type(x) in (int, float, str, bool, type(None)):
            return x
        return id(x)

    def make_key(
        args: tuple[Any, ...], kwargs: dict[str, Any]
    ) -> tuple[tuple[Any, ...], tuple[tuple[str, Any], ...]]:
        return tuple(get_id(arg) for arg in args), tuple(
            sorted((key, get_id(value)) for key, value in kwargs.items())
        )

    @functools.wraps(fn)
    def wrapper(*args: Any, **kwargs: Any) -> Any:
        key = make_key(args, kwargs)
        if key in cache:
            cache.move_to_end(key)
            return cache[key][2]
        result = fn(*args, **kwargs)
        cache[key] = (args, kwargs, result)
        cache.move_to_end(key)
        if len(cache) > cache_size:
            cache.popitem(last=False)
        return result

    return wrapper


def _row_major_strides(shape):
    stride = 1
    strides = []
    for extent in reversed(shape):
        strides.append(stride)
        stride *= extent
    return tuple(reversed(strides))


def cosize(shape, strides=None):
    shape = tuple(shape)
    if strides is None:
        strides = _row_major_strides(shape)
    else:
        strides = tuple(strides)
        assert len(shape) == len(strides), "shape and strides must have the same rank"

    size = 1
    for extent, stride in zip(shape, strides):
        size += (extent - 1) * stride
    return size


@tensor_cache
def prepare_lens(cu_seqlens: torch.LongTensor) -> torch.LongTensor:
    return torch.diff(cu_seqlens)


@tensor_cache
def prepare_chunk_indices(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    indices = torch.cat(
        [
            torch.arange(n, device=cu_seqlens.device)
            for n in tilelang.cdiv(prepare_lens(cu_seqlens), chunk_size).tolist()
        ]
    )
    return torch.stack([indices.eq(0).cumsum(0) - 1, indices], 1).to(cu_seqlens)


@tilelang.jit()
def tilelang_prepare_chunk_offsets(
    chunk_size,
    block_size,
    dtype,
):
    batch_size_plus_1 = T.dynamic("batch_size_plus_1")
    num_threads = min(max(block_size, 32), 128)

    @T.prim_func
    def tilelang_prepare_chunk_offsets_kernel(
        cu_seqlens: T.Tensor([batch_size_plus_1], dtype=dtype),
        chunk_offsets: T.Tensor([batch_size_plus_1], dtype=dtype),
    ):
        with T.Kernel(1, threads=num_threads) as (bb,):
            seqlen_start_fragment = T.alloc_fragment((block_size), dtype=dtype)
            seqlen_end_fragment = T.alloc_fragment((block_size), dtype=dtype)
            chunk_offset_fragment = T.alloc_fragment((block_size), dtype=dtype)

            T.copy(cu_seqlens[: batch_size_plus_1 - 1], seqlen_start_fragment)
            T.copy(cu_seqlens[1:], seqlen_end_fragment)

            for i in T.Parallel(block_size):
                chunk_offset_fragment[i] = (
                    seqlen_end_fragment[i] - seqlen_start_fragment[i]
                )
                chunk_offset_fragment[i] = (
                    chunk_offset_fragment[i] + chunk_size - 1
                ) // chunk_size
            T.cumsum(src=chunk_offset_fragment, dim=0)

            chunk_offsets[0] = 0
            T.copy(chunk_offset_fragment, chunk_offsets[1:])

    return tilelang_prepare_chunk_offsets_kernel


@tensor_cache
def prepare_chunk_offsets(
    cu_seqlens: torch.LongTensor,
    chunk_size: int,
) -> torch.LongTensor:
    chunk_offsets = torch.empty_like(cu_seqlens)
    tilelang_prepare_chunk_offsets_kernel = tilelang_prepare_chunk_offsets(
        chunk_size=chunk_size,
        block_size=tilelang.next_power_of_2(cu_seqlens.shape[0] - 1),
        dtype=cu_seqlens.dtype,
    )
    tilelang_prepare_chunk_offsets_kernel(cu_seqlens, chunk_offsets)
    return chunk_offsets
