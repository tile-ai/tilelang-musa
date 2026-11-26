"""Block-parallel sort, argsort, and topk operations exposed on the TileLang language surface."""

from __future__ import annotations

from tilelang._typing import BufferLikeType
from tilelang.language.builtin import access_ptr
from tilelang.language.common import alloc_shared, copy, macro
from tilelang.language.loop import Parallel
from tilelang.language.tir.op import call_extern
from tilelang.utils.language import _get_buffer, is_shared, retrieve_shape
from tvm import tirx


def _const_int(expr, name: str) -> int:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    raise ValueError(f"{name} must be a compile-time constant integer, got {expr}")


def _next_pow2(n: int) -> int:
    if n <= 1:
        return 1
    return 1 << (n - 1).bit_length()


def _normalize_dim(shape: list, dim: int, op_name: str) -> int:
    rank = len(shape)
    if rank not in (1, 2):
        raise ValueError(f"{op_name} currently supports 1D and 2D buffers, got rank {rank} (shape={shape})")
    if dim < 0:
        dim += rank
    if dim < 0 or dim >= rank:
        raise ValueError(f"{op_name} dim={dim} out of range for shape {shape}")
    return dim


def _buffer_dtype(buf: BufferLikeType) -> str:
    if isinstance(buf, tirx.Buffer):
        return str(buf.dtype)
    return str(_get_buffer(buf).dtype)


@macro
def _sort_1d_macro(
    src: BufferLikeType,
    dst: BufferLikeType,
    valid_n: int,
    padded_n: int,
    descending: bool,
    in_place_shared: bool,
) -> None:
    dtype = _buffer_dtype(src)
    desc_str = "true" if descending else "false"
    fn = f"tl::BlockBitonicSort1D<{desc_str}>::run"
    if in_place_shared:
        call_extern("handle", fn, access_ptr(src, "rw"), valid_n, padded_n)
    else:
        val_smem = alloc_shared((padded_n,), dtype, "shared.dyn")
        if valid_n == padded_n:
            copy(src, val_smem)
        else:
            for i in Parallel(valid_n):
                val_smem[i] = src[i]
        call_extern("handle", fn, access_ptr(val_smem, "rw"), valid_n, padded_n)
        if valid_n == padded_n:
            copy(val_smem, dst)
        else:
            for i in Parallel(valid_n):
                dst[i] = val_smem[i]


@macro
def _sort_2d_macro(
    src: BufferLikeType,
    dst: BufferLikeType,
    h: int,
    w: int,
    dim: int,
    padded_n: int,
    descending: bool,
    in_place_shared: bool,
) -> None:
    dtype = _buffer_dtype(src)
    desc_str = "true" if descending else "false"
    fn = f"tl::BlockBitonicSort2D<{dim}, {desc_str}>::run"
    outer_dim = h if dim == 1 else w
    valid_n = w if dim == 1 else h
    if in_place_shared:
        call_extern("handle", fn, access_ptr(src, "rw"), outer_dim, valid_n, padded_n)
    else:
        padded_shape = (h, padded_n) if dim == 1 else (padded_n, w)
        val_smem = alloc_shared(padded_shape, dtype, "shared.dyn")
        if valid_n == padded_n:
            copy(src, val_smem)
        else:
            for i, j in Parallel(h, w):
                val_smem[i, j] = src[i, j]
        call_extern("handle", fn, access_ptr(val_smem, "rw"), outer_dim, valid_n, padded_n)
        if valid_n == padded_n:
            copy(val_smem, dst)
        else:
            for i, j in Parallel(h, w):
                dst[i, j] = val_smem[i, j]


@macro
def _sort_pairs_1d_macro(
    src: BufferLikeType,
    dst_vals: BufferLikeType | None,
    dst_indices: BufferLikeType | None,
    valid_n: int,
    padded_n: int,
    out_k: int,
    descending: bool,
    idx_dtype: str,
) -> None:
    dtype = _buffer_dtype(src)
    desc_str = "true" if descending else "false"
    fn = f"tl::BlockBitonicSortPairs1D<{desc_str}>::run"
    val_smem = alloc_shared((padded_n,), dtype, "shared.dyn")
    idx_smem = alloc_shared((padded_n,), idx_dtype, "shared.dyn")
    if valid_n == padded_n:
        copy(src, val_smem)
    else:
        for i in Parallel(valid_n):
            val_smem[i] = src[i]
    call_extern("handle", fn, access_ptr(val_smem, "rw"), access_ptr(idx_smem, "rw"), valid_n, padded_n)
    if dst_vals is not None:
        if out_k == padded_n:
            copy(val_smem, dst_vals)
        else:
            for i in Parallel(out_k):
                dst_vals[i] = val_smem[i]
    if dst_indices is not None:
        if out_k == padded_n:
            copy(idx_smem, dst_indices)
        else:
            for i in Parallel(out_k):
                dst_indices[i] = idx_smem[i]


@macro
def _sort_pairs_2d_macro(
    src: BufferLikeType,
    dst_vals: BufferLikeType | None,
    dst_indices: BufferLikeType | None,
    h: int,
    w: int,
    dim: int,
    padded_n: int,
    out_k: int,
    descending: bool,
    idx_dtype: str,
) -> None:
    dtype = _buffer_dtype(src)
    desc_str = "true" if descending else "false"
    fn = f"tl::BlockBitonicSortPairs2D<{dim}, {desc_str}>::run"
    outer_dim = h if dim == 1 else w
    valid_n = w if dim == 1 else h
    padded_shape = (h, padded_n) if dim == 1 else (padded_n, w)
    val_smem = alloc_shared(padded_shape, dtype, "shared.dyn")
    idx_smem = alloc_shared(padded_shape, idx_dtype, "shared.dyn")
    if valid_n == padded_n:
        copy(src, val_smem)
    else:
        for i, j in Parallel(h, w):
            val_smem[i, j] = src[i, j]
    call_extern("handle", fn, access_ptr(val_smem, "rw"), access_ptr(idx_smem, "rw"), outer_dim, valid_n, padded_n)
    out_h = h if dim == 1 else out_k
    out_w = out_k if dim == 1 else w
    if dst_vals is not None:
        if out_k == padded_n:
            copy(val_smem, dst_vals)
        else:
            for i, j in Parallel(out_h, out_w):
                dst_vals[i, j] = val_smem[i, j]
    if dst_indices is not None:
        if out_k == padded_n:
            copy(idx_smem, dst_indices)
        else:
            for i, j in Parallel(out_h, out_w):
                dst_indices[i, j] = idx_smem[i, j]


def sort(
    src: BufferLikeType,
    dst: BufferLikeType | None = None,
    dim: int = -1,
    descending: bool = False,
) -> None:
    """Sort elements of ``src`` along ``dim`` into ``dst`` (in-place if ``dst`` is None)."""
    shape = retrieve_shape(src)
    dim = _normalize_dim(shape, dim, "sort")
    if dst is None:
        dst = src
    else:
        dst_shape = retrieve_shape(dst)
        if len(dst_shape) != len(shape) or any(
            _const_int(dst_shape[i], "dst_shape") != _const_int(shape[i], "src_shape") for i in range(len(shape))
        ):
            raise ValueError(f"sort dst shape {dst_shape} must match src shape {shape}")

    same_buf = isinstance(src, tirx.Buffer) and isinstance(dst, tirx.Buffer) and src.same_as(dst)
    if len(shape) == 1:
        n = _const_int(shape[0], "sort length")
        p = _next_pow2(n)
        in_place_shared = same_buf and is_shared(src) and (n == p)
        _sort_1d_macro(src, dst, n, p, bool(descending), in_place_shared)
    else:
        h = _const_int(shape[0], "sort height")
        w = _const_int(shape[1], "sort width")
        n = w if dim == 1 else h
        p = _next_pow2(n)
        in_place_shared = same_buf and is_shared(src) and (n == p)
        _sort_2d_macro(src, dst, h, w, dim, p, bool(descending), in_place_shared)


def argsort(
    src: BufferLikeType,
    dst: BufferLikeType,
    dim: int = -1,
    descending: bool = False,
) -> None:
    """Compute stable sorted indices of ``src`` along ``dim`` into ``dst``."""
    shape = retrieve_shape(src)
    dim = _normalize_dim(shape, dim, "argsort")
    dst_shape = retrieve_shape(dst)
    if len(dst_shape) != len(shape) or any(
        _const_int(dst_shape[i], "dst_shape") != _const_int(shape[i], "src_shape") for i in range(len(shape))
    ):
        raise ValueError(f"argsort dst shape {dst_shape} must match src shape {shape}")
    idx_dtype = _buffer_dtype(dst)
    if len(shape) == 1:
        n = _const_int(shape[0], "argsort length")
        p = _next_pow2(n)
        _sort_pairs_1d_macro(src, None, dst, n, p, n, bool(descending), idx_dtype)
    else:
        h = _const_int(shape[0], "argsort height")
        w = _const_int(shape[1], "argsort width")
        n = w if dim == 1 else h
        p = _next_pow2(n)
        _sort_pairs_2d_macro(src, None, dst, h, w, dim, p, n, bool(descending), idx_dtype)


def topk(
    src: BufferLikeType,
    dst_values: BufferLikeType | None = None,
    dst_indices: BufferLikeType | None = None,
    k: int | None = None,
    dim: int = -1,
    largest: bool = True,
) -> None:
    """Extract top-``k`` values and/or stable indices of ``src`` along ``dim``."""
    if dst_values is None and dst_indices is None:
        raise ValueError("topk requires at least one of dst_values or dst_indices")
    shape = retrieve_shape(src)
    dim = _normalize_dim(shape, dim, "topk")
    ref_dst = dst_values if dst_values is not None else dst_indices
    ref_shape = retrieve_shape(ref_dst)
    if len(ref_shape) != len(shape):
        raise ValueError(f"topk output rank {len(ref_shape)} must match src rank {len(shape)}")
    out_k = _const_int(ref_shape[dim], "topk k") if k is None else _const_int(k, "topk k")
    sort_n = _const_int(shape[dim], "topk src extent")
    if out_k <= 0 or out_k > sort_n:
        raise ValueError(f"topk k={out_k} must satisfy 1 <= k <= {sort_n}")

    for name, out_buf in (("dst_values", dst_values), ("dst_indices", dst_indices)):
        if out_buf is None:
            continue
        oshape = retrieve_shape(out_buf)
        if len(oshape) != len(shape):
            raise ValueError(f"topk {name} rank must match src rank")
        for axis in range(len(shape)):
            expected = out_k if axis == dim else _const_int(shape[axis], "src_shape")
            actual = _const_int(oshape[axis], f"{name}_shape")
            if actual != expected:
                raise ValueError(f"topk {name} shape {oshape} mismatch on dim {axis}: expected {expected}, got {actual}")

    idx_dtype = _buffer_dtype(dst_indices) if dst_indices is not None else "int32"
    descending = bool(largest)
    if len(shape) == 1:
        n = _const_int(shape[0], "topk length")
        p = _next_pow2(n)
        _sort_pairs_1d_macro(src, dst_values, dst_indices, n, p, out_k, descending, idx_dtype)
    else:
        h = _const_int(shape[0], "topk height")
        w = _const_int(shape[1], "topk width")
        n = w if dim == 1 else h
        p = _next_pow2(n)
        _sort_pairs_2d_macro(src, dst_values, dst_indices, h, w, dim, p, out_k, descending, idx_dtype)
