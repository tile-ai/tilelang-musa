"""MUSA-specific copy frontends."""

from __future__ import annotations

from tilelang import tvm as tvm
from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import tma_copy as _common_tma_copy
from tilelang.language.frame import has_let_value
from tilelang.language.utils import get_extent


def _runtime_pointer_buffer(value: BufferLikeType) -> tvm.tirx.Buffer | None:
    if isinstance(value, tvm.tirx.Buffer):
        buffer = value
    elif isinstance(value, (tvm.tirx.BufferLoad, tvm.tirx.BufferRegion)):
        buffer = value.buffer
    else:
        return None
    return buffer if has_let_value(buffer.data) else None


def _recover_runtime_pointer_region(src: BufferLikeType, dst: BufferLikeType) -> tuple[BufferLikeType, BufferLikeType, bool]:
    runtime_buffer = _runtime_pointer_buffer(src)
    if runtime_buffer is None:
        runtime_buffer = _runtime_pointer_buffer(dst)
    if runtime_buffer is None:
        return src, dst, False

    src_extents = get_extent(src)
    dst_extents = get_extent(dst)
    if src_extents is not None and dst_extents is not None:
        return src, dst, True

    def extent_product(extents) -> tvm.tirx.PrimExpr:
        result = tvm.tirx.const(1, "int32")
        for extent in extents:
            result *= extent
        return result

    if src_extents is not None:
        linear_extent = extent_product(src_extents)
    elif dst_extents is not None:
        linear_extent = extent_product(dst_extents)
    else:
        if not runtime_buffer.shape:
            raise ValueError("MUSA runtime-pointer TME requires a non-scalar buffer")
        linear_extent = runtime_buffer.shape[-1]

    def make_region(value: BufferLikeType) -> BufferLikeType:
        if not isinstance(value, tvm.tirx.BufferLoad):
            return value
        if len(value.indices) != len(value.buffer.shape):
            raise ValueError("MUSA runtime-pointer TME scalar syntax requires one index per buffer dimension")
        ranges = [tvm.ir.Range.from_min_extent(index, 1) for index in value.indices]
        ranges[-1] = tvm.ir.Range.from_min_extent(value.indices[-1], linear_extent)
        return tvm.tirx.BufferRegion(
            value.buffer,
            ranges,
        )

    return make_region(src), make_region(dst), True


def tma_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    barrier=None,
    cluster_mask: int | None = None,
    leader_scope_threads: int | None = None,
    eviction_policy: str | None = None,
    annotations: dict | None = None,
):
    """Issue a MUSA TME copy for contiguous runtime-pointer regions."""

    src, dst, is_runtime_pointer = _recover_runtime_pointer_region(src, dst)
    ann = dict(annotations or {})
    if is_runtime_pointer:
        ann["musa_runtime_pointer_tme"] = 1
    return _common_tma_copy(
        src,
        dst,
        barrier=barrier,
        cluster_mask=cluster_mask,
        leader_scope_threads=leader_scope_threads,
        eviction_policy=eviction_policy,
        annotations=ann,
    )


__all__ = ["tma_copy"]
