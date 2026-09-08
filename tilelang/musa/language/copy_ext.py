"""MUSA-specific copy frontends."""

from __future__ import annotations

from tilelang import tvm as tvm
from tilelang._typing import BufferLikeType
from tilelang.language.copy_op import copy as _common_copy
from tilelang.language.copy_op import tma_copy as _common_tma_copy
from tilelang.language.frame import has_let_value
from tilelang.language.utils import get_extent

from .memory import (
    _COHERENCE,
    _INNER_PERSISTENCE,
    _L2_POLICY,
    _OUTER_PERSISTENCE,
    normalize_lsu_hint,
)


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    eviction_policy: str | None = None,
    prefer_instruction: str | None = None,
    annotations: dict | None = None,
    loop_layout=None,
    inner_cache_policy: str | int | None = None,
    outer_cache_policy: str | int | None = None,
    chrnt: str | int | None = None,
    l2: str | int | None = None,
    is_volatile: bool | None = None,
):
    """Copy data with optional MUSA LSU load-cache policies."""

    lsu_mode = any(
        value is not None
        for value in (
            inner_cache_policy,
            outer_cache_policy,
            chrnt,
            l2,
            is_volatile,
        )
    )
    ann = dict(annotations or {})
    if lsu_mode:
        if eviction_policy is not None or prefer_instruction in ("tma", "cp_async"):
            raise ValueError("MUSA LSU cache hints require normal synchronous copy")
        if is_volatile is not None and not isinstance(is_volatile, bool):
            raise TypeError(f"is_volatile must be bool, got {type(is_volatile)}")
        ann.update(
            {
                "musa_lsu_cache_hint": 1,
                "musa_lsu_inner": normalize_lsu_hint(
                    4 if inner_cache_policy is None else inner_cache_policy,
                    "inner_cache_policy",
                    _INNER_PERSISTENCE,
                    5,
                ),
                "musa_lsu_outer": normalize_lsu_hint(
                    2 if outer_cache_policy is None else outer_cache_policy,
                    "outer_cache_policy",
                    _OUTER_PERSISTENCE,
                    3,
                ),
                "musa_lsu_chrnt": normalize_lsu_hint(
                    0 if chrnt is None else chrnt,
                    "chrnt",
                    _COHERENCE,
                    1,
                ),
                "musa_lsu_l2": normalize_lsu_hint(
                    0 if l2 is None else l2,
                    "l2",
                    _L2_POLICY,
                    1,
                ),
                "musa_lsu_volatile": int(bool(is_volatile)),
            }
        )
        prefer_instruction = "sync"

    return _common_copy(
        src,
        dst,
        coalesced_width=coalesced_width,
        disable_tma=disable_tma,
        eviction_policy=eviction_policy,
        prefer_instruction=prefer_instruction,
        annotations=ann,
        loop_layout=loop_layout,
    )


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


__all__ = ["copy", "tma_copy"]
