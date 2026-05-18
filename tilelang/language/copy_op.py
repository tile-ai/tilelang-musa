"""Copy operations exposed on the TileLang language surface."""

from __future__ import annotations
from typing import Any, Literal
from tilelang._typing import BufferLikeType, BarrierType
from tilelang.language.frame import get_let_value, has_let_value
from tilelang.utils.language import (
    to_buffer_region,
    legalize_pairwise_extents,
)
from tilelang.language.utils import get_extent
from tvm import ir, tir


EvictionPolicy = Literal["evict_normal", "evict_first", "evict_last"]
CachePolicy = Literal["cache_none", "cache_once", "cache_normal", "cache_persist"]
_EVICTION_POLICY_MAP = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}
_CACHE_POLICY_MAP = {
    "cache_none": 0,
    "cache_once": 1,
    "cache_normal": 2,
    "cache_persist": 3,
}
_EVICTION_TO_CACHE_POLICY = {
    _EVICTION_POLICY_MAP["evict_normal"]: _CACHE_POLICY_MAP["cache_normal"],
    _EVICTION_POLICY_MAP["evict_first"]: _CACHE_POLICY_MAP["cache_once"],
    _EVICTION_POLICY_MAP["evict_last"]: _CACHE_POLICY_MAP["cache_persist"],
}


def _set_tma_cache_annotations(ann: dict) -> None:
    if "eviction_policy" in ann:
        if "inner_cache_policy" in ann or "outer_cache_policy" in ann:
            raise ValueError("eviction_policy cannot be combined with inner_cache_policy or outer_cache_policy.")
    else:
        ann["eviction_policy"] = _EVICTION_POLICY_MAP["evict_normal"]

    if "inner_cache_policy" not in ann:
        ann["inner_cache_policy"] = _EVICTION_TO_CACHE_POLICY[ann["eviction_policy"]]
    if "outer_cache_policy" not in ann:
        ann["outer_cache_policy"] = _EVICTION_TO_CACHE_POLICY[ann["eviction_policy"]]


def _manual_tma_barrier(expr: tir.PrimExpr) -> tir.Call:
    return tir.call_intrin("handle", tir.op.Op.get("tl.manual_tma_barrier"), expr)


def _normalize_copy_regions(
    src: BufferLikeType, dst: BufferLikeType
) -> tuple[
    tir.BufferRegion | tir.BufferLoad | tir.Buffer,
    tir.BufferRegion | tir.BufferLoad | tir.Buffer,
]:
    # If both side are buffers, we should make sure their shapes are equal
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    src_is_scalar_load = src_extent is None and isinstance(src, tir.BufferLoad)
    dst_is_scalar_load = dst_extent is None and isinstance(dst, tir.BufferLoad)

    # copy(buffer_a[i], buffer_b[i]) where both are BufferLoad nodes
    # In this case, lower it to a simple BufferStore: buffer_b[i] = buffer_a[i]
    if src_is_scalar_load and dst_is_scalar_load:
        return src, dst

    assert src_extent or dst_extent, "Can't deduce copy extents from args. Both src and dst miss extents info."
    # Treat missing extent as length-matched ones for convenience. This provides limited
    # broadcasting-like syntactic sugar, but does not implement general broadcasting support.
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    # Align and broadcast extents from the right (tail) side.
    # This is majorly for supporting some syntactic sugar, not the whole broadcasting ability of copy op.
    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    # Use legalized extents for src and dst respectively.
    src = to_buffer_region(src, access_type="r", extents=src_extent)
    dst = to_buffer_region(dst, access_type="w", extents=dst_extent)
    return src, dst


def copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    disable_tma: bool = False,
    force_async_copy: bool = False,
    eviction_policy: EvictionPolicy | None = None,
    inner_cache_policy: CachePolicy | None = None,
    outer_cache_policy: CachePolicy | None = None,
    src_robust_desc: tir.PrimExpr | None = None,
    barrier: BarrierType | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt:
    """Copy data between memory regions.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], keyword-only): Width for coalesced memory access. Defaults to None.
        disable_tma (bool, keyword-only): Whether to disable TMA acceleration. Defaults to False.
        force_async_copy (bool, keyword-only): Force MUSA async-copy lowering for
            this copy site. Defaults to False.
        eviction_policy (Optional[str], keyword-only): NV-compatible cache eviction policy.
            Defaults to None.
        inner_cache_policy (Optional[str], keyword-only): MUSA inner-cache policy.
            One of ``cache_none``, ``cache_once``, ``cache_normal``, or ``cache_persist``.
            Defaults to ``cache_normal`` when omitted.
        outer_cache_policy (Optional[str], keyword-only): MUSA outer-cache policy.
            One of ``cache_none``, ``cache_once``, ``cache_normal``, or ``cache_persist``.
            Defaults to ``cache_normal`` when omitted.
        src_robust_desc (Optional[tir.PrimExpr], keyword-only): MUSA robust source
            descriptor created by `T.make_robust_desc(addr, size_bytes)`.
        barrier (Optional[BarrierType], keyword-only):
            User-managed `shared.barrier` for TMA load synchronization. When
            provided, TileLang emits the TMA load against this barrier and
            leaves `T.barrier_arrive` / `T.barrier_wait` under user control.
        annotations (Optional[dict], keyword-only): Additional annotations dict. If provided,
            coalesced_width, disable_tma, force_async_copy, src_robust_desc, and
            cache-policy fields can also be specified here.
            Values in annotations take precedence over individual arguments.
        loop_layout (Optional[Fragment], keyword-only): A parallel loop layout hint for the SIMT copy
            (only valid for normal SIMT copy; incompatible with TMA/LDSM/STSM/TMem). When provided,
            it is attached to the outermost parallel loop generated by this copy.

    Raises:
        TypeError: If copy extents cannot be deduced from arguments

    Returns:
        tir.Call: A handle to the copy operation

    Range handling notes:
    - Accepts `Buffer`/`BufferRegion`/`BufferLoad` on either side. Extents are
      derived as follows: `Buffer -> shape`, `BufferRegion -> [r.extent]`,
      `BufferLoad -> extents from its inferred/encoded region`.
    - Normally, we require the extents of both sides to be the same. If they
      differ, the copy instruction follows an internal rule to select one side
      as the base range and create iteration space. This may generate unexpected
      code. And if some dimensions are 1, unexpected errors may happen.
    - Small Optimization: If both `src` and `dst` are scalar `BufferLoad` without
      region extents, lowers to a direct store: `dst[...] = src[...]`.
    - Syntactic Sugar: TileLang supports passing the head address of a buffer to represent
      the whole buffer if there are no ambiguity. For example, T.copy(A, A_shared[i, j]).
      To support this, we need some special shape checking. But remember currently we don't
      support something like "broadcast".
    - The finalized extents are encoded with `tl.region` via `to_buffer_region`
      and passed through to the backend; low-level loop construction and any
      scope-specific decisions happen during lowering.
    """
    src, dst = _normalize_copy_regions(src, dst)
    if isinstance(src_robust_desc, tir.Var) and has_let_value(src_robust_desc):
        src_robust_desc = get_let_value(src_robust_desc)
    if src_robust_desc is not None and not (
        isinstance(src_robust_desc, tir.Call)
        and src_robust_desc.op.same_as(tir.op.Op.get("tl.make_robust_desc"))
        and len(src_robust_desc.args) == 2
    ):
        raise ValueError("src_robust_desc must be created by T.make_robust_desc(addr, size_bytes)")
    barrier_expr = None
    if barrier is not None:
        if isinstance(barrier, tir.Buffer):
            barrier_load = tir.BufferLoad(barrier, [0])
        elif isinstance(barrier, tir.BufferLoad):
            barrier_load = barrier
        else:
            raise TypeError(f"barrier must be a tir.Buffer or tir.BufferLoad, but got {type(barrier)}")
        barrier_buf = barrier_load.buffer
        if barrier_buf.scope() != "shared.barrier":
            raise ValueError(f"barrier must be in scope 'shared.barrier', but got scope {barrier_buf.scope()!r}")
        if len(barrier_buf.shape) != 1:
            raise ValueError(f"barrier must come from a 1-D buffer, but got shape {barrier_buf.shape}")
        if len(barrier_load.indices) != 1:
            raise ValueError(f"barrier index must be 1-D, but got {len(barrier_load.indices)} indices")
        barrier_expr = _manual_tma_barrier(barrier_load)

    if isinstance(src, tir.BufferLoad) and isinstance(dst, tir.BufferLoad):
        if barrier_expr is not None:
            raise ValueError("barrier is only supported for region/bulk TMA copy, not scalar copy")
        body: tir.Stmt = tir.BufferStore(dst.buffer, src, dst.indices)
        if src_robust_desc is not None:
            body = tir.AttrStmt(
                src.buffer.data,
                "tl.source_robust_desc",
                src_robust_desc,
                body,
            )
        if force_async_copy:
            body = tir.AttrStmt(
                tir.IntImm("int32", 0),
                "tl.force_async_copy",
                tir.IntImm("int32", 1),
                body,
            )
        return body

    # Build annotations dict
    ann = annotations.copy() if annotations else {}

    # Individual arguments take lower precedence than annotations
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if "disable_tma" not in ann and disable_tma:
        ann["disable_tma"] = disable_tma
    if "force_async_copy" not in ann and force_async_copy:
        ann["force_async_copy"] = tir.IntImm("int32", 1)
    if "src_robust_desc" not in ann and src_robust_desc is not None:
        ann["src_robust_desc"] = src_robust_desc
    if "barrier" not in ann and barrier_expr is not None:
        ann["barrier"] = barrier_expr

    if "eviction_policy" not in ann and eviction_policy is not None:
        ann["eviction_policy"] = _EVICTION_POLICY_MAP[eviction_policy]
    if "inner_cache_policy" not in ann and inner_cache_policy is not None:
        ann["inner_cache_policy"] = _CACHE_POLICY_MAP[inner_cache_policy]
    if "outer_cache_policy" not in ann and outer_cache_policy is not None:
        ann["outer_cache_policy"] = _CACHE_POLICY_MAP[outer_cache_policy]
    _set_tma_cache_annotations(ann)

    # Parallel loop layout hint (Fragment). Mirrors T.Parallel(loop_layout=...)
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.copy"), src, dst, annotations=ann if ann else None)


def async_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    coalesced_width: int | None = None,
    annotations: dict | None = None,
    loop_layout: Any | None = None,
) -> tir.PrimExpr | tir.Stmt:
    """Asynchronous copy primitive lowered through cp.async.

    This operator is intended for explicitly asynchronous global->shared copy.
    The backend enforces cp.async constraints and emits:
      `ptx_cp_async(...)` + `ptx_commit_group()`.
    No wait is auto-inserted for `T.async_copy`; synchronization is explicit.

    Args:
        src (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Source memory region
        dst (Union[tir.Buffer, tir.BufferLoad, tir.BufferRegion]): Destination memory region
        coalesced_width (Optional[int], keyword-only): Width for coalesced memory access. Defaults to None.
        annotations (Optional[dict], keyword-only): Additional annotations dict.
        loop_layout (Optional[Fragment], keyword-only): A parallel loop layout hint for the SIMT copy loop.

    Returns:
        tir.Call: A handle to the async copy operation
    """
    src, dst = _normalize_copy_regions(src, dst)
    if isinstance(src, tir.BufferLoad) and isinstance(dst, tir.BufferLoad):
        return tir.BufferStore(dst.buffer, src, dst.indices)

    ann = annotations.copy() if annotations else {}
    if "coalesced_width" not in ann and coalesced_width is not None:
        ann["coalesced_width"] = coalesced_width
    if loop_layout is not None and "parallel_loop_layout" not in ann:
        ann["parallel_loop_layout"] = loop_layout

    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.async_copy"),
        src,
        dst,
        annotations=ann if ann else None,
    )


def tma_copy(
    src: BufferLikeType,
    dst: BufferLikeType,
    *,
    barrier,
    eviction_policy: EvictionPolicy | None = None,
    inner_cache_policy: CachePolicy | None = None,
    outer_cache_policy: CachePolicy | None = None,
    annotations: dict | None = None,
) -> tir.PrimExpr | tir.Stmt:
    """TMA copy — issues arrive_and_expect_tx + tma_load, no wait.

    Unlike T.copy() which emits a full synchronous TMA sequence (arrive + load + wait),
    T.tma_copy() emits only the producer part (arrive_and_expect_tx + tma_load).
    The user manages synchronization explicitly via T.mbarrier_wait_parity().

    Args:
        src: Source memory region (global or shared)
        dst: Destination memory region (shared or global)
        barrier: Mbarrier (from T.alloc_barrier()) for TMA synchronization.
            The TMA load will arrive at this barrier with expected byte count.
            The user must wait on the same barrier via T.mbarrier_wait_parity().
        eviction_policy: NV-compatible cache eviction policy. Defaults to None.
        inner_cache_policy: MUSA inner-cache policy. Defaults to
            ``cache_normal`` when omitted.
        outer_cache_policy: MUSA outer-cache policy. Defaults to
            ``cache_normal`` when omitted.
        annotations: Additional annotations dict. Barrier and cache-policy
            fields in annotations take precedence over individual arguments.

    Returns:
        tir.Call: A handle to the tma_copy operation
    """
    # If both side are buffers, we should make sure their shapes are equal
    if isinstance(src, tir.Buffer) and isinstance(dst, tir.Buffer):
        ir.assert_structural_equal(src.shape, dst.shape)

    src_extent = get_extent(src)
    dst_extent = get_extent(dst)

    assert src_extent or dst_extent, "Can't deduce copy extents from args. Both src and dst miss extents info."
    src_extent = list(src_extent) if src_extent else [1] * len(dst_extent)
    dst_extent = list(dst_extent) if dst_extent else [1] * len(src_extent)

    src_extent, dst_extent = legalize_pairwise_extents(src_extent, dst_extent)

    src = to_buffer_region(src, access_type="r", extents=src_extent)
    dst = to_buffer_region(dst, access_type="w", extents=dst_extent)

    ann = annotations.copy() if annotations else {}

    from .builtin import _mbar_to_buffer_load

    if "barrier" not in ann:
        ann["barrier"] = _mbar_to_buffer_load(barrier)

    if "eviction_policy" not in ann and eviction_policy is not None:
        ann["eviction_policy"] = _EVICTION_POLICY_MAP[eviction_policy]
    if "inner_cache_policy" not in ann and inner_cache_policy is not None:
        ann["inner_cache_policy"] = _CACHE_POLICY_MAP[inner_cache_policy]
    if "outer_cache_policy" not in ann and outer_cache_policy is not None:
        ann["outer_cache_policy"] = _CACHE_POLICY_MAP[outer_cache_policy]
    _set_tma_cache_annotations(ann)

    return tir.call_intrin("handle", tir.op.Op.get("tl.tileop.tma_copy"), src, dst, annotations=ann if ann else None)


def c2d_im2col(
    img: BufferLikeType,
    col: BufferLikeType,
    nhw_step: tir.PrimExpr,
    c_step: tir.PrimExpr,
    kernel: int,
    stride: int,
    dilation: int,
    pad: int,
    eviction_policy: EvictionPolicy | None = None,
) -> tir.PrimExpr:
    """Perform im2col transformation for 2D convolution.

    Args:
        img (tir.Buffer): Input image buffer
        col (tir.Buffer): Output column buffer
        nhw_step (tir.PrimExpr): Step size for batch and spatial dimensions
        c_step (tir.PrimExpr): Step size for channel dimension
        kernel (int): Kernel size
        stride (int): Stride of the convolution
        dilation (int): Dilation rate
        pad (int): Padding size

    Returns:
        tir.Call: A handle to the im2col operation
    """
    if eviction_policy is None:
        eviction_policy = 0
    else:
        eviction_policy = {"evict_normal": 0, "evict_first": 1, "evict_last": 2}[eviction_policy]
    img_region = to_buffer_region(img, access_type="r")
    col_region = to_buffer_region(col, access_type="w")
    return tir.call_intrin(
        "handle",
        tir.op.Op.get("tl.tileop.c2d_im2col"),
        img_region,
        col_region,
        nhw_step,
        c_step,
        kernel,
        stride,
        dilation,
        pad,
        eviction_policy,
    )
