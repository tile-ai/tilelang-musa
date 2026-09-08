"""MUSA peer-memory language intrinsics."""

from __future__ import annotations

from tilelang import tvm as tvm
from tilelang._typing import BufferLikeType, BufferLikeTypeTuple
from tilelang.utils.language import retrieve_ptr
from tvm.tirx import PrimExpr


def ldg128_peer(src: BufferLikeType) -> PrimExpr:
    """Load 128 raw bits from peer-visible global memory."""

    if not isinstance(src, BufferLikeTypeTuple):
        raise TypeError(f"T.ldg128_peer expects Buffer, BufferRegion, or BufferLoad; got {type(src)}: {src}")
    ptr = retrieve_ptr(src, access_type="r")
    return tvm.tirx.call_intrin("uint32x4", tvm.ir.Op.get("tl.musa.ldg128_peer"), ptr)


def stg128_peer(dst: BufferLikeType, value: PrimExpr) -> None:
    """Store 128 bits to peer-visible global memory."""

    if not isinstance(dst, BufferLikeTypeTuple):
        raise TypeError(f"T.stg128_peer expects Buffer, BufferRegion, or BufferLoad; got {type(dst)}: {dst}")
    ptr = retrieve_ptr(dst, access_type="w")
    return tvm.tirx.call_intrin("handle", tvm.ir.Op.get("tl.musa.stg128_peer"), ptr, value)


def peer_warp_reduce128(value: PrimExpr, scratch: BufferLikeType, dtype: str) -> PrimExpr:
    """Reduce one packed 128-bit value across MP31 peer groups."""

    if not isinstance(value, PrimExpr):
        raise TypeError(f"value must be a PrimExpr, got {type(value)}")
    if not isinstance(scratch, BufferLikeTypeTuple):
        raise TypeError(f"scratch must be Buffer, BufferRegion, or BufferLoad; got {type(scratch)}")
    if value.dtype != tvm.DataType("uint32x4"):
        raise ValueError(f"T.peer_warp_reduce128 expects uint32x4 packed input, got {value.dtype}")
    dtype_codes = {"float16": 0, "bfloat16": 1, "float32": 2}
    if dtype not in dtype_codes:
        raise ValueError(f"T.peer_warp_reduce128 does not support dtype {dtype!r}")
    scratch_ptr = retrieve_ptr(scratch, access_type="rw")
    return tvm.tirx.call_intrin(
        "uint32x4",
        tvm.ir.Op.get("tl.musa.peer_warp_reduce128"),
        value,
        scratch_ptr,
        tvm.tirx.IntImm("int32", dtype_codes[dtype]),
    )


def peer_release_fence() -> None:
    """Publish prior MP31 peer writes through SLC."""

    return tvm.tirx.call_intrin("handle", tvm.ir.Op.get("tl.musa.peer_release_fence"))


def peer_signal_store(dst: BufferLikeType, value: PrimExpr | int) -> None:
    """Volatile-store a 32-bit doorbell value to peer-visible memory."""

    if not isinstance(dst, BufferLikeTypeTuple):
        raise TypeError(f"T.peer_signal_store expects Buffer, BufferRegion, or BufferLoad; got {type(dst)}: {dst}")
    ptr = retrieve_ptr(dst, access_type="w")
    return tvm.tirx.call_intrin("handle", tvm.ir.Op.get("tl.musa.peer_signal_store"), ptr, value)


__all__ = [
    "ldg128_peer",
    "peer_release_fence",
    "peer_signal_store",
    "peer_warp_reduce128",
    "stg128_peer",
]
