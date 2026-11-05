"""Explicit accelerated operations exposed on the TileLang language surface."""

from tvm import tirx as tir
from tvm.tirx import PrimExpr


def _buffer_region_to_load(region: tir.BufferRegion) -> tir.BufferLoad:
    indices = []
    for range_ in region.region:
        extent = range_.extent
        if isinstance(extent, tir.IntImm) and extent.value == 1:
            indices.append(range_.min)
        else:
            indices.append(tir.Ramp(range_.min, 1, extent))
    return tir.BufferLoad(region.buffer, indices)


def _dtype_lanes(dtype: str, base: str) -> int:
    if dtype == base:
        return 1
    prefix = f"{base}x"
    if dtype.startswith(prefix):
        return int(dtype[len(prefix) :])
    raise ValueError(f"Expected dtype '{base}' or '{base}xN', got '{dtype}'")


def mul_half_float_to_bfloat16_x4(x: PrimExpr, y: PrimExpr) -> PrimExpr:
    """Accelerated half * float -> bfloat16 multiply backed by the x4 helper.

    `x` must be float16x4 and `y` must be float32x4. The return value is
    bfloat16x4.
    """
    x = _buffer_region_to_load(x) if isinstance(x, tir.BufferRegion) else tir.convert(x)
    y = _buffer_region_to_load(y) if isinstance(y, tir.BufferRegion) else tir.convert(y)
    x_lanes = _dtype_lanes(str(x.dtype), "float16")
    y_lanes = _dtype_lanes(str(y.dtype), "float32")
    if x_lanes != 4:
        raise ValueError(f"Expected lhs dtype to be float16x4, got {x.dtype}")
    if y_lanes != 4:
        raise ValueError(f"Expected rhs dtype to be float32x4, got {y.dtype}")
    out_dtype = "bfloat16x4"
    return tir.call_intrin(out_dtype, tir.op.Op.get("tl.mul_half_float_to_bfloat16_x4"), x, y)


__all__ = [
    "mul_half_float_to_bfloat16_x4",
]
