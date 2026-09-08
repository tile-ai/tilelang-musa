"""MUSA memory-access language intrinsics."""

from __future__ import annotations

from tilelang import tvm as tvm
from tvm.tirx import BufferLoad, PrimExpr


_INNER_PERSISTENCE = {
    "cache_none": 0,
    "cache_once": 1,
    "cache_normal": 2,
    "cache_persist": 3,
    "no_override": 4,
    "last_use": 5,
}
_OUTER_PERSISTENCE = {
    "cache_none": 0,
    "cache_once": 1,
    "cache_normal": 2,
    "cache_persist": 3,
}
_COHERENCE = {"mcu": 0, "slc": 1, "l1": 0, "l2_l3": 1}
_L2_POLICY = {"new_alloc": 0, "new": 0, "bypass": 1}


def normalize_lsu_hint(
    value: str | int | tvm.tirx.IntImm,
    name: str,
    mapping: dict[str, int],
    maximum: int,
) -> int:
    """Normalize one LSU policy to its immediate MTCC encoding."""

    if isinstance(value, tvm.tirx.IntImm):
        value = int(value.value)
    elif isinstance(value, int) and not isinstance(value, bool):
        value = int(value)
    elif isinstance(value, str):
        if value not in mapping:
            choices = ", ".join(sorted(mapping))
            raise ValueError(f"Unsupported {name} {value!r}; expected one of: {choices}")
        value = mapping[value]
    else:
        raise TypeError(f"{name} must be an integer immediate or policy string, got {type(value)}")
    if value < 0 or value > maximum:
        raise ValueError(f"{name} must be in [0, {maximum}], got {value}")
    return value


def lsu_ld_cache_hint(
    src: BufferLoad,
    inner_persistence: str | int = 4,
    outer_persistence: str | int = 2,
    chrnt: str | int = 0,
    l2: str | int = 0,
    is_volatile: bool = False,
) -> PrimExpr:
    """Load one global element with immediate MUSA LSU cache policies."""

    if not isinstance(src, BufferLoad):
        raise TypeError("T.lsu_ld_cache_hint expects a BufferLoad such as x[i]")
    if len(src.indices) != 1:
        raise ValueError("T.lsu_ld_cache_hint currently supports flattened 1D accesses only")
    if not isinstance(is_volatile, bool):
        raise TypeError(f"is_volatile must be bool, got {type(is_volatile)}")

    inner = normalize_lsu_hint(inner_persistence, "inner_persistence", _INNER_PERSISTENCE, 5)
    outer = normalize_lsu_hint(outer_persistence, "outer_persistence", _OUTER_PERSISTENCE, 3)
    coherence = normalize_lsu_hint(chrnt, "chrnt", _COHERENCE, 1)
    l2_policy = normalize_lsu_hint(l2, "l2", _L2_POLICY, 1)
    op_name = "tl.musa.lsu_ld_volatile_cache_hint" if is_volatile else "tl.musa.lsu_ld_cache_hint"
    return tvm.tirx.call_intrin(
        str(src.dtype),
        tvm.ir.Op.get(op_name),
        src,
        inner,
        outer,
        coherence,
        l2_policy,
    )


__all__ = ["lsu_ld_cache_hint"]
