"""Compile-time MP31 shared-memory launch-ceiling guard and residency annotator.

Measured on MTT S5000 (MP31):
- ``192,512 B`` (188 KiB) is the hard per-block shared-memory launch ceiling
  (192,768 B is rejected at launch even though architectural SMEM is 196,608 B).
- ``96,256 B`` (94 KiB) is the 2-CTA/SM dual-residency cliff.

Running this check right after ``MergeSharedMemoryAllocations`` catches
unlaunchable kernels deterministically at compile time instead of failing late
at device launch.
"""

from __future__ import annotations

from typing import Any

from tvm import IRModule, tirx
from tvm.target import Target

from tilelang.musa.target import target_get_arch
from tilelang.transform import PassContext

MP31_SMEM_LAUNCH_CEILING_BYTES = 188 * 1024  # 192,512 B
MP31_SMEM_DUAL_RESIDENCY_BYTES = 94 * 1024   # 96,256 B

CONFIG_ENABLE = "tl.enable_musa_smem_limit_check"
ATTR_SMEM_BYTES = "mp31_smem_bytes"
ATTR_SMEM_CTAS_PER_SM = "mp31_smem_ctas_per_sm"


def _int_value(expr) -> int | None:
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    if isinstance(expr, int):
        return expr
    return None


def _dtype_bytes(dtype_str: str) -> int:
    s = str(dtype_str)
    lanes = 1
    if "x" in s:
        base, lane_s = s.rsplit("x", 1)
        if lane_s.isdigit():
            lanes = int(lane_s)
            s = base
    digits = "".join(ch for ch in s if ch.isdigit())
    bits = int(digits) if digits else 32
    return max(1, (bits * lanes) // 8)


def _blocks_per_sm_by_smem(total_bytes: int, max_bytes: int = MP31_SMEM_LAUNCH_CEILING_BYTES) -> int:
    if total_bytes <= 0:
        return 8
    if total_bytes > max_bytes:
        return 0
    return max(1, max_bytes // total_bytes)


def measure_func_smem_bytes(func: tirx.PrimFunc) -> tuple[int, int]:
    """Return ``(dynamic_smem_bytes, static_smem_bytes)`` for a lowered PrimFunc."""
    dyn_bytes = 0
    static_bytes = 0

    if func.attrs and "dyn_shared_memory_buf" in func.attrs:
        v = _int_value(func.attrs["dyn_shared_memory_buf"])
        if v is not None and v > 0:
            dyn_bytes = max(dyn_bytes, v)

    def visit(node):
        nonlocal dyn_bytes, static_bytes
        if isinstance(node, tirx.AttrStmt) and node.attr_key == "dyn_shared_memory_size":
            v = _int_value(node.value)
            if v is not None and v > 0:
                dyn_bytes = max(dyn_bytes, v)
        elif isinstance(node, tirx.AllocBuffer):
            buf = node.buffer
            scope = buf.scope() if callable(getattr(buf, "scope", None)) else ""
            if scope not in ("shared.dyn", "shared"):
                return
            elems = 1
            for ext in buf.shape:
                iv = _int_value(ext)
                if iv is None or iv <= 0:
                    return
                elems *= iv
            nbytes = elems * _dtype_bytes(str(buf.dtype))
            if scope == "shared.dyn":
                dyn_bytes = max(dyn_bytes, nbytes)
            else:
                static_bytes += nbytes
        elif hasattr(tirx, "Allocate") and isinstance(node, tirx.Allocate):
            try:
                scope = node.buffer_var.type_annotation.storage_scope
            except Exception:
                scope = ""
            if scope not in ("shared.dyn", "shared"):
                return
            elems = 1
            for ext in node.extents:
                iv = _int_value(ext)
                if iv is None or iv <= 0:
                    return
                elems *= iv
            nbytes = elems * _dtype_bytes(str(node.dtype))
            if scope == "shared.dyn":
                dyn_bytes = max(dyn_bytes, nbytes)
            else:
                static_bytes += nbytes

    tirx.stmt_functor.post_order_visit(func.body, visit)
    return dyn_bytes, static_bytes


def verify_mp31_shared_memory_limit(
    mod: IRModule,
    target: Target | None = None,
    max_bytes: int = MP31_SMEM_LAUNCH_CEILING_BYTES,
) -> IRModule:
    """Enforce the MP31 188 KiB (192,512 B) per-block shared-memory launch ceiling."""
    if target is not None and target_get_arch(target) != "mp_31":
        return mod

    ctx = PassContext.current()
    cfg = ctx.config if ctx is not None else {}
    raw_en = cfg.get(CONFIG_ENABLE, True)
    enabled = bool(int(raw_en)) if hasattr(raw_en, "__int__") else bool(raw_en)

    out_funcs: dict[Any, Any] = {}
    changed = False
    for gv, func in mod.functions.items():
        if not isinstance(func, tirx.PrimFunc):
            out_funcs[gv] = func
            continue
        calling_conv = func.attrs.get("calling_conv", None) if func.attrs else None
        is_device_kernel = (calling_conv is None) or (int(calling_conv) == 2)
        if not is_device_kernel:
            out_funcs[gv] = func
            continue

        dyn_b, static_b = measure_func_smem_bytes(func)
        total_b = dyn_b + static_b
        ctas_per_sm = _blocks_per_sm_by_smem(total_b, max_bytes=max_bytes)

        if enabled and total_b > max_bytes:
            kname = (
                str(func.attrs["global_symbol"])
                if func.attrs and "global_symbol" in func.attrs
                else str(gv.name_hint)
            )
            raise ValueError(
                f"MP31 shared memory limit exceeded in kernel '{kname}': "
                f"requires {total_b} B ({total_b / 1024:.1f} KiB; "
                f"dynamic={dyn_b} B, static={static_b} B), exceeding the "
                f"measured {max_bytes} B (188 KiB) per-block launch ceiling on "
                f"MTT S5000 ({MP31_SMEM_DUAL_RESIDENCY_BYTES} B / 94 KiB is the "
                f"2-CTA/SM residency cliff). Reduce tile size, pipeline stages, "
                f"or set '{CONFIG_ENABLE}': False to bypass for IR inspection."
            )

        updated = func.with_attr(ATTR_SMEM_BYTES, tirx.IntImm("int32", total_b))
        updated = updated.with_attr(ATTR_SMEM_CTAS_PER_SM, tirx.IntImm("int32", ctas_per_sm))
        out_funcs[gv] = updated
        changed = True

    if not changed:
        return mod
    return IRModule(out_funcs, attrs=mod.attrs)


__all__ = [
    "MP31_SMEM_LAUNCH_CEILING_BYTES",
    "MP31_SMEM_DUAL_RESIDENCY_BYTES",
    "CONFIG_ENABLE",
    "ATTR_SMEM_BYTES",
    "ATTR_SMEM_CTAS_PER_SM",
    "measure_func_smem_bytes",
    "verify_mp31_shared_memory_limit",
]
