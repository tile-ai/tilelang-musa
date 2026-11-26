"""Legalize a register-resident (fragment) A operand of MP31 SQMMA GEMMs.

MP31 (MTT S5000, ``__MUSA_ARCH__ == 310``) only exposes shared x shared SQMMA
instructions (register-A ``_RS_mma`` forms are MP32+).  For attention / Mamba /
linear-attention kernels where ``P`` is produced in ``local.fragment`` and
multiplied by a shared-memory ``V`` operand via ``T.gemm(P_frag, V_shared, O)``,
this pass stages ``A`` through a ``shared.dyn`` buffer before
``LayoutInference`` so SQMMA layout inference assigns it the hardware 256B
swizzle layout automatically.

Fragment x fragment ``T.gemm`` calls (MP31 WMMA) are left untouched.
"""

from __future__ import annotations

import tvm
from tvm import tirx

_GEMM_OPS = ("tl.tileop.gemm", "tl.tileop.sqmma_gemm")
_SHARED_SCOPES = ("shared", "shared.dyn")

CONFIG_ENABLE = "tl.enable_musa_legalize_fragment_a_gemm"
CONFIG_REUSE = "tl.enable_musa_gemm_a_staging_reuse"
CONFIG_ENABLE_INT8_FP32 = "tl.enable_musa_legalize_int8_fp32_gemm"


def _op_name(call) -> str:
    op = getattr(call, "op", None)
    return getattr(op, "name", None) or ""


def _is_region(expr) -> bool:
    return (
        isinstance(expr, tirx.Call)
        and len(expr.args) >= 3
        and isinstance(expr.args[0], tirx.BufferLoad)
    )


def _region_buffer(region):
    return region.args[0].buffer


def _retarget_region(region, buffer, access_mask=None):
    args = list(region.args)
    load = args[0]
    mask = args[1] if access_mask is None else tirx.IntImm("int32", access_mask)
    return tirx.Call(
        region.dtype,
        region.op,
        [tirx.BufferLoad(buffer, list(load.indices)), mask] + args[2:],
    )


def _needs_staging(call: tirx.Call) -> bool:
    op_name = _op_name(call)
    if op_name not in _GEMM_OPS:
        return False
    if len(call.args) < 2 or not _is_region(call.args[0]) or not _is_region(call.args[1]):
        return False
    a_buf = _region_buffer(call.args[0])
    b_buf = _region_buffer(call.args[1])
    if a_buf.scope() != "local.fragment":
        return False
    # Explicit sqmma_gemm always requires shared A; plain T.gemm only stages
    # mixed (fragment A, shared B) so fragment x fragment WMMA stays untouched.
    if op_name == "tl.tileop.sqmma_gemm":
        return True
    return b_buf.scope() in _SHARED_SCOPES


def _gemm_split(stmt, staging_for):
    value = getattr(stmt, "value", None)
    if not isinstance(value, tirx.Call) or not _needs_staging(value):
        return None

    a_region = value.args[0]
    a_buf = _region_buffer(a_region)
    staging = staging_for(a_buf)

    copy_op = tvm.ir.Op.get("tl.tileop.copy")
    staging_copy = tirx.Call(
        value.dtype,
        copy_op,
        [
            _retarget_region(a_region, a_buf),
            _retarget_region(a_region, staging, access_mask=2),
        ],
    )
    new_gemm = tirx.Call(
        value.dtype,
        value.op,
        [_retarget_region(a_region, staging)] + list(value.args[1:]),
        value.annotations,
    )
    return [
        tirx.Evaluate(staging_copy),
        tirx.Evaluate(new_gemm),
    ]


def _bool_cfg(config, key: str, default: bool) -> bool:
    raw = config.get(key, default)
    return bool(int(raw)) if hasattr(raw, "__int__") else bool(raw)


def _legalize_func(func, config):
    if not isinstance(func, tirx.PrimFunc):
        return func

    reuse = _bool_cfg(config, CONFIG_REUSE, True)
    staging_by_fragment: dict = {}
    staging_by_shape: dict = {}
    new_buffers: list = []

    def _shape_key(buf):
        return (
            str(buf.dtype),
            tuple(int(s) if isinstance(s, tirx.IntImm) else str(s) for s in buf.shape),
        )

    def staging_for(a_buf):
        if reuse:
            key = _shape_key(a_buf)
            if key in staging_by_shape:
                return staging_by_shape[key]
        else:
            key = a_buf.data
            if key in staging_by_fragment:
                return staging_by_fragment[key]
        name = f"{a_buf.name}_gemm_a_staging"
        staging = tirx.decl_buffer(
            a_buf.shape,
            a_buf.dtype,
            name=name,
            scope="shared.dyn",
        )
        if reuse:
            staging_by_shape[key] = staging
        else:
            staging_by_fragment[key] = staging
        new_buffers.append(staging)
        return staging

    def preorder(stmt):
        if isinstance(stmt, tirx.SeqStmt):
            out = []
            changed = False
            for inner in stmt.seq:
                split = _gemm_split(inner, staging_for)
                if split is None:
                    out.append(inner)
                else:
                    changed = True
                    out.extend(split)
            return tirx.SeqStmt(out) if changed else None

        if isinstance(stmt, tirx.SBlock):
            split = _gemm_split(stmt.body, staging_for)
            if split is None:
                return None
            return tirx.SBlock(
                stmt.iter_vars,
                stmt.reads,
                stmt.writes,
                stmt.name_hint,
                tirx.SeqStmt(split),
                stmt.init,
                stmt.alloc_buffers,
                stmt.match_buffers,
                stmt.annotations,
            )

        if isinstance(stmt, tirx.For):
            split = _gemm_split(stmt.body, staging_for)
            if split is None:
                return None
            return tirx.For(
                stmt.loop_var,
                stmt.min,
                stmt.extent,
                stmt.kind,
                tirx.SeqStmt(split),
                stmt.thread_binding,
                stmt.annotations,
            )

        if isinstance(stmt, tirx.AttrStmt):
            split = _gemm_split(stmt.body, staging_for)
            if split is None:
                return None
            return tirx.AttrStmt(
                stmt.node,
                stmt.attr_key,
                stmt.value,
                tirx.SeqStmt(split),
            )

        if isinstance(stmt, tirx.IfThenElse):
            then_split = _gemm_split(stmt.then_case, staging_for)
            else_split = (
                _gemm_split(stmt.else_case, staging_for)
                if stmt.else_case is not None
                else None
            )
            if then_split is None and else_split is None:
                return None
            return tirx.IfThenElse(
                stmt.condition,
                tirx.SeqStmt(then_split) if then_split is not None else stmt.then_case,
                tirx.SeqStmt(else_split) if else_split is not None else stmt.else_case,
            )
        return None

    body = tirx.stmt_functor.ir_transform(func.body, preorder, None)
    if not new_buffers:
        return func

    placed = False

    def place_allocs(stmt):
        nonlocal placed
        if isinstance(stmt, tirx.SBlock) and not placed:
            placed = True
            return tirx.SBlock(
                stmt.iter_vars,
                stmt.reads,
                stmt.writes,
                stmt.name_hint,
                stmt.body,
                stmt.init,
                list(stmt.alloc_buffers) + list(new_buffers),
                stmt.match_buffers,
                stmt.annotations,
            )
        return None

    body = tirx.stmt_functor.ir_transform(body, place_allocs, None)
    return tirx.PrimFunc(
        func.params,
        body,
        func.ret_type,
        func.buffer_map,
        func.attrs,
        func.span,
    )


def legalize_fragment_a_gemm(mod, config=None):
    """Stage fragment A operands of MP31 SQMMA GEMMs through shared.dyn memory."""
    config = config or {}
    if not _bool_cfg(config, CONFIG_ENABLE, True):
        return mod
    functions = {gv: _legalize_func(func, config) for gv, func in mod.functions.items()}
    return tvm.IRModule(functions, attrs=mod.attrs)


def _needs_int8_fp32_legalize(call: tirx.Call) -> bool:
    op_name = _op_name(call)
    if op_name not in _GEMM_OPS:
        return False
    if len(call.args) < 10:
        return False
    if not (_is_region(call.args[0]) and _is_region(call.args[1]) and _is_region(call.args[2])):
        return False
    a_buf = _region_buffer(call.args[0])
    b_buf = _region_buffer(call.args[1])
    c_buf = _region_buffer(call.args[2])
    if a_buf.scope() not in _SHARED_SCOPES or b_buf.scope() not in _SHARED_SCOPES:
        return False
    if c_buf.scope() != "local.fragment":
        return False
    a_dt = str(a_buf.dtype)
    b_dt = str(b_buf.dtype)
    c_dt = str(c_buf.dtype)
    return a_dt in ("int8", "uint8") and b_dt == a_dt and c_dt == "float32"


def _int8_fp32_split(stmt, accum_for):
    value = getattr(stmt, "value", None)
    if not isinstance(value, tirx.Call) or not _needs_int8_fp32_legalize(value):
        return None

    c_region = value.args[2]
    c_buf = _region_buffer(c_region)
    c_i32 = accum_for(c_buf)
    clear_accum = value.args[9]

    new_args = list(value.args)
    new_args[2] = _retarget_region(c_region, c_i32)
    new_args[9] = tirx.const(1, "bool")
    new_gemm = tirx.Call(value.dtype, value.op, new_args, value.annotations)

    # Build T.Parallel epilogue converting int32 SQMMA accumulator into float32 C.
    c_load = c_region.args[0]
    base_indices = list(c_load.indices)
    extents = list(c_region.args[2:])
    rank = len(base_indices)
    loop_vars = []
    idx_exprs = list(base_indices)
    for axis in range(rank - 2, rank):
        lv = tirx.Var(f"_i8f32_{axis}", "int32")
        loop_vars.append((lv, extents[axis]))
        idx_exprs[axis] = base_indices[axis] + lv

    i32_val = tirx.Cast("float32", tirx.BufferLoad(c_i32, idx_exprs))
    c_val = tirx.BufferLoad(c_buf, idx_exprs)
    if isinstance(clear_accum, tirx.IntImm):
        rhs = i32_val if int(clear_accum.value) != 0 else (c_val + i32_val)
    else:
        rhs = tirx.Select(clear_accum, i32_val, c_val + i32_val)

    body = tirx.BufferStore(c_buf, rhs, idx_exprs)
    for lv, ext in reversed(loop_vars):
        body = tirx.For(lv, tirx.IntImm("int32", 0), ext, tirx.ForKind.PARALLEL, body)

    return [tirx.Evaluate(new_gemm), body]


def _legalize_int8_fp32_func(func, config):
    if not isinstance(func, tirx.PrimFunc):
        return func

    accum_by_shape: dict = {}
    new_buffers: list = []

    def _shape_key(buf):
        return tuple(int(s) if isinstance(s, tirx.IntImm) else str(s) for s in buf.shape)

    def accum_for(c_buf):
        key = _shape_key(c_buf)
        if key in accum_by_shape:
            return accum_by_shape[key]
        buf = tirx.decl_buffer(
            c_buf.shape,
            "int32",
            name=f"{c_buf.name}_i32_accum",
            scope="local.fragment",
        )
        accum_by_shape[key] = buf
        new_buffers.append(buf)
        return buf

    def preorder(stmt):
        if isinstance(stmt, tirx.SeqStmt):
            out = []
            changed = False
            for inner in stmt.seq:
                split = _int8_fp32_split(inner, accum_for)
                if split is None:
                    out.append(inner)
                else:
                    changed = True
                    out.extend(split)
            return tirx.SeqStmt(out) if changed else None

        if isinstance(stmt, tirx.SBlock):
            split = _int8_fp32_split(stmt.body, accum_for)
            if split is None:
                return None
            return tirx.SBlock(
                stmt.iter_vars,
                stmt.reads,
                stmt.writes,
                stmt.name_hint,
                tirx.SeqStmt(split),
                stmt.init,
                stmt.alloc_buffers,
                stmt.match_buffers,
                stmt.annotations,
            )

        if isinstance(stmt, tirx.For):
            split = _int8_fp32_split(stmt.body, accum_for)
            if split is None:
                return None
            return tirx.For(
                stmt.loop_var,
                stmt.min,
                stmt.extent,
                stmt.kind,
                tirx.SeqStmt(split),
                stmt.thread_binding,
                stmt.annotations,
            )

        if isinstance(stmt, tirx.AttrStmt):
            split = _int8_fp32_split(stmt.body, accum_for)
            if split is None:
                return None
            return tirx.AttrStmt(
                stmt.node,
                stmt.attr_key,
                stmt.value,
                tirx.SeqStmt(split),
            )

        if isinstance(stmt, tirx.IfThenElse):
            then_split = _int8_fp32_split(stmt.then_case, accum_for)
            else_split = (
                _int8_fp32_split(stmt.else_case, accum_for)
                if stmt.else_case is not None
                else None
            )
            if then_split is None and else_split is None:
                return None
            return tirx.IfThenElse(
                stmt.condition,
                tirx.SeqStmt(then_split) if then_split is not None else stmt.then_case,
                tirx.SeqStmt(else_split) if else_split is not None else stmt.else_case,
            )
        return None

    body = tirx.stmt_functor.ir_transform(func.body, preorder, None)
    if not new_buffers:
        return func

    placed = False

    def place_allocs(stmt):
        nonlocal placed
        if isinstance(stmt, tirx.SBlock) and not placed:
            placed = True
            return tirx.SBlock(
                stmt.iter_vars,
                stmt.reads,
                stmt.writes,
                stmt.name_hint,
                stmt.body,
                stmt.init,
                list(stmt.alloc_buffers) + list(new_buffers),
                stmt.match_buffers,
                stmt.annotations,
            )
        return None

    body = tirx.stmt_functor.ir_transform(body, place_allocs, None)
    return tirx.PrimFunc(
        func.params,
        body,
        func.ret_type,
        func.buffer_map,
        func.attrs,
        func.span,
    )


def legalize_int8_fp32_gemm(mod, config=None):
    """Legalize int8/uint8 SQMMA GEMMs accumulating into float32 via int32 accum + cast."""
    config = config or {}
    if not _bool_cfg(config, CONFIG_ENABLE_INT8_FP32, True):
        return mod
    functions = {gv: _legalize_int8_fp32_func(func, config) for gv, func in mod.functions.items()}
    return tvm.IRModule(functions, attrs=mod.attrs)


__all__ = [
    "legalize_fragment_a_gemm",
    "legalize_int8_fp32_gemm",
    "CONFIG_ENABLE",
    "CONFIG_REUSE",
    "CONFIG_ENABLE_INT8_FP32",
]
