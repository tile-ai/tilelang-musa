"""MUSA memory-bound kernel optimizations on lowered TIR.

Two general, semantics-preserving transformations that run after VectorizeLoop
and UnrollLoop (before SplitHostDevice):

1. ``fuse_round_to_int_cast``:
   On MP31 (MTT S5000), ``nearbyintf(float) -> float`` followed by
   ``(int8_t/int16_t/int32_t)`` is ~2x slower than the single hardware
   round-to-nearest-even conversion ``__float2int_rn(float) -> int``.
   Because ``min(round_even(x), c) == round_even(min(x, c))`` and
   ``max(round_even(x), c) == round_even(max(x, c))`` for every finite float
   ``x`` and exact-integer float ``c``, this rewrite pushes ``tirx.round`` /
   ``tirx.nearbyint`` past exact-integer ``Min``/``Max`` clamps and lowers
   ``Cast(int{8,16,32}, round(f32))`` to ``Cast(int{8,16,32}, __float2int_rn(f32))``
   while preserving vectorized stores emitted by ``VectorizeLoop``.

2. ``guard_uniform_global_scalar_stores``:
   When all threads in a CTA (or all ``K`` threads sharing ``floordiv(tx, K)``)
   unconditionally execute a scalar ``BufferStore`` to the exact same global
   memory address, concurrent colliding writes serialize on L2 sectors (e.g.
   256-way or 512-way scale writes in quantization/reduction kernels).
   Wrapping such unpredicated stores in ``if (tx == 0)`` (or
   ``if (tx % K == 0)``) preserves semantics for every race-free kernel and
   eliminates colliding write traffic.
"""

from __future__ import annotations

import math
from typing import Optional

import tvm
from tvm import tirx

_THREAD_TAGS = ("threadIdx.x", "threadIdx.y", "threadIdx.z")
_ALLOCATE_CLS = getattr(tirx, "Allocate", None)
_LET_STMT_CLS = getattr(tirx, "LetStmt", None)
_BIND_CLS = getattr(tirx, "Bind", None)
_ROUND_OPS = ("tirx.round", "tirx.nearbyint")
_MAX_EXACT_INT32_F32 = 2147483520.0  # largest float32 <= INT32_MAX


def _op_name(call: tirx.Call) -> str:
    op = getattr(call, "op", None)
    return getattr(op, "name", None) or ""


def _base_dtype(dtype_str: str) -> tuple[str, int]:
    if "x" in dtype_str:
        base, lanes = dtype_str.rsplit("x", 1)
        return base, int(lanes)
    return dtype_str, 1


def _is_exact_int32_float_const(expr: tirx.PrimExpr) -> bool:
    if isinstance(expr, tirx.Broadcast):
        expr = expr.value
    if not isinstance(expr, tirx.FloatImm):
        return False
    val = float(expr.value)
    if not math.isfinite(val):
        return False
    if abs(val) > _MAX_EXACT_INT32_F32:
        return False
    return val == float(int(val))


def _peel_exact_int_clamp_round(expr: tirx.PrimExpr) -> Optional[tirx.PrimExpr]:
    """If ``expr`` is ``clamp(round(x))`` with exact-integer bounds, return ``clamp(x)``."""
    if isinstance(expr, tirx.Call) and _op_name(expr) in _ROUND_OPS and len(expr.args) == 1:
        arg = expr.args[0]
        base, _ = _base_dtype(str(arg.dtype))
        if base == "float32":
            return arg
        return None
    if isinstance(expr, tirx.Min):
        if _is_exact_int32_float_const(expr.b):
            inner = _peel_exact_int_clamp_round(expr.a)
            if inner is not None:
                return tirx.Min(inner, expr.b)
        if _is_exact_int32_float_const(expr.a):
            inner = _peel_exact_int_clamp_round(expr.b)
            if inner is not None:
                return tirx.Min(expr.a, inner)
        return None
    if isinstance(expr, tirx.Max):
        if _is_exact_int32_float_const(expr.b):
            inner = _peel_exact_int_clamp_round(expr.a)
            if inner is not None:
                return tirx.Max(inner, expr.b)
        if _is_exact_int32_float_const(expr.a):
            inner = _peel_exact_int_clamp_round(expr.b)
            if inner is not None:
                return tirx.Max(expr.a, inner)
        return None
    return None


def _fuse_round_to_int_cast_in_stmt(stmt: tirx.Stmt) -> tirx.Stmt:
    def postorder(node):
        if not isinstance(node, tirx.Cast):
            return None
        dst_base, dst_lanes = _base_dtype(str(node.dtype))
        if dst_base not in ("int8", "int16", "int32"):
            return None
        src_base, src_lanes = _base_dtype(str(node.value.dtype))
        if src_base != "float32" or src_lanes != dst_lanes:
            return None
        peeled = _peel_exact_int_clamp_round(node.value)
        if peeled is None:
            return None
        int32_dtype = f"int32x{dst_lanes}" if dst_lanes > 1 else "int32"
        rn = tirx.call_pure_extern(int32_dtype, "__float2int_rn", peeled)
        if str(node.dtype) == int32_dtype:
            return rn
        return tirx.Cast(node.dtype, rn)

    return tirx.stmt_functor.ir_transform(stmt, None, postorder, ["tirx.Cast"])


def _expr_uses_any_var(expr: tirx.PrimExpr, var_set: set) -> bool:
    found = False

    def visit(n):
        nonlocal found
        if not found and isinstance(n, tirx.Var) and n in var_set:
            found = True

    tirx.stmt_functor.post_order_visit(expr, visit)
    return found


def _const_int(expr: tirx.PrimExpr) -> Optional[int]:
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    return None


def _thread_var_stride_in_expr(expr: tirx.PrimExpr, tvar: tirx.Var) -> Optional[int]:
    """Return:
    - ``0`` if ``tvar`` does not appear in ``expr`` at all (fully uniform -> guard ``tvar == 0``),
    - ``K > 1`` if every occurrence of ``tvar`` in ``expr`` is ``FloorDiv(tvar, c)`` or
      ``shift_right(tvar, s)`` and the gcd of all such divisors is ``K > 1`` (guard ``tvar % K == 0``),
    - ``1`` if ``tvar`` is used directly / with stride 1 (no guard).
    """
    strides: list[int] = []
    bare = False

    def walk(node):
        nonlocal bare
        if bare:
            return
        if isinstance(node, tirx.Var):
            if node.same_as(tvar):
                bare = True
            return
        if isinstance(node, tirx.FloorDiv):
            c = _const_int(node.b)
            if isinstance(node.a, tirx.Var) and node.a.same_as(tvar) and c is not None and c > 1:
                strides.append(c)
                return
            walk(node.a)
            walk(node.b)
            return
        if isinstance(node, tirx.Call) and _op_name(node) == "tirx.shift_right" and len(node.args) == 2:
            s = _const_int(node.args[1])
            if isinstance(node.args[0], tirx.Var) and node.args[0].same_as(tvar) and s is not None and s > 0:
                strides.append(1 << s)
                return
            for a in node.args:
                walk(a)
            return
        # Recurse into child expressions
        for attr in ("a", "b", "value", "condition", "true_value", "false_value", "base", "stride"):
            child = getattr(node, attr, None)
            if isinstance(child, tirx.PrimExpr):
                walk(child)
        args = getattr(node, "args", None)
        if args is not None:
            for a in args:
                if isinstance(a, tirx.PrimExpr):
                    walk(a)
        indices = getattr(node, "indices", None)
        if indices is not None:
            for idx in indices:
                if isinstance(idx, tirx.PrimExpr):
                    walk(idx)

    walk(expr)
    if bare:
        return 1
    if not strides:
        return 0
    g = strides[0]
    for s in strides[1:]:
        g = math.gcd(g, s)
    return g if g > 1 else 1


def _guard_uniform_stores_stmt(
    stmt: tirx.Stmt,
    thread_vars: dict[str, tuple[tirx.Var, int]],
    predicated_on_thread: bool,
    thread_dep_vars: Optional[set] = None,
) -> tirx.Stmt:
    if thread_dep_vars is None:
        thread_dep_vars = set()

    if isinstance(stmt, tirx.AttrStmt):
        new_tvars = thread_vars
        new_dep = thread_dep_vars
        if stmt.attr_key == "thread_extent" and isinstance(stmt.node, tirx.IterVar):
            tag = str(stmt.node.thread_tag)
            ext = _const_int(stmt.value)
            if tag in _THREAD_TAGS and ext is not None and ext > 1:
                new_tvars = dict(thread_vars)
                new_tvars[tag] = (stmt.node.var, ext)
                new_dep = set(thread_dep_vars)
                new_dep.add(stmt.node.var)
        new_body = _guard_uniform_stores_stmt(stmt.body, new_tvars, predicated_on_thread, new_dep)
        if new_body.same_as(stmt.body):
            return stmt
        return tirx.AttrStmt(stmt.node, stmt.attr_key, stmt.value, new_body)

    if isinstance(stmt, tirx.SeqStmt):
        cur_dep = set(thread_dep_vars)
        new_seq = []
        changed = False
        for s in stmt.seq:
            if _BIND_CLS is not None and isinstance(s, _BIND_CLS):
                if _expr_uses_any_var(s.value, cur_dep):
                    cur_dep.add(s.var)
                new_seq.append(s)
                continue
            ns = _guard_uniform_stores_stmt(s, thread_vars, predicated_on_thread, cur_dep)
            if not ns.same_as(s):
                changed = True
            new_seq.append(ns)
        return tirx.SeqStmt(new_seq) if changed else stmt

    if isinstance(stmt, tirx.IfThenElse):
        cond_uses_t = predicated_on_thread or _expr_uses_any_var(stmt.condition, thread_dep_vars)
        new_then = _guard_uniform_stores_stmt(stmt.then_case, thread_vars, cond_uses_t, thread_dep_vars)
        new_else = (
            _guard_uniform_stores_stmt(stmt.else_case, thread_vars, cond_uses_t, thread_dep_vars)
            if stmt.else_case is not None
            else None
        )
        if new_then.same_as(stmt.then_case) and (
            stmt.else_case is None or new_else.same_as(stmt.else_case)
        ):
            return stmt
        return tirx.IfThenElse(stmt.condition, new_then, new_else)

    if isinstance(stmt, tirx.For):
        loop_uses_t = (
            predicated_on_thread
            or _expr_uses_any_var(stmt.min, thread_dep_vars)
            or _expr_uses_any_var(stmt.extent, thread_dep_vars)
        )
        new_body = _guard_uniform_stores_stmt(stmt.body, thread_vars, loop_uses_t, thread_dep_vars)
        if new_body.same_as(stmt.body):
            return stmt
        return tirx.For(
            stmt.loop_var,
            stmt.min,
            stmt.extent,
            stmt.kind,
            new_body,
            stmt.thread_binding,
            stmt.annotations,
        )

    if _ALLOCATE_CLS is not None and isinstance(stmt, _ALLOCATE_CLS):
        new_body = _guard_uniform_stores_stmt(stmt.body, thread_vars, predicated_on_thread, thread_dep_vars)
        if new_body.same_as(stmt.body):
            return stmt
        return _ALLOCATE_CLS(
            stmt.buffer_var, stmt.dtype, stmt.extents, stmt.condition, new_body, stmt.annotations
        )

    if _LET_STMT_CLS is not None and isinstance(stmt, _LET_STMT_CLS):
        new_dep = set(thread_dep_vars)
        if _expr_uses_any_var(stmt.value, new_dep):
            new_dep.add(stmt.var)
        new_body = _guard_uniform_stores_stmt(stmt.body, thread_vars, predicated_on_thread, new_dep)
        if new_body.same_as(stmt.body):
            return stmt
        return _LET_STMT_CLS(stmt.var, stmt.value, new_body)

    if isinstance(stmt, tirx.BufferStore):
        if predicated_on_thread or not thread_vars:
            return stmt
        scope = str(stmt.buffer.scope())
        if scope not in ("", "global"):
            return stmt
        if len(stmt.indices) != 1:
            return stmt
        idx = stmt.indices[0]
        _, val_lanes = _base_dtype(str(stmt.value.dtype))
        _, idx_lanes = _base_dtype(str(idx.dtype))
        if val_lanes != 1 or idx_lanes != 1:
            return stmt
        # If index uses any non-raw thread-dependent bound variable, leave untouched.
        raw_tvars = {v for v, _ in thread_vars.values()}
        derived_dep = thread_dep_vars - raw_tvars
        if derived_dep and _expr_uses_any_var(idx, derived_dep):
            return stmt

        conds = []
        for tag in _THREAD_TAGS:
            if tag not in thread_vars:
                continue
            tvar, ext = thread_vars[tag]
            stride = _thread_var_stride_in_expr(idx, tvar)
            if stride == 0:
                conds.append(tirx.EQ(tvar, tirx.IntImm(tvar.dtype, 0)))
            elif stride is not None and stride > 1 and ext % stride == 0:
                conds.append(tirx.EQ(tirx.FloorMod(tvar, tirx.IntImm(tvar.dtype, stride)), tirx.IntImm(tvar.dtype, 0)))
        if not conds:
            return stmt
        cond = conds[0]
        for c in conds[1:]:
            cond = tirx.And(cond, c)
        return tirx.IfThenElse(cond, stmt, None)

    return stmt


def _optimize_func(func: tirx.PrimFunc) -> tirx.PrimFunc:
    if not isinstance(func, tirx.PrimFunc):
        return func
    body = _fuse_round_to_int_cast_in_stmt(func.body)
    body = _guard_uniform_stores_stmt(body, {}, False)
    if body.same_as(func.body):
        return func
    return tirx.PrimFunc(
        func.params,
        body,
        func.ret_type,
        func.buffer_map,
        func.attrs,
        func.span,
    )


def optimize_musa_membound(mod: tvm.IRModule, config=None) -> tvm.IRModule:
    """Apply MUSA memory-bound TIR optimizations (round-to-int fusion & uniform store guard)."""
    config = config or {}
    if config.get("tl.musa_disable_membound_opt", False):
        return mod
    functions = {gv: _optimize_func(func) for gv, func in mod.functions.items()}
    return tvm.IRModule(functions, attrs=mod.attrs)


__all__ = ["optimize_musa_membound"]
