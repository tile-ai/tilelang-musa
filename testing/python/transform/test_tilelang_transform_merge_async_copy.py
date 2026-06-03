"""Tests for TileLang `MergeAsyncCopy` transform pass."""

import tilelang as tl
import tilelang.language as T
from tilelang import tvm
from tvm import tir
from tvm.tir.stmt_functor import post_order_visit


def _collect_async_copy_calls(func: tvm.tir.PrimFunc):
    calls = []

    def _visit(node):
        if (
            isinstance(node, tvm.tir.Call)
            and isinstance(node.op, tvm.ir.Op)
            and str(node.op.name)
            in {
                "tir.ptx_cp_async",
                "tl.ptx_cp_async",
                "tl.musa_cp_async_robust",
            }
        ):
            calls.append(node)

    post_order_visit(func.body, _visit)
    return calls


def _count_for_loops(func: tvm.tir.PrimFunc) -> int:
    count = 0

    def _visit(node):
        nonlocal count
        if isinstance(node, tvm.tir.For):
            count += 1

    post_order_visit(func.body, _visit)
    return count


def test_merge_async_copy_folds_plain_unrolled_loop():
    @T.prim_func
    def before(A: T.Tensor((16,), T.float16), B: T.Tensor((16,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        for v in T.unroll(2):
            T.evaluate(
                T.ptx_cp_async(
                    T.access_ptr(S[v * 4], "w", 4),
                    T.access_ptr(A[v * 4], "r", 4),
                    4,
                )
            )
        B[0] = S[0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.MergeAsyncCopy()(mod)

    calls = _collect_async_copy_calls(mod["main"])
    assert len(calls) == 1
    assert int(calls[0].args[2]) == 8
    assert _count_for_loops(mod["main"]) == 0


def test_merge_async_copy_folds_robust_unrolled_loop():
    robust_op = tir.op.Op.get("tl.musa_cp_async_robust")

    @T.prim_func
    def before(A: T.Tensor((16,), T.float16), B: T.Tensor((16,), T.float16)):
        S = T.alloc_buffer((16,), dtype=T.float16, scope="shared")
        robust_desc = T.make_robust_desc(T.address_of(A[0]), 32)
        for v in T.unroll(2):
            T.attr(0, "tl.force_async_copy", 1)
            T.attr(A.data, "tl.source_robust_desc", robust_desc)
            T.evaluate(
                tir.call_intrin(
                    "",
                    robust_op,
                    T.access_ptr(S[v * 4], "w", 4),
                    T.access_ptr(A[v * 4], "r", 4),
                    8,
                    T.address_of(A[0]),
                    32,
                )
            )
        B[0] = S[0]

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tl.transform.MergeAsyncCopy()(mod)

    calls = _collect_async_copy_calls(mod["main"])
    assert len(calls) == 1
    assert str(calls[0].op.name) == "tl.musa_cp_async_robust"
    assert int(calls[0].args[2]) == 16
    assert _count_for_loops(mod["main"]) == 0
