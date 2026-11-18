"""Regression guards for splitting shared updates from strided local reads."""

import pytest

import tilelang
from tilelang import tvm
from tvm import tirx


def _update(kind="contiguous"):
    i = tirx.Var("i", "int32")
    shared = tirx.decl_buffer(
        (128,),
        "float16",
        name="shared",
        scope="shared",
        strides=[2] if kind == "buffer_stride" else None,
        elem_offset=1 if kind == "buffer_offset" else (8 if kind == "aligned_offset" else 0),
        data_alignment=2 if kind == "buffer_alignment" else 64,
    )
    fragment = tirx.decl_buffer((64,), "float32", name="fragment", scope="local")
    index = i // 8 * 16 + i % 8
    local_index = i * 2
    if kind == "permuted":
        index = i % 2 * 16 + i // 2
        local_index = i
    elif kind == "four_lanes":
        index = i // 4 * 16 + i % 4
    elif kind == "broadcast_store":
        index = tirx.const(0)
    elif kind == "repeated_store":
        index = i % 8
    elif kind == "strided_store":
        index = i * 2
    elif kind == "unaligned":
        index += 1
    elif kind == "contiguous_local":
        local_index = i
    elif kind == "overlapping_splits":
        index = i // 8 * 4 + i % 8
    elif kind == "indirect":
        index = tirx.Cast("int32", fragment[i])

    load_buffer = shared
    load_index = index
    if kind == "recurrence":
        load_index = index + 1
    elif kind == "alias":
        load_buffer = tirx.decl_buffer((128,), "float16", name="alias", data=shared.data, scope="shared")
        load_index = index + 1
    load = load_buffer[load_index]
    if kind == "predicated_load":
        load = tirx.BufferLoad(load_buffer, [load_index], predicate=i < 8)
    value = tirx.Cast("float32", load) - fragment[local_index]
    if kind == "call":
        value = tirx.call_extern("float32", "unknown", value)
    store = tirx.BufferStore(shared, tirx.Cast("float16", value), [index], predicate=i < 8 if kind == "predicated_store" else None)
    if kind == "conditional":
        body = tirx.IfThenElse(i < 7, store, None)
    elif kind == "multiple_stores":
        body = tirx.SeqStmt([store, tirx.BufferStore(fragment, value, [i])])
    else:
        body = store
    loop = tirx.For(
        i,
        1 if kind == "nonzero_min" else 0,
        15 if kind == "partial_extent" else 16,
        tirx.ForKind.UNROLLED,
        body,
        annotations={"unknown_annotation": 1} if kind == "annotated" else None,
        step=2 if kind == "nonunit_step" else None,
    )
    return tirx.PrimFunc([], tirx.SeqStmt([tirx.AllocBuffer(shared), tirx.AllocBuffer(fragment), loop]))


@pytest.mark.parametrize("kind,width", [("contiguous", 8), ("permuted", 8), ("four_lanes", 4), ("aligned_offset", 8)])
def test_shared_rmw_keeps_vector_copies_and_scalar_compute(kind, width):
    func = _update(kind).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    with tvm.target.Target({"kind": "musa", "arch": "mp_31"}):
        mod = tilelang.transform.DecoupleTypeCast()(mod)
        mod = tilelang.transform.LegalizeVectorizedLoop()(mod)
    loops = []
    tirx.stmt_functor.post_order_visit(mod["main"].body, lambda node: loops.append(node) if isinstance(node, tirx.For) else None)
    vector_loops = [loop for loop in loops if loop.kind == tirx.ForKind.VECTORIZED]
    assert len(vector_loops) == 2
    assert all(int(loop.extent) == width for loop in vector_loops)
    assert any(loop.kind == tirx.ForKind.SERIAL and int(loop.extent) == width for loop in loops)


@pytest.mark.parametrize(
    "kind",
    [
        "broadcast_store",
        "repeated_store",
        "strided_store",
        "unaligned",
        "contiguous_local",
        "recurrence",
        "alias",
        "call",
        "conditional",
        "multiple_stores",
        "overlapping_splits",
        "indirect",
        "predicated_load",
        "predicated_store",
        "nonzero_min",
        "partial_extent",
        "nonunit_step",
        "annotated",
        "buffer_stride",
        "buffer_offset",
        "buffer_alignment",
    ],
)
def test_shared_rmw_preserves_unsupported_or_dependent_loops(kind):
    func = _update(kind).with_attr("global_symbol", "main")
    mod = tvm.IRModule.from_expr(func)
    transformed = tilelang.transform.DecoupleTypeCast()(mod)
    tvm.ir.assert_structural_equal(mod, transformed)
