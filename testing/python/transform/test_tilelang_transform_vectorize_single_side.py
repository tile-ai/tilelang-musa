"""Tests for VectorizeSingleSide pass.

The pass should only form Ramp/Shuffle vector IR for strict scalar copy groups.
LowerLDGSTG is responsible for turning those Ramp-based global accesses into
ldg/stg intrinsics.
"""

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey
from tvm import tirx as tir


def _apply_vectorize_single_side(
    mod,
    target="musa",
    enable_lower_ldgstg=True,
    disable_vectorize_single_side=False,
    inject_assumes=False,
    run_vectorize_loop=True,
):
    if inject_assumes:
        mod = tl.transform.InjectAssumes()(mod)
    mod = tvm.tirx.transform.BindTarget(tvm.target.Target(target))(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    if run_vectorize_loop:
        mod = tl.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(
        config={
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG: enable_lower_ldgstg,
            PassConfigKey.TL_DISABLE_VECTORIZE_SINGLE_SIDE: disable_vectorize_single_side,
        }
    ):
        mod = tl.transform.VectorizeSingleSide()(mod)
    return mod


def _apply_vectorize_single_side_and_ldgstg(mod, target="musa"):
    mod = tvm.tirx.transform.BindTarget(tvm.target.Target(target))(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(config={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True}):
        mod = tl.transform.VectorizeSingleSide()(mod)
        mod = tl.transform.LowerLDGSTG()(mod)
    return mod


def _check_has_intrinsic(mod, intrinsic_name):
    found = [False]

    def visitor(obj):
        if isinstance(obj, tir.Call) and hasattr(obj.op, "name") and intrinsic_name in obj.op.name:
            found[0] = True

    tir.stmt_functor.post_order_visit(mod["main"].body, visitor)
    return found[0]


def _check_has_ramp_store_from_shuffle(mod):
    found = [False]

    def visitor(obj):
        if isinstance(obj, tir.BufferStore) and isinstance(obj.indices[0], tir.Ramp) and isinstance(obj.value, tir.Shuffle):
            found[0] = True

    tir.stmt_functor.post_order_visit(mod["main"].body, visitor)
    return found[0]


def _check_has_ramp_store(mod):
    found = [False]

    def visitor(obj):
        if isinstance(obj, tir.BufferStore) and isinstance(obj.indices[0], tir.Ramp):
            found[0] = True

    tir.stmt_functor.post_order_visit(mod["main"].body, visitor)
    return found[0]


def test_vectorize_single_side_forms_ramp_without_ldgstg():
    """VectorizeSingleSide should form vector IR without emitting intrinsics."""

    @T.prim_func
    def func(B: T.Buffer((32,), "float8_e4m3fn")):
        S = T.alloc_buffer((96,), "float8_e4m3fn", scope="shared")
        for tx in T.thread_binding(8, "threadIdx.x"):
            for i in T.unroll(4):
                B[tx * 4 + i] = S[tx * 5 + i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_vectorize_single_side(mod)

    assert _check_has_ramp_store_from_shuffle(mod)
    assert not _check_has_intrinsic(mod, "ldg")
    assert not _check_has_intrinsic(mod, "stg")


def test_vectorize_single_side_shared_to_global_lowers_to_stg():
    """Shared-to-global scalar groups should form a global Ramp store."""

    @T.prim_func
    def func(B: T.Buffer((32,), "float8_e4m3fn")):
        S = T.alloc_buffer((96,), "float8_e4m3fn", scope="shared")
        for tx in T.thread_binding(8, "threadIdx.x"):
            for i in T.unroll(4):
                B[tx * 4 + i] = S[tx * 5 + i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_vectorize_single_side_and_ldgstg(mod)

    assert _check_has_intrinsic(mod, "stg32")
    assert not _check_has_intrinsic(mod, "ldg32")


def test_vectorize_single_side_can_be_disabled():
    """The dedicated disable option should skip this pass only."""

    @T.prim_func
    def func(B: T.Buffer((32,), "float8_e4m3fn")):
        S = T.alloc_buffer((96,), "float8_e4m3fn", scope="shared")
        for tx in T.thread_binding(8, "threadIdx.x"):
            for i in T.unroll(4):
                B[tx * 4 + i] = S[tx * 5 + i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_vectorize_single_side(
        mod,
        disable_vectorize_single_side=True,
    )

    assert not _check_has_ramp_store_from_shuffle(mod)


def test_vectorize_single_side_shared_to_global_uses_wide_stg():
    """The pass should use the widest LowerLDGSTG-compatible vector width."""

    @T.prim_func
    def func(B: T.Buffer((256,), "float8_e4m3fn")):
        S = T.alloc_buffer((512,), "float8_e4m3fn", scope="shared")
        for tx in T.thread_binding(8, "threadIdx.x"):
            for i in T.unroll(32):
                B[tx * 32 + i] = S[tx * 33 + i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_vectorize_single_side_and_ldgstg(mod)

    assert _check_has_intrinsic(mod, "stg256")
    assert not _check_has_intrinsic(mod, "ldg")


def test_vectorize_single_side_global_to_shared_lowers_to_ldg():
    """Global-to-shared scalar groups should form a global Ramp load."""

    @T.prim_func
    def func(A: T.Buffer((32,), "float8_e4m3fn")):
        S = T.alloc_buffer((96,), "float8_e4m3fn", scope="shared")
        for tx in T.thread_binding(8, "threadIdx.x"):
            for i in T.unroll(4):
                S[tx * 5 + i] = A[tx * 4 + i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_vectorize_single_side_and_ldgstg(mod)

    assert _check_has_intrinsic(mod, "ldg32")
    assert not _check_has_intrinsic(mod, "stg32")


def test_vectorize_single_side_skips_global_to_global():
    """Global-to-global copies are left to the earlier VectorizeLoop pass."""

    @T.prim_func
    def func(
        A: T.Buffer((64,), "float8_e4m3fn"),
        B: T.Buffer((64,), "float8_e4m3fn"),
    ):
        for tx in T.thread_binding(8, "threadIdx.x"):
            for i in T.unroll(8):
                B[tx * 8 + i] = A[tx * 8 + i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_vectorize_single_side(mod, run_vectorize_loop=False)

    assert not _check_has_ramp_store(mod)


def test_vectorize_single_side_rejects_unaligned_global_store():
    """A contiguous but unaligned global store should stay scalar."""

    @T.prim_func
    def func(B: T.Buffer((65,), "float8_e4m3fn")):
        S = T.alloc_buffer((96,), "float8_e4m3fn", scope="shared")
        for tx in T.thread_binding(8, "threadIdx.x"):
            for i in T.unroll(4):
                B[tx * 4 + i + 1] = S[tx * 5 + i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_vectorize_single_side(mod)

    assert not _check_has_ramp_store_from_shuffle(mod)


def test_vectorize_single_side_rejects_unaligned_global_load():
    """A contiguous but unaligned global load should not lower to ldg."""

    @T.prim_func
    def func(A: T.Buffer((65,), "float8_e4m3fn")):
        S = T.alloc_buffer((96,), "float8_e4m3fn", scope="shared")
        for tx in T.thread_binding(8, "threadIdx.x"):
            for i in T.unroll(4):
                S[tx * 5 + i] = A[tx * 4 + i + 1]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_vectorize_single_side_and_ldgstg(mod)

    assert not _check_has_intrinsic(mod, "ldg32")
    assert not _check_has_intrinsic(mod, "stg32")


def test_vectorize_single_side_uses_assume_for_global_alignment():
    """Alignment can come from tl.assume on a dynamic stride."""

    @T.prim_func
    def func(B: T.Buffer((1024,), "float8_e4m3fn"), stride: T.int32):
        T.assume(stride % 4 == 0)
        S = T.alloc_buffer((128,), "float8_e4m3fn", scope="shared")
        for tx in T.thread_binding(8, "threadIdx.x"):
            for i in T.unroll(8):
                B[tx * stride + i] = S[tx * 13 + i // 4 * 8 + i % 4]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = _apply_vectorize_single_side(mod, inject_assumes=True)

    assert _check_has_ramp_store_from_shuffle(mod)


if __name__ == "__main__":
    tilelang.testing.main()
