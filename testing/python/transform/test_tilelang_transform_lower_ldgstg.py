"""Tests for LowerLDGSTG pass that converts Ramp-based global memory
load/store to ldg/stg intrinsics.

Pass configurations:
- tl.enable_lower_ldgstg: Enable non-predicated ldg/stg lowering (default: OFF)
- tl.enable_lower_ldgstg_predicated: Enable predicated ldg/stg lowering (default: OFF)
"""

from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang.transform import PassConfigKey
from tvm import tir

SUPPORTED_LDGSTG_TARGETS = ("musa",)


def _apply_passes(mod, enable_non_predicated=False, enable_predicated=False, target="musa"):
    """Apply the LowerLDGSTG pass and related lowering passes."""
    mod = tvm.tir.transform.BindTarget(tvm.target.Target(target))(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(
        config={
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG: enable_non_predicated,
            PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: enable_predicated,
        }
    ):
        mod = tl.transform.LowerLDGSTG()(mod)
    return mod


def _check_has_intrinsic(mod, intrinsic_name):
    """Check if the module contains a specific intrinsic call."""
    found = [False]

    def visitor(obj):
        if isinstance(obj, tir.Call) and hasattr(obj.op, "name") and intrinsic_name in obj.op.name:
            found[0] = True

    tir.stmt_functor.post_order_visit(mod["main"].body, visitor)
    return found[0]


def _assert_intrinsics_on_targets(
    mod,
    *,
    expected=(),
    unexpected=(),
    enable_non_predicated=False,
    enable_predicated=False,
    targets=SUPPORTED_LDGSTG_TARGETS,
    label="LowerLDGSTG",
):
    """Apply lowering for each target and assert expected intrinsics."""
    for target in targets:
        lowered = _apply_passes(
            mod,
            enable_non_predicated=enable_non_predicated,
            enable_predicated=enable_predicated,
            target=target,
        )
        print(f"=== {label} [{target}] ===")
        print(lowered)
        for intrinsic in expected:
            assert _check_has_intrinsic(lowered, intrinsic), f"Expected {intrinsic} when lowering for {target}"
        for intrinsic in unexpected:
            assert not _check_has_intrinsic(lowered, intrinsic), f"Did not expect {intrinsic} when lowering for {target}"


def test_lower_ldg32_default_off():
    """Test that non-predicated ldg/stg lowering is OFF by default."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(128, "threadIdx.x"):
            B[i] = A[i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        unexpected=("ldg32", "stg32"),
        label="test_lower_ldg32_default_off",
    )


def test_lower_ldg32_enabled():
    """Test that ldg32/stg32 works when enabled."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(128, "threadIdx.x"):
            B[i] = A[i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("ldg32", "stg32"),
        enable_non_predicated=True,
        label="test_lower_ldg32_enabled",
    )


def test_lower_ldg64_enabled():
    """Test that ldg64/stg64 works when enabled."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(64, "threadIdx.x"):
            for j in T.vectorized(2):
                B[i * 2 + j] = A[i * 2 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("ldg64", "stg64"),
        enable_non_predicated=True,
        label="test_lower_ldg64_enabled",
    )


def test_lower_ldg128_enabled():
    """Test that ldg128/stg128 works when enabled."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                B[i * 4 + j] = A[i * 4 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("ldg128", "stg128"),
        enable_non_predicated=True,
        label="test_lower_ldg128_enabled",
    )


def test_lower_ldg256_enabled():
    """Test that ldg256/stg256 works when enabled."""

    @T.prim_func
    def func(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32")):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(8):
                B[i * 8 + j] = A[i * 8 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("ldg256", "stg256"),
        enable_non_predicated=True,
        label="test_lower_ldg256_enabled",
    )


def test_lower_ldg32_predicated():
    """Test predicated ldg32 for single element load."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(128, "threadIdx.x"):
            # Predicate doesn't depend on loop var, so it can be lowered
            B[i] = T.if_then_else(pred > 0, A[i], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("ldg32",),
        enable_predicated=True,
        label="test_lower_ldg32_predicated",
    )


def test_lower_stg32_predicated():
    """Test predicated stg32 for single element store."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(128, "threadIdx.x"):
            # Predicate doesn't depend on loop var, so it can be lowered
            with T.If(pred > 0), T.Then():
                B[i] = A[i]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("stg32",),
        enable_predicated=True,
        label="test_lower_stg32_predicated",
    )


def test_lower_ldg64_predicated():
    """Test predicated ldg64 for vectorized load."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(64, "threadIdx.x"):
            for j in T.vectorized(2):
                B[i * 2 + j] = T.if_then_else(pred > 0, A[i * 2 + j], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("ldg64",),
        enable_predicated=True,
        label="test_lower_ldg64_predicated",
    )


def test_lower_stg64_predicated():
    """Test predicated stg64 for vectorized store."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(64, "threadIdx.x"):
            for j in T.vectorized(2):
                with T.If(pred > 0), T.Then():
                    B[i * 2 + j] = A[i * 2 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("stg64",),
        enable_predicated=True,
        label="test_lower_stg64_predicated",
    )


def test_lower_ldg128_predicated():
    """Test predicated ldg128 for vectorized load."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                # Predicate doesn't depend on vectorized loop var
                B[i * 4 + j] = T.if_then_else(pred > 0, A[i * 4 + j], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("ldg128",),
        enable_predicated=True,
        label="test_lower_ldg128_predicated",
    )


def test_lower_stg128_predicated():
    """Test predicated stg128 for vectorized store."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                # Predicate doesn't depend on vectorized loop var
                with T.If(pred > 0), T.Then():
                    B[i * 4 + j] = A[i * 4 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("stg128",),
        enable_predicated=True,
        label="test_lower_stg128_predicated",
    )


def test_lower_ldg256_predicated():
    """Test predicated ldg256 for vectorized load."""

    @T.prim_func
    def func(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(8):
                B[i * 8 + j] = T.if_then_else(pred > 0, A[i * 8 + j], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("ldg256",),
        enable_predicated=True,
        label="test_lower_ldg256_predicated",
    )


def test_lower_stg256_predicated():
    """Test predicated stg256 for vectorized store."""

    @T.prim_func
    def func(A: T.Buffer((256,), "float32"), B: T.Buffer((256,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(8):
                with T.If(pred > 0), T.Then():
                    B[i * 8 + j] = A[i * 8 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("stg256",),
        enable_predicated=True,
        label="test_lower_stg256_predicated",
    )


def test_predicated_store_with_load():
    """Test that when a predicated store contains a load, the load also gets predicated.

    This tests the pattern: if (pred) { B[i] = A[i] }
    Both the store and the load should use predicated versions to avoid
    out-of-bounds memory access when pred is false.
    """

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), pred: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                with T.If(pred > 0), T.Then():
                    B[i * 4 + j] = A[i * 4 + j]

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        expected=("ldg128", "stg128"),
        enable_predicated=True,
        label="test_predicated_store_with_load",
    )


def test_predicated_disabled():
    """Test that predicated lowering can be disabled."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32"), N: T.int32):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                idx = i * 4 + j
                B[idx] = T.if_then_else(idx < N, A[idx], T.float32(0))

    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    _assert_intrinsics_on_targets(
        mod,
        unexpected=("ldg", "stg"),
        enable_predicated=False,
        label="test_predicated_disabled",
    )


def test_non_supported_target_skip():
    """Test that the pass is skipped for unsupported targets."""

    @T.prim_func
    def func(A: T.Buffer((128,), "float32"), B: T.Buffer((128,), "float32")):
        for i in T.thread_binding(32, "threadIdx.x"):
            for j in T.vectorized(4):
                B[i * 4 + j] = A[i * 4 + j]

    # Use a CPU target
    cpu_target = tvm.target.Target("llvm")
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(cpu_target)(mod)
    mod = tl.transform.FlattenBuffer()(mod)
    mod = tl.transform.VectorizeLoop()(mod)
    with tvm.transform.PassContext(config={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True}):
        mod = tl.transform.LowerLDGSTG()(mod)
    print("=== test_non_musa_target_skip ===")
    print(mod)
    # The load should NOT be lowered to ldg because target is unsupported.
    assert not _check_has_intrinsic(mod, "ldg"), "Unsupported targets should NOT use ldg intrinsics"
    assert not _check_has_intrinsic(mod, "stg"), "Unsupported targets should NOT use stg intrinsics"


@tilelang.testing.requires_musa
def test_e2e_load_global_store_global():
    """End-to-end test that ldg/stg intrinsics work correctly when enabled."""
    import torch

    @tilelang.jit(pass_configs={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True})
    def copy_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 4, threads=32) as pid:
            for j in T.vectorized(4):
                Y[pid * 4 + j] = X[pid * 4 + j]

    X = torch.randn(128, dtype=torch.float32, device="musa")
    Y = torch.empty(128, dtype=torch.float32, device="musa")

    copy_kernel(X, Y)

    # Verify correctness
    torch.testing.assert_close(Y, X, atol=1e-5, rtol=1e-5)

    # Verify codegen contains ldg/stg
    src = copy_kernel.get_kernel_source(N=128)
    print("=== Generated kernel source ===")
    print(src)
    assert "load_global_128" in src or "store_global_128" in src, "Expected load_global_128/store_global_128 in generated source"


@tilelang.testing.requires_musa
def test_e2e_load_global_store_global_predicated():
    """End-to-end test that load_global/store_global intrinsics work correctly when enabled."""
    import torch

    @tilelang.jit(pass_configs={PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True, PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True})
    def copy_kernel(X, Y):
        N = T.const("N")
        X: T.Tensor[[N], T.float32]
        Y: T.Tensor[[N], T.float32]

        with T.Kernel(N // 4, threads=32) as pid:
            for j in T.vectorized(4):
                Y[pid * 4 + j] = T.if_then_else(pid < N // 8, X[pid * 4 + j], T.float32(0))

    X = torch.randn(128, dtype=torch.float32, device="musa")
    Y = torch.empty(128, dtype=torch.float32, device="musa")

    copy_kernel(X, Y)

    # Verify correctness
    Y_ref = torch.zeros(128, dtype=torch.float32, device="musa")
    for i in range(128):
        if i < 64:
            Y_ref[i] = X[i]
        else:
            Y_ref[i] = 0

    torch.testing.assert_close(Y, Y_ref, atol=1e-5, rtol=1e-5)

    # Verify codegen contains load_global/store_global
    src = copy_kernel.get_kernel_source(N=128)
    print("=== Generated kernel source ===")
    print(src)
    assert "load_global_128_conditional" in src or "store_global_128_conditional" in src, (
        "Expected load_global_128_conditional/store_global_128_conditional in generated source"
    )


if __name__ == "__main__":
    tilelang.testing.main()
