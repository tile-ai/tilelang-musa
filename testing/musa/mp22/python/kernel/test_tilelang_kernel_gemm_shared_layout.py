import re

import pytest
import tilelang
import tilelang.language as T
import tilelang.musa.op.gemm.qy2_shared_layout as qy2_shared_layout
import tilelang.testing
import torch

from tilelang.layout import (
    make_gemm_ab_layout,
    make_mp22_fragment_inverse_shared_layout,
    make_mp22_fragment_packed_shared_layout,
)
from tilelang.layout.musa_mp22 import _get_qy2_shared_layout_policy


_LAYOUT_BUILDERS = {
    "fragment_inverse": make_mp22_fragment_inverse_shared_layout,
    "fragment_packed": make_mp22_fragment_packed_shared_layout,
}
_FLAGS = {
    "fragment_inverse": {
        "ss": (1, 1, 0, 0),
        "sr": (1, 0, 0, 0),
        "rs": (0, 1, 0, 0),
    },
    "fragment_packed": {
        "ss": (0, 0, 1, 1),
        "sr": (0, 0, 1, 0),
        "rs": (0, 0, 0, 1),
    },
}
_MMA_CASES = [
    pytest.param("m8n32k16", (8, 32, 16), id="m8n32k16"),
    pytest.param("m16n16k16", (16, 16, 16), id="m16n16k16"),
    pytest.param("m32n32k16", (32, 32, 16), id="m32n32k16"),
]


@pytest.fixture(autouse=True)
def disable_tilelang_cache(monkeypatch):
    cache_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    yield
    if cache_enabled:
        tilelang.enable_cache()


def _template_args(source: str, op: str) -> tuple[int, ...]:
    match = re.search(rf"tl::gemm_{op}<([^>]*)>", source)
    assert match is not None
    return tuple(int(arg.strip()) for arg in match.group(1).split(","))


def _all_template_args(source: str, op: str) -> list[tuple[int, ...]]:
    return [
        tuple(int(arg.strip()) for arg in match.split(","))
        for match in re.findall(rf"tl::gemm_{op}<([^>]*)>", source)
    ]


def _atom_strides(atom_elements: int, atom_order: str) -> tuple[int, int]:
    if atom_order == "x_major":
        return atom_elements, atom_elements * 2
    return atom_elements * 2, atom_elements


def _make_layout_matrix_gemm(
    layout_kind: str,
    op: str,
    inst: tuple[int, int, int],
    transpose_a: bool,
    transpose_b: bool,
    atom_order: str,
):
    inst_m, inst_n, inst_k = inst
    full_m = inst_m * 2
    full_n = inst_n * 2
    full_k = inst_k * 2
    a_shape = (full_k, full_m) if transpose_a else (full_m, full_k)
    b_shape = (full_n, full_k) if transpose_b else (full_k, full_n)
    a_view_shape = (full_k, inst_m) if transpose_a else (inst_m, full_k)
    b_view_shape = (inst_n, full_k) if transpose_b else (full_k, inst_n)

    make_layout = _LAYOUT_BUILDERS[layout_kind]
    a_layout = make_layout(
        inst,
        "A",
        transpose_a,
        full_m,
        full_k,
        atom_order=atom_order,
    )
    b_layout = make_layout(
        inst,
        "B",
        transpose_b,
        full_n,
        full_k,
        atom_order=atom_order,
    )
    threads = 128 if inst_m == 32 else 32

    if op == "ss":

        @T.prim_func
        def main(
            A: T.Tensor(a_shape, "float16"),
            B: T.Tensor(b_shape, "float16"),
            C: T.Tensor((inst_m, inst_n), "float16"),
        ):
            with T.Kernel(1, threads=threads):
                A_local = T.alloc_shared(a_shape, "float16")
                B_local = T.alloc_shared(b_shape, "float16")
                C_local = T.alloc_fragment((inst_m, inst_n), "float32")
                T.annotate_layout({A_local: a_layout, B_local: b_layout})
                T.copy(A, A_local)
                T.copy(B, B_local)
                T.clear(C_local)
                T.gemm(
                    A_local[0 : a_view_shape[0], 0 : a_view_shape[1]],
                    B_local[0 : b_view_shape[0], 0 : b_view_shape[1]],
                    C_local,
                    transpose_A=transpose_a,
                    transpose_B=transpose_b,
                )
                T.copy(C_local, C)

    elif op == "sr":

        @T.prim_func
        def main(
            A: T.Tensor(a_shape, "float16"),
            B: T.Tensor(b_shape, "float16"),
            C: T.Tensor((inst_m, inst_n), "float16"),
        ):
            with T.Kernel(1, threads=threads):
                A_local = T.alloc_shared(a_shape, "float16")
                B_local = T.alloc_fragment(b_view_shape, "float16")
                C_local = T.alloc_fragment((inst_m, inst_n), "float32")
                T.annotate_layout({A_local: a_layout})
                T.copy(A, A_local)
                T.copy(
                    B[0 : b_view_shape[0], 0 : b_view_shape[1]],
                    B_local,
                )
                T.clear(C_local)
                T.gemm(
                    A_local[0 : a_view_shape[0], 0 : a_view_shape[1]],
                    B_local,
                    C_local,
                    transpose_A=transpose_a,
                    transpose_B=transpose_b,
                )
                T.copy(C_local, C)

    elif op == "rs":

        @T.prim_func
        def main(
            A: T.Tensor(a_shape, "float16"),
            B: T.Tensor(b_shape, "float16"),
            C: T.Tensor((inst_m, inst_n), "float16"),
        ):
            with T.Kernel(1, threads=threads):
                A_local = T.alloc_fragment(a_view_shape, "float16")
                B_local = T.alloc_shared(b_shape, "float16")
                C_local = T.alloc_fragment((inst_m, inst_n), "float32")
                T.annotate_layout({B_local: b_layout})
                T.copy(
                    A[0 : a_view_shape[0], 0 : a_view_shape[1]],
                    A_local,
                )
                T.copy(B, B_local)
                T.clear(C_local)
                T.gemm(
                    A_local,
                    B_local[0 : b_view_shape[0], 0 : b_view_shape[1]],
                    C_local,
                    transpose_A=transpose_a,
                    transpose_B=transpose_b,
                )
                T.copy(C_local, C)

    else:
        raise ValueError(f"unsupported GEMM operand mode: {op}")

    return main, a_shape, b_shape, a_view_shape, b_view_shape


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("layout_kind", _LAYOUT_BUILDERS)
@pytest.mark.parametrize("atom_order", ["x_major", "k_major"])
@pytest.mark.parametrize("op", ["ss", "sr", "rs"])
@pytest.mark.parametrize("transpose_a", [False, True], ids=["an", "at"])
@pytest.mark.parametrize("transpose_b", [False, True], ids=["bn", "bt"])
@pytest.mark.parametrize("mma_shape,inst", _MMA_CASES)
def test_gemm_optimized_shared_layout_matrix(
    monkeypatch,
    layout_kind,
    atom_order,
    op,
    transpose_a,
    transpose_b,
    mma_shape,
    inst,
):
    monkeypatch.setenv("TILELANG_MUSA_MP22_MMA_SHAPE", mma_shape)
    program, a_shape, b_shape, a_view_shape, b_view_shape = _make_layout_matrix_gemm(
        layout_kind,
        op,
        inst,
        transpose_a,
        transpose_b,
        atom_order,
    )

    def reject_other_policy(*args, **kwargs):
        raise AssertionError("QY2 shared-layout detection must inspect only the annotated policy")

    other_policy_builder = (
        "_make_mp22_fragment_packed_shared_layout" if layout_kind == "fragment_inverse" else "_make_mp22_fragment_inverse_shared_layout"
    )
    monkeypatch.setattr(qy2_shared_layout, other_policy_builder, reject_other_policy)
    kernel = tilelang.compile(program, out_idx=[2])
    a = torch.randn(a_shape, device="musa", dtype=torch.float16)
    b = torch.randn(b_shape, device="musa", dtype=torch.float16)
    a_physical = a[: a_view_shape[0], : a_view_shape[1]].float()
    b_physical = b[: b_view_shape[0], : b_view_shape[1]].float()
    a_ref = a_physical.T if transpose_a else a_physical
    b_ref = b_physical.T if transpose_b else b_physical
    torch.testing.assert_close(
        kernel(a, b),
        (a_ref @ b_ref).half(),
        rtol=1e-2,
        atol=1e-2,
    )

    args = _template_args(kernel.get_kernel_source(), op)
    assert args[-10:-6] == _FLAGS[layout_kind][op]
    assert args[-6:-4] == (0, 0)
    inst_m, inst_n, inst_k = inst
    if op[0] == "s":
        expected_a_strides = _atom_strides(inst_m * inst_k, atom_order)
        assert args[-4:-2] == expected_a_strides
    if op[1] == "s":
        expected_b_strides = _atom_strides(inst_n * inst_k, atom_order)
        assert args[-2:] == expected_b_strides


def _make_static_subview_gemm(layout_kind: str, op: str):
    block_m = block_n = block_k = 16
    full_k = 32
    inst = (16, 16, 16)
    make_layout = _LAYOUT_BUILDERS[layout_kind]
    a_layout = make_layout(inst, "A", False, block_m, full_k)
    b_layout = make_layout(inst, "B", True, block_n, full_k)

    if op == "ss":

        @T.prim_func
        def main(
            A: T.Tensor((block_m, full_k), "float16"),
            B: T.Tensor((block_n, full_k), "float16"),
            C: T.Tensor((block_m, block_n), "float16"),
        ):
            with T.Kernel(1, threads=32):
                A_local = T.alloc_shared((block_m, full_k), "float16")
                B_local = T.alloc_shared((block_n, full_k), "float16")
                C_local = T.alloc_fragment((block_m, block_n), "float32")
                T.annotate_layout({A_local: a_layout, B_local: b_layout})
                T.copy(A, A_local)
                T.copy(B, B_local)
                T.clear(C_local)
                T.gemm(
                    A_local[:, 16:],
                    B_local[:, 16:],
                    C_local,
                    transpose_B=True,
                )
                T.copy(C_local, C)

    elif op == "sr":

        @T.prim_func
        def main(
            A: T.Tensor((block_m, full_k), "float16"),
            B: T.Tensor((block_n, full_k), "float16"),
            C: T.Tensor((block_m, block_n), "float16"),
        ):
            with T.Kernel(1, threads=32):
                A_local = T.alloc_shared((block_m, full_k), "float16")
                B_local = T.alloc_fragment((block_n, block_k), "float16")
                C_local = T.alloc_fragment((block_m, block_n), "float32")
                T.annotate_layout({A_local: a_layout})
                T.copy(A, A_local)
                T.copy(B[:, 16:], B_local)
                T.clear(C_local)
                T.gemm(
                    A_local[:, 16:],
                    B_local,
                    C_local,
                    transpose_B=True,
                )
                T.copy(C_local, C)

    elif op == "rs":

        @T.prim_func
        def main(
            A: T.Tensor((block_m, full_k), "float16"),
            B: T.Tensor((block_n, full_k), "float16"),
            C: T.Tensor((block_m, block_n), "float16"),
        ):
            with T.Kernel(1, threads=32):
                A_local = T.alloc_fragment((block_m, block_k), "float16")
                B_local = T.alloc_shared((block_n, full_k), "float16")
                C_local = T.alloc_fragment((block_m, block_n), "float32")
                T.annotate_layout({B_local: b_layout})
                T.copy(A[:, 16:], A_local)
                T.copy(B, B_local)
                T.clear(C_local)
                T.gemm(
                    A_local,
                    B_local[:, 16:],
                    C_local,
                    transpose_B=True,
                )
                T.copy(C_local, C)

    else:
        raise ValueError(f"unsupported GEMM operand mode: {op}")

    return main


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("layout_kind", _LAYOUT_BUILDERS)
@pytest.mark.parametrize("op", ["ss", "sr", "rs"])
def test_gemm_optimized_shared_layout_static_subview(layout_kind, op):
    kernel = tilelang.compile(
        _make_static_subview_gemm(layout_kind, op),
        out_idx=[2],
    )
    a = torch.randn((16, 32), device="musa", dtype=torch.float16)
    b = torch.randn((16, 32), device="musa", dtype=torch.float16)
    torch.testing.assert_close(
        kernel(a, b),
        (a[:, 16:].float() @ b[:, 16:].float().T).half(),
        rtol=1e-2,
        atol=1e-2,
    )

    args = _template_args(kernel.get_kernel_source(), op)
    assert args[-10:-6] == _FLAGS[layout_kind][op]
    expected_offsets = {
        "ss": (256, 256),
        "sr": (256, 0),
        "rs": (0, 256),
    }
    assert args[-6:-4] == expected_offsets[op]


def _make_default_shared_layout_gemm(in_dtype: str, accum_dtype: str, block_k: int):
    block_m = block_n = 32

    @T.prim_func
    def main(
        A: T.Tensor((block_m, block_k), in_dtype),
        B: T.Tensor((block_n, block_k), in_dtype),
        C: T.Tensor((block_m, block_n), accum_dtype),
    ):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_shared((block_m, block_k), in_dtype)
            B_local = T.alloc_shared((block_n, block_k), in_dtype)
            C_local = T.alloc_fragment((block_m, block_n), accum_dtype)
            T.copy(A, A_local)
            T.copy(B, B_local)
            T.clear(C_local)
            T.gemm(A_local, B_local, C_local, transpose_B=True)
            T.copy(C_local, C)

    return main


def _make_explicit_default_shared_layout_gemm(layout_source: str):
    shape = [32, 16]
    a_layout = make_gemm_ab_layout(shape, 16, 16, True)
    b_layout = make_gemm_ab_layout(shape, 16, 16, True)
    if layout_source == "handwritten":
        a_default = a_layout
        b_default = b_layout
        a_layout = T.Layout(shape, lambda i, j: list(a_default.map_forward_index([i, j])))
        b_layout = T.Layout(shape, lambda i, j: list(b_default.map_forward_index([i, j])))

    @T.prim_func
    def main(
        A: T.Tensor((32, 16), "float16"),
        B: T.Tensor((32, 16), "float16"),
        C: T.Tensor((32, 32), "float32"),
    ):
        with T.Kernel(1, threads=128):
            A_local = T.alloc_shared((32, 16), "float16")
            B_local = T.alloc_shared((32, 16), "float16")
            C_local = T.alloc_fragment((32, 32), "float32")
            T.annotate_layout({A_local: a_layout, B_local: b_layout})
            T.copy(A, A_local)
            T.copy(B, B_local)
            T.clear(C_local)
            T.gemm(A_local, B_local, C_local, transpose_B=True)
            T.copy(C_local, C)

    return main


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize(
    "in_dtype,accum_dtype,block_k,torch_dtype,rtol,atol",
    [
        pytest.param("float16", "float32", 16, torch.float16, 1e-2, 1e-2, id="fp16"),
        pytest.param("float32", "float32", 8, torch.float32, 1.25e-1, 1.25e-1, id="tf32"),
        pytest.param("int8", "int32", 32, torch.int8, 0, 0, id="int8"),
    ],
)
def test_gemm_default_shared_layout_preserves_legacy_lowering(
    monkeypatch,
    in_dtype,
    accum_dtype,
    block_k,
    torch_dtype,
    rtol,
    atol,
):
    def reject_optimized_layout(*args, **kwargs):
        raise AssertionError("default GEMM must not construct an optimized shared layout")

    monkeypatch.setattr(
        qy2_shared_layout,
        "_make_mp22_fragment_inverse_shared_layout",
        reject_optimized_layout,
    )
    monkeypatch.setattr(
        qy2_shared_layout,
        "_make_mp22_fragment_packed_shared_layout",
        reject_optimized_layout,
    )
    kernel = tilelang.compile(
        _make_default_shared_layout_gemm(in_dtype, accum_dtype, block_k),
        out_idx=[2],
    )
    args = _template_args(kernel.get_kernel_source(), "ss")
    assert len(args) == 12

    if torch_dtype != torch.int8:
        a = torch.randn((32, block_k), device="musa", dtype=torch_dtype)
        b = torch.randn((32, block_k), device="musa", dtype=torch_dtype)
        expected = a.float() @ b.float().T
    else:
        a = torch.randint(-4, 5, (32, block_k), device="musa", dtype=torch_dtype)
        b = torch.randint(-4, 5, (32, block_k), device="musa", dtype=torch_dtype)
        expected = (a.cpu().int() @ b.cpu().int().T).to("musa")
    torch.testing.assert_close(kernel(a, b), expected, rtol=rtol, atol=atol)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("layout_source", ["default_factory", "handwritten"])
def test_gemm_non_policy_annotated_layout_preserves_legacy_lowering(monkeypatch, layout_source):
    def reject_optimized_layout(*args, **kwargs):
        raise AssertionError("non-policy layout must not construct an optimized shared layout")

    monkeypatch.setattr(
        qy2_shared_layout,
        "_make_mp22_fragment_inverse_shared_layout",
        reject_optimized_layout,
    )
    monkeypatch.setattr(
        qy2_shared_layout,
        "_make_mp22_fragment_packed_shared_layout",
        reject_optimized_layout,
    )
    kernel = tilelang.compile(
        _make_explicit_default_shared_layout_gemm(layout_source),
        out_idx=[2],
    )
    assert len(_template_args(kernel.get_kernel_source(), "ss")) == 12

    a = torch.randn((32, 16), device="musa", dtype=torch.float16)
    b = torch.randn((32, 16), device="musa", dtype=torch.float16)
    torch.testing.assert_close(kernel(a, b), a.float() @ b.float().T, rtol=1e-2, atol=1e-2)


def _make_unaligned_subview_gemm():
    layout = make_mp22_fragment_inverse_shared_layout((16, 16, 16), "A", False, 16, 32)

    @T.prim_func
    def main(
        A: T.Tensor((16, 32), "float16"),
        B: T.Tensor((16, 16), "float16"),
        C: T.Tensor((16, 16), "float16"),
    ):
        with T.Kernel(1, threads=32):
            A_local = T.alloc_shared((16, 32), "float16")
            B_local = T.alloc_fragment((16, 16), "float16")
            C_local = T.alloc_fragment((16, 16), "float32")
            T.annotate_layout({A_local: layout})
            T.copy(A, A_local)
            T.copy(B, B_local)
            T.clear(C_local)
            T.gemm(A_local[:, 8:24], B_local, C_local, transpose_B=True)
            T.copy(C_local, C)

    return main


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_optimized_shared_layout_rejects_unaligned_subview():
    with pytest.raises(Exception, match="must be aligned to the MMA K atom"):
        tilelang.compile(_make_unaligned_subview_gemm(), out_idx=[2])


def test_mp22_shared_layout_policy_survives_expand():
    cases = [
        (make_mp22_fragment_inverse_shared_layout, "fragment_inverse"),
        (make_mp22_fragment_packed_shared_layout, "fragment_packed"),
    ]
    for make_layout, expected_policy in cases:
        layout = make_layout((16, 16, 16), "A", False, 16, 32)
        expanded = layout.expand([2])
        for candidate in (layout, expanded):
            assert _get_qy2_shared_layout_policy(candidate) == expected_policy
            assert _get_qy2_shared_layout_policy(candidate) == expected_policy


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_mp22_shared_layout_policy_survives_repeated_lowering():
    program = _make_static_subview_gemm("fragment_inverse", "sr")
    for _ in range(2):
        kernel = tilelang.compile(program, out_idx=[2])
        args = _template_args(kernel.get_kernel_source(), "sr")
        assert args[-10:-6] == _FLAGS["fragment_inverse"]["sr"]


def _make_reannotated_shared_layout_gemm():
    inverse_layout = make_mp22_fragment_inverse_shared_layout(
        (16, 16, 16),
        "A",
        False,
        16,
        16,
    )
    packed_layout = make_mp22_fragment_packed_shared_layout(
        (16, 16, 16),
        "A",
        False,
        16,
        16,
    )

    @T.prim_func
    def main(
        A0: T.Tensor((16, 16), "float16"),
        A1: T.Tensor((16, 16), "float16"),
        B: T.Tensor((16, 16), "float16"),
        C: T.Tensor((16, 16), "float16"),
    ):
        with T.Kernel(1, threads=32):
            A_local = T.alloc_shared((16, 16), "float16")
            B_local = T.alloc_fragment((16, 16), "float16")
            C_local = T.alloc_fragment((16, 16), "float32")
            T.copy(B, B_local)
            T.clear(C_local)

            T.annotate_layout(
                {A_local: inverse_layout},
                allow_reannotation=True,
            )
            T.copy(A0, A_local)
            T.gemm(A_local, B_local, C_local, transpose_B=True)

            T.annotate_layout(
                {A_local: packed_layout},
                allow_reannotation=True,
            )
            T.copy(A1, A_local)
            T.gemm(A_local, B_local, C_local, transpose_B=True)
            T.copy(C_local, C)

    return main


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_shared_layout_policy_reannotation_is_per_operation():
    kernel = tilelang.compile(
        _make_reannotated_shared_layout_gemm(),
        out_idx=[3],
    )
    args = _all_template_args(kernel.get_kernel_source(), "sr")
    assert len(args) == 2
    assert args[0][-10:-6] == _FLAGS["fragment_inverse"]["sr"]
    assert args[1][-10:-6] == _FLAGS["fragment_packed"]["sr"]

    a0 = torch.randn((16, 16), device="musa", dtype=torch.float16)
    a1 = torch.randn((16, 16), device="musa", dtype=torch.float16)
    b = torch.randn((16, 16), device="musa", dtype=torch.float16)
    expected = (a0.float() @ b.float().T + a1.float() @ b.float().T).half()
    torch.testing.assert_close(
        kernel(a0, a1, b),
        expected,
        rtol=1e-2,
        atol=1e-2,
    )


if __name__ == "__main__":
    tilelang.testing.main()
