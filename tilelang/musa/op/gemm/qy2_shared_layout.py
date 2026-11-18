"""Detection helpers for MP22 fragment-oriented shared layouts."""

from __future__ import annotations

from dataclasses import dataclass
from itertools import product

from tilelang import tvm
from tilelang.layout.musa_mp22 import (
    _get_qy2_shared_layout_policy,
    _make_mp22_fragment_inverse_shared_layout,
    _make_mp22_fragment_packed_shared_layout,
)
from tilelang.utils.language import is_shared
from tvm import tirx


_DEFAULT = "default"
_FRAGMENT_INVERSE = "fragment_inverse"
_FRAGMENT_PACKED = "fragment_packed"
_OPTIMIZED_SHARED_LAYOUT_DTYPES = ("float16", "bfloat16")
_OPTIMIZED_SHARED_LAYOUT_INSTRUCTION_SHAPES = frozenset(((8, 32, 16), (16, 16, 16), (32, 32, 16)))
_MAX_COMPARED_ELEMENTS = 1 << 20


@dataclass(frozen=True)
class QY2SharedLayoutMatch:
    kind: str = _DEFAULT
    physical_offset: int = 0
    atom_x_stride: int = 0
    atom_k_stride: int = 0

    @property
    def optimized(self) -> bool:
        return self.kind != _DEFAULT

    @property
    def inverse(self) -> bool:
        return self.kind == _FRAGMENT_INVERSE

    @property
    def packed(self) -> bool:
        return self.kind == _FRAGMENT_PACKED


def _const_int(expr) -> int | None:
    if isinstance(expr, int):
        return expr
    if isinstance(expr, tirx.IntImm):
        return int(expr.value)
    simplified = tvm.arith.Analyzer().simplify(expr)
    if isinstance(simplified, tirx.IntImm):
        return int(simplified.value)
    return None


def _lookup_layout(layout_map, buffer):
    if buffer in layout_map:
        return layout_map[buffer]
    for candidate, layout in layout_map.items():
        if candidate.same_as(buffer):
            return layout
    return None


def _as_expr_list(value):
    if isinstance(value, tirx.PrimExpr):
        return [value]
    return list(value)


def _are_static_layouts_equivalent(actual, expected) -> bool:
    if actual.is_equal(expected):
        return True

    actual_shape = [_const_int(value) for value in actual.get_input_shape()]
    expected_shape = [_const_int(value) for value in expected.get_input_shape()]
    if (
        len(actual_shape) != len(expected_shape)
        or actual_shape != expected_shape
        or any(extent is None or extent <= 0 for extent in actual_shape)
    ):
        return False

    num_elements = 1
    for extent in actual_shape:
        num_elements *= extent
        if num_elements > _MAX_COMPARED_ELEMENTS:
            return False

    analyzer = tvm.arith.Analyzer()
    ranges = [range(extent) for extent in actual_shape]
    for indices in product(*ranges):
        actual_indices = _as_expr_list(actual.map_forward_index(list(indices)))
        expected_indices = _as_expr_list(expected.map_forward_index(list(indices)))
        if len(actual_indices) != len(expected_indices):
            return False
        if any(not analyzer.can_prove_equal(lhs, rhs) for lhs, rhs in zip(actual_indices, expected_indices)):
            return False
    return True


def _flatten_static_layout_point(layout, indices) -> int | None:
    outputs = _as_expr_list(layout.map_forward_index(indices))
    output_shape = list(layout.get_output_shape())
    if len(outputs) != len(output_shape):
        return None

    analyzer = tvm.arith.Analyzer()
    linear = 0
    for output, extent_expr in zip(outputs, output_shape):
        coordinate = _const_int(analyzer.simplify(output))
        extent = _const_int(analyzer.simplify(extent_expr))
        if coordinate is None or extent is None or extent <= 0 or coordinate < 0 or coordinate >= extent:
            return None
        linear = linear * extent + coordinate
    return linear


def _physical_indices(ndim: int, operand: str, transposed: bool, x: int, k: int) -> list[int]:
    if operand == "A":
        row, col = (k, x) if transposed else (x, k)
    else:
        row, col = (x, k) if transposed else (k, x)
    return [0] * (ndim - 2) + [row, col]


def _get_static_subview_mapping(
    kind: str,
    actual,
    atom_expected,
    region,
    x_atom: int,
    k_atom: int,
    operand: str,
    transposed: bool,
    full_x: int,
    full_k: int,
) -> QY2SharedLayoutMatch | None:
    ndim = len(region.region)
    if ndim < 2 or len(actual.get_input_shape()) != ndim:
        return None

    row_min = _const_int(region.region[-2].min)
    col_min = _const_int(region.region[-1].min)
    row_extent = _const_int(region.region[-2].extent)
    col_extent = _const_int(region.region[-1].extent)
    if (
        row_min is None
        or col_min is None
        or row_extent is None
        or col_extent is None
        or row_min < 0
        or col_min < 0
        or row_extent <= 0
        or col_extent <= 0
    ):
        return None

    if operand == "A":
        x_offset, k_offset = (col_min, row_min) if transposed else (row_min, col_min)
    else:
        x_offset, k_offset = (row_min, col_min) if transposed else (col_min, row_min)
    if x_offset % x_atom != 0:
        raise ValueError(f"QY2 optimized shared-layout subview X offset {x_offset} must be aligned to the MMA X atom {x_atom}")
    if k_offset % k_atom != 0:
        raise ValueError(f"QY2 optimized shared-layout subview K offset {k_offset} must be aligned to the MMA K atom {k_atom}")

    def actual_at(x: int, k: int) -> int | None:
        return _flatten_static_layout_point(
            actual,
            _physical_indices(ndim, operand, transposed, x, k),
        )

    def atom_at(x: int, k: int) -> int | None:
        return _flatten_static_layout_point(
            atom_expected,
            _physical_indices(2, operand, transposed, x, k),
        )

    full_base = actual_at(0, 0)
    if full_base is None:
        return None
    atom_elements = x_atom * k_atom
    atom_x_stride = atom_elements
    atom_k_stride = atom_elements
    if full_x > x_atom:
        next_x = actual_at(x_atom, 0)
        if next_x is None:
            return None
        atom_x_stride = next_x - full_base
    if full_k > k_atom:
        next_k = actual_at(0, k_atom)
        if next_k is None:
            return None
        atom_k_stride = next_k - full_base
    if atom_x_stride <= 0 or atom_k_stride <= 0:
        return None

    physical_base = None
    for row in range(row_extent):
        for col in range(col_extent):
            if operand == "A":
                local_x, local_k = (col, row) if transposed else (row, col)
            else:
                local_x, local_k = (row, col) if transposed else (col, row)
            actual_linear = actual_at(x_offset + local_x, k_offset + local_k)
            atom_linear = atom_at(local_x % x_atom, local_k % k_atom)
            if actual_linear is None or atom_linear is None:
                return None
            expected_relative = (local_x // x_atom) * atom_x_stride + (local_k // k_atom) * atom_k_stride + atom_linear
            delta = actual_linear - expected_relative
            if physical_base is None:
                physical_base = delta
            elif physical_base != delta:
                return None

    if physical_base is None or physical_base < 0:
        return None
    if physical_base % 8 != 0:
        raise ValueError(f"QY2 optimized shared-layout physical subview base {physical_base} elements must be 16-byte aligned")
    return QY2SharedLayoutMatch(
        kind,
        physical_base,
        atom_x_stride,
        atom_k_stride,
    )


def detect_qy2_shared_layout(
    layout_map,
    region,
    operand: str,
    transposed: bool,
    instruction_shape: tuple[int, int, int],
) -> QY2SharedLayoutMatch:
    buffer = region.buffer
    actual = _lookup_layout(layout_map, buffer)
    if not is_shared(buffer) or actual is None:
        return QY2SharedLayoutMatch()
    policy = _get_qy2_shared_layout_policy(actual)
    if policy is None:
        return QY2SharedLayoutMatch()
    if policy not in (_FRAGMENT_INVERSE, _FRAGMENT_PACKED):
        raise ValueError(f"unsupported QY2 shared-layout policy: {policy}")
    if (
        str(buffer.dtype) not in _OPTIMIZED_SHARED_LAYOUT_DTYPES
        or tuple(instruction_shape) not in _OPTIMIZED_SHARED_LAYOUT_INSTRUCTION_SHAPES
    ):
        return QY2SharedLayoutMatch()

    buffer_shape = [_const_int(value) for value in buffer.shape]
    if len(buffer_shape) < 2 or len(region.region) != len(buffer_shape) or buffer_shape[-2] is None or buffer_shape[-1] is None:
        return QY2SharedLayoutMatch()
    physical_rows, physical_cols = buffer_shape[-2:]
    if physical_rows <= 0 or physical_cols <= 0:
        return QY2SharedLayoutMatch()

    if operand == "A":
        full_x, full_k = (physical_cols, physical_rows) if transposed else (physical_rows, physical_cols)
    else:
        full_x, full_k = (physical_rows, physical_cols) if transposed else (physical_cols, physical_rows)
    x_atom = instruction_shape[0] if operand == "A" else instruction_shape[1]
    k_atom = instruction_shape[2]
    if full_x % x_atom != 0:
        raise ValueError(f"QY2 optimized shared layout requires the full X extent {full_x} to be divisible by the MMA X atom {x_atom}")
    if full_k % k_atom != 0:
        raise ValueError(f"QY2 optimized shared layout requires the full K extent {full_k} to be divisible by the MMA K atom {k_atom}")

    leading_shape = list(buffer.shape[: len(buffer.shape) - 2])
    make_layout = _make_mp22_fragment_inverse_shared_layout if policy == _FRAGMENT_INVERSE else _make_mp22_fragment_packed_shared_layout
    for atom_order in ("x_major", "k_major"):
        full_candidate = make_layout(
            instruction_shape,
            operand,
            transposed,
            full_x,
            full_k,
            atom_order=atom_order,
        )
        candidates = [full_candidate]
        if leading_shape:
            candidates.append(full_candidate.expand(leading_shape))
        if not any(_are_static_layouts_equivalent(actual, candidate) for candidate in candidates):
            continue

        atom_candidate = make_layout(
            instruction_shape,
            operand,
            transposed,
            x_atom,
            k_atom,
        )
        match = _get_static_subview_mapping(
            policy,
            actual,
            atom_candidate,
            region,
            x_atom,
            k_atom,
            operand,
            transposed,
            full_x,
            full_k,
        )
        if match is None:
            raise ValueError(
                f"QY2 {policy.replace('_', '-')} shared layout does not "
                "support this static BufferRegion: the restricted full "
                "layout is not an atom-affine subview of the parent "
                "shared layout"
            )
        return match
    return QY2SharedLayoutMatch()


def validate_qy2_shared_layouts(
    gemm,
    instruction_shape: tuple[int, int, int],
    a_match: QY2SharedLayoutMatch,
    b_match: QY2SharedLayoutMatch,
) -> None:
    if not a_match.optimized and not b_match.optimized:
        return
    if str(gemm.A.dtype) not in _OPTIMIZED_SHARED_LAYOUT_DTYPES or str(gemm.B.dtype) not in _OPTIMIZED_SHARED_LAYOUT_DTYPES:
        raise ValueError("QY2 optimized shared layouts currently support only fp16/bf16 MMA operands")
    if str(gemm.C.dtype) != "float32":
        raise ValueError("QY2 optimized shared layouts require a float32 fragment accumulator")
    if instruction_shape not in _OPTIMIZED_SHARED_LAYOUT_INSTRUCTION_SHAPES:
        raise ValueError(f"QY2 optimized shared layouts do not support instruction shape {instruction_shape}")

    for match, region, rows, cols in (
        (
            a_match,
            gemm.ARegion,
            gemm.K if gemm.trans_A else gemm.M,
            gemm.M if gemm.trans_A else gemm.K,
        ),
        (
            b_match,
            gemm.BRegion,
            gemm.N if gemm.trans_B else gemm.K,
            gemm.K if gemm.trans_B else gemm.N,
        ),
    ):
        if not match.optimized:
            continue
        row_min = _const_int(region.region[-2].min)
        col_min = _const_int(region.region[-1].min)
        row_extent = _const_int(region.region[-2].extent)
        col_extent = _const_int(region.region[-1].extent)
        if None in (row_min, col_min, row_extent, col_extent):
            raise ValueError("QY2 optimized shared layouts require a static shared region")
        if row_extent != rows or col_extent != cols:
            raise ValueError(
                "QY2 optimized shared layout region does not match the GEMM "
                f"operand shape: got {row_extent}x{col_extent}, expected "
                f"{rows}x{cols}"
            )
