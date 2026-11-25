"""MP22-specific MUSA shared-memory layout helpers."""

from __future__ import annotations

from collections.abc import Sequence

from tilelang import _ffi_api
from tvm import tirx

from .fragment import Fragment
from .swizzle import make_gemm_fragment_8x8


_MP22_MMA_SHAPES = ((8, 32, 16), (16, 16, 16), (32, 32, 16))
_QY2_SHARED_LAYOUT_POLICY_METADATA = "tl.internal.musa.qy2_shared_layout_policy"


def _mark_layout_policy(layout, policy: str):
    return _ffi_api.Layout_with_metadata(
        layout,
        _QY2_SHARED_LAYOUT_POLICY_METADATA,
        tirx.StringImm(policy),
    )


def _get_qy2_shared_layout_policy(layout) -> str | None:
    policy = _ffi_api.Layout_get_metadata(
        layout,
        _QY2_SHARED_LAYOUT_POLICY_METADATA,
    )
    return None if policy is None else str(policy.value)


def _normalize_mp22_operand(operand: str) -> str:
    operand = operand.upper()
    if operand not in ("A", "B"):
        raise ValueError(f"operand must be 'A' or 'B', but got {operand!r}")
    return operand


def _normalize_mp22_instruction_shape(
    instruction_shape: Sequence[int],
) -> tuple[int, int, int]:
    instruction_shape = tuple(int(value) for value in instruction_shape)
    if instruction_shape not in _MP22_MMA_SHAPES:
        raise ValueError(f"unsupported MP22 MMA instruction: {instruction_shape}")
    return instruction_shape


def _base_fragment_4x16() -> Fragment:
    return Fragment((4, 16), lambda i, j: (j // 2 + 8 * i, j % 2))


def _mp22_operand_atom_fragment(instruction_shape: tuple[int, int, int], operand: str, transposed: bool) -> Fragment:
    inst_m, inst_n, inst_k = instruction_shape
    if inst_m in (8, 16):
        base = make_gemm_fragment_8x8()
        if operand == "A":
            repeats = (inst_k // 8, inst_m // 8) if transposed else (inst_m // 8, inst_k // 8)
        else:
            repeats = (inst_n // 8, inst_k // 8) if transposed else (inst_k // 8, inst_n // 8)
        return base.repeat(list(repeats), False, True)

    row = make_gemm_fragment_8x8().repeat([2, 2], True, False).repeat([2, 1], False, False)
    col = _base_fragment_4x16().repeat([4, 1], True, False).repeat([1, 2], False, False)
    if operand == "A":
        return col if transposed else row
    return row if transposed else col


def _physical_shape(operand: str, x_extent: int, k_extent: int, transposed: bool) -> tuple[int, int]:
    if operand == "A":
        return (k_extent, x_extent) if transposed else (x_extent, k_extent)
    return (x_extent, k_extent) if transposed else (k_extent, x_extent)


def _physical_to_logical(operand: str, transposed: bool, row, col):
    if operand == "A":
        return (col, row) if transposed else (row, col)
    return (row, col) if transposed else (col, row)


def _logical_to_atom_physical(operand: str, transposed: bool, x, k):
    if operand == "A":
        return (k, x) if transposed else (x, k)
    return (x, k) if transposed else (k, x)


def _normalize_atom_order(atom_order: str) -> str:
    atom_order = atom_order.lower().replace("-", "_")
    if atom_order not in ("x_major", "k_major"):
        raise ValueError(f"atom_order must be 'x_major' or 'k_major', but got {atom_order!r}")
    return atom_order


def _make_mp22_fragment_inverse_shared_layout(
    instruction_shape: Sequence[int],
    operand: str,
    transposed: bool,
    x_extent: int,
    k_extent: int,
    atom_order: str = "x_major",
):
    """Derive a dense shared layout inverse to an MP22 operand fragment."""
    import tilelang.language as T

    instruction_shape = _normalize_mp22_instruction_shape(instruction_shape)
    operand = _normalize_mp22_operand(operand)
    atom_order = _normalize_atom_order(atom_order)
    inst_m, inst_n, inst_k = instruction_shape
    x_atom = inst_m if operand == "A" else inst_n
    if x_extent % x_atom != 0 or k_extent % inst_k != 0:
        raise ValueError(f"{operand} extent {x_extent}x{k_extent} is not tiled by {x_atom}x{inst_k}")

    fragment = _mp22_operand_atom_fragment(instruction_shape, operand, transposed)
    physical_shape = _physical_shape(operand, x_extent, k_extent, transposed)
    physical_cols = physical_shape[1]
    threads = 32 if inst_m in (8, 16) else 128
    atom_elements = x_atom * inst_k
    values_per_thread = atom_elements // threads
    num_x_tiles = x_extent // x_atom
    num_k_tiles = k_extent // inst_k

    def index_map(row, col):
        x, k = _physical_to_logical(operand, transposed, row, col)
        local_x = x % x_atom
        local_k = k % inst_k
        local_row, local_col = _logical_to_atom_physical(operand, transposed, local_x, local_k)
        tid = fragment.map_forward_thread([local_row, local_col])[0]
        vid = fragment.map_forward_index([local_row, local_col])[0]
        local_physical = tid * values_per_thread + vid
        tile_x = x // x_atom
        tile_k = k // inst_k
        tile_id = tile_x + num_x_tiles * tile_k if atom_order == "x_major" else tile_k + num_k_tiles * tile_x
        physical = tile_id * atom_elements + local_physical
        return physical // physical_cols, physical % physical_cols

    return T.Layout(list(physical_shape), index_map)


def make_mp22_fragment_inverse_shared_layout(
    instruction_shape: Sequence[int],
    operand: str,
    transposed: bool,
    x_extent: int,
    k_extent: int,
    atom_order: str = "x_major",
):
    """Derive and mark an inverse MP22 operand shared layout."""
    layout = _make_mp22_fragment_inverse_shared_layout(
        instruction_shape,
        operand,
        transposed,
        x_extent,
        k_extent,
        atom_order,
    )
    return _mark_layout_policy(layout, "fragment_inverse")


def _make_mp22_fragment_packed_shared_layout(
    instruction_shape: Sequence[int],
    operand: str,
    transposed: bool,
    x_extent: int,
    k_extent: int,
    pack_values: int = 8,
    atom_order: str = "x_major",
):
    """Derive a producer/consumer joint MP22 shared layout."""
    import tilelang.language as T

    if pack_values != 8:
        raise ValueError("fragment-packed MP22 layout currently requires 8-value packs")
    instruction_shape = _normalize_mp22_instruction_shape(instruction_shape)
    operand = _normalize_mp22_operand(operand)
    atom_order = _normalize_atom_order(atom_order)
    inst_m, inst_n, inst_k = instruction_shape
    x_atom = inst_m if operand == "A" else inst_n
    if x_extent % x_atom != 0 or k_extent % inst_k != 0:
        raise ValueError(f"{operand} extent {x_extent}x{k_extent} is not tiled by {x_atom}x{inst_k}")

    fragment = _mp22_operand_atom_fragment(instruction_shape, operand, transposed)
    physical_shape = _physical_shape(operand, x_extent, k_extent, transposed)
    physical_cols = physical_shape[1]
    threads = 32 if inst_m in (8, 16) else 128
    atom_elements = x_atom * inst_k
    values_per_thread = atom_elements // threads
    if values_per_thread % 2 != 0 or threads % (pack_values // 2) != 0:
        raise ValueError("fragment-packed layout requires complete fp16/bf16 pairs and lane packs")
    pair_count = values_per_thread // 2
    slots = pack_values // 2
    num_x_tiles = x_extent // x_atom
    num_k_tiles = k_extent // inst_k

    def index_map(row, col):
        x, k = _physical_to_logical(operand, transposed, row, col)
        local_x = x % x_atom
        local_k = k % inst_k
        local_row, local_col = _logical_to_atom_physical(operand, transposed, local_x, local_k)
        tid = fragment.map_forward_thread([local_row, local_col])[0]
        vid = fragment.map_forward_index([local_row, local_col])[0]
        group = tid // slots
        slot = tid % slots
        local_physical = group * (pack_values * pair_count) + (vid // 2) * pack_values + slot * 2 + vid % 2
        tile_x = x // x_atom
        tile_k = k // inst_k
        tile_id = tile_x + num_x_tiles * tile_k if atom_order == "x_major" else tile_k + num_k_tiles * tile_x
        physical = tile_id * atom_elements + local_physical
        return physical // physical_cols, physical % physical_cols

    return T.Layout(list(physical_shape), index_map)


def make_mp22_fragment_packed_shared_layout(
    instruction_shape: Sequence[int],
    operand: str,
    transposed: bool,
    x_extent: int,
    k_extent: int,
    pack_values: int = 8,
    atom_order: str = "x_major",
):
    """Derive and mark a packed MP22 operand shared layout."""
    layout = _make_mp22_fragment_packed_shared_layout(
        instruction_shape,
        operand,
        transposed,
        x_extent,
        k_extent,
        pack_values,
        atom_order,
    )
    return _mark_layout_policy(layout, "fragment_packed")
