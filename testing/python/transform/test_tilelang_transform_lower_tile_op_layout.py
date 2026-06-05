from tilelang import tvm
from tilelang.layout import Layout
from tilelang.tileop.gemm.gemm_sqmma import GemmSQMMA


_expand_input_shape = tvm.get_global_func("tl.transform._TestingExpandLayoutToBufferInputShape")
_expand_output_shape = tvm.get_global_func("tl.transform._TestingExpandLayoutToBufferOutputShape")
_make_buffer_strides = tvm.get_global_func("tl.transform._TestingMakeBufferWithLayoutStrides")
_make_access_ptr_offset = tvm.get_global_func("tl.op._TestingMakeAccessPtrFromRegionOffset")


def _shared_buffer(shape, strides=None):
    return tvm.tir.decl_buffer(shape, "float16", strides=strides, scope="shared")


def _shape_values(shape):
    return [int(dim) for dim in shape]


def _mapped_values(layout, indices):
    analyzer = tvm.arith.Analyzer()
    return [int(analyzer.simplify(value)) for value in layout.map_forward_index(indices)]


def test_ph1_sqmma_transposed_shared_operand_layout_uses_logical_mn_view():
    layout = GemmSQMMA._make_transposed_shared_operand_layout(
        [32, 256],
        logical_rows=256,
        logical_cols=32,
        dtype="float8_e4m3fn",
        k_major=True,
    )

    assert _shape_values(layout.get_input_shape()) == [32, 256]
    assert _shape_values(layout.get_output_shape()) == [256, 32]
    assert _mapped_values(layout, [0, 128]) == [128, 0]
    assert _mapped_values(layout, [16, 0]) == [0, 16]


def test_expand_layout_allows_trailing_subdomain_shape():
    layout = Layout([32, 64], lambda i, j: [i, j])
    buffer = _shared_buffer((3, 32, 32))

    expanded_input = _expand_input_shape(buffer, layout)
    expanded_output = _expand_output_shape(buffer, layout)

    assert _shape_values(expanded_input) == [3, 32, 32]
    assert _shape_values(expanded_output) == [3, 32, 32]


def test_expand_layout_rejects_output_extent_larger_than_buffer():
    layout = Layout([32, 64], lambda i, j: [i, j + 64])
    buffer = _shared_buffer((3, 32, 32))

    expanded_input = _expand_input_shape(buffer, layout)
    expanded_output = _expand_output_shape(buffer, layout)

    assert _shape_values(expanded_input) == [32, 64]
    assert _shape_values(expanded_output) == [32, 128]


def test_expand_layout_rejects_product_only_trailing_mismatch():
    layout = Layout([32, 64], lambda i, j: [i, j])

    expanded = _expand_input_shape(_shared_buffer((3, 64, 16)), layout)

    assert _shape_values(expanded) == [32, 64]


def test_same_rank_lift_allows_extent_compatible_output():
    layout = Layout([3, 32, 32], lambda stage, i, j: [i, j])

    expanded = _expand_output_shape(_shared_buffer((3, 32, 32)), layout)

    assert _shape_values(expanded) == [3, 32, 32]


def test_same_rank_lift_rejects_product_only_mismatch():
    layout = Layout([3, 32, 64], lambda stage, i, j: [i, j])

    expanded = _expand_output_shape(_shared_buffer((3, 64, 16)), layout)

    assert _shape_values(expanded) == [32, 64]


def test_same_rank_lift_rejects_output_extent_larger_than_buffer():
    layout = Layout([3, 32, 64], lambda stage, i, j: [i, j + 64])

    expanded = _expand_output_shape(_shared_buffer((3, 32, 32)), layout)

    assert _shape_values(expanded) == [32, 128]


def test_stride_preserving_remap_keeps_pipeline_stage_gap():
    layout = Layout([2, 16, 16], lambda stage, i, j: [stage, i, j])
    buffer = _shared_buffer((2, 16, 16), strides=[2048, 16, 1])

    strides = _make_buffer_strides(buffer, layout)

    assert _shape_values(strides) == [2048, 16, 1]


def test_stride_preserving_remap_keeps_stage_gap_after_rank_collapse():
    layout = Layout([1, 32, 32], lambda _, i, j: [i, j])
    buffer = _shared_buffer((2, 1, 32, 32), strides=[2048, 2048, 32, 1])

    strides = _make_buffer_strides(buffer, layout)

    assert _shape_values(strides) == [2048, 32, 1]


def test_access_ptr_from_region_uses_buffer_strides():
    buffer = _shared_buffer((2, 16, 16), strides=[2048, 16, 1])
    region = tvm.tir.BufferRegion(
        buffer,
        [
            tvm.ir.Range.from_min_extent(1, 1),
            tvm.ir.Range.from_min_extent(0, 16),
            tvm.ir.Range.from_min_extent(0, 16),
        ],
    )

    offset = tvm.arith.Analyzer().simplify(_make_access_ptr_offset(region))

    assert int(offset) == 2048
