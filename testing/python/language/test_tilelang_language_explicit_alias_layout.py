import pytest
import tilelang
import tilelang.language as T
from tilelang import tvm

def _lower_without_device_compile(func):
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target, tvm.transform.PassContext(opt_level=3):
        return tilelang.lower(func, target=target, enable_device_compile=False)

def test_strided_alias_layout_rejects_out_of_bounds_addresses():
    @T.prim_func
    def kernel(
        input: T.Tensor((2, 8), T.float16),
        output: T.Tensor((2, 8), T.float16),
    ):
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((4, 8), T.float16)
            alias = T.alias_buffer(backing, (2, 8), (32, 1))
            T.annotate_layout({backing: T.Layout(backing.shape, lambda i, j: [i, j])})
            T.copy(input, alias)
            T.copy(alias, output)

    with pytest.raises(ValueError, match="Cannot prove that every address of alias"):
        _lower_without_device_compile(kernel)


def test_strided_alias_layout_accepts_in_bounds_relative_offset():
    @T.prim_func
    def kernel(
        input: T.Tensor((2, 8), T.float16),
        output: T.Tensor((2, 8), T.float16),
    ):
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((4, 8), T.float16)
            alias = T.alias_buffer(backing, (2, 8), (8, 1), elem_offset=8)
            T.annotate_layout({backing: T.Layout(backing.shape, lambda i, j: [i, j])})
            T.copy(input, alias)
            T.copy(alias, output)

    assert _lower_without_device_compile(kernel).kernel_source


def test_strided_alias_layout_rejects_non_compact_source():
    @T.prim_func
    def kernel(
        input: T.Tensor((2, 2, 8), T.float16),
        output: T.Tensor((4, 8), T.float16),
    ):
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((4, 8), T.float16)
            strided_source = T.alias_buffer(backing, (2, 2, 8), (1, 16, 2))
            T.annotate_layout({strided_source: T.Layout(strided_source.shape, lambda i, j, k: [i, j, k])})
            T.copy(input, strided_source)
            T.copy(backing, output)

    with pytest.raises(ValueError, match="Cannot propagate layout from non-compact alias"):
        _lower_without_device_compile(kernel)


def test_strided_alias_layout_rejects_ambiguous_same_shape_anchor():
    @T.prim_func
    def kernel(
        input: T.Tensor((4, 8), T.float16),
        output: T.Tensor((4, 8), T.float16),
    ):
        with T.Kernel(1, threads=32):
            backing = T.alloc_shared((4, 8), T.float16)
            transposed_addressing = T.alias_buffer(backing, (4, 8), (1, 4))
            T.annotate_layout({backing: T.Layout(backing.shape, lambda i, j: [i, j])})
            T.copy(input, transposed_addressing)
            T.copy(transposed_addressing, output)

    with pytest.raises(ValueError, match="Ambiguous layout annotation"):
        _lower_without_device_compile(kernel)
