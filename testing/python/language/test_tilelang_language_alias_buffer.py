import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


def test_alias_buffer_preserves_storage_identity_and_scope():
    src = tvm.tirx.decl_buffer((64, 128), T.bfloat16, scope="shared.dyn")
    alias = T.alias_buffer(
        src,
        shape=(8, 8, 128),
        strides=(1024, 128, 1),
    )

    assert alias.data.same_as(src.data)
    assert alias.scope() == src.scope() == "shared.dyn"
    assert alias.dtype == src.dtype
    assert tuple(int(stride) for stride in alias.strides) == (1024, 128, 1)
    assert int(alias.elem_offset) == int(src.elem_offset)


def test_alias_buffer_supports_overlapping_strides_and_relative_offset():
    src = tvm.tirx.decl_buffer((512, 512), T.bfloat16)
    alias = T.alias_buffer(
        src,
        shape=(4, 512 - 56, 8, 128),
        strides=(128, 512, 4096, 1),
        elem_offset=128,
    )

    assert alias.data.same_as(src.data)
    assert tuple(int(dim) for dim in alias.shape) == (4, 456, 8, 128)
    assert tuple(int(stride) for stride in alias.strides) == (128, 512, 4096, 1)
    assert int(alias.elem_offset) == int(src.elem_offset) + 128


def test_alias_buffer_builds_compact_strides_by_default():
    src = tvm.tirx.decl_buffer((64, 128), T.bfloat16, scope="shared.dyn")
    alias = T.alias_buffer(src, shape=(8, 8, 128))

    assert tuple(int(stride) for stride in alias.strides) == (1024, 128, 1)


def test_alias_buffer_rejects_rank_mismatch():
    src = tvm.tirx.decl_buffer((64, 128), T.bfloat16)
    with pytest.raises(ValueError, match="same rank"):
        T.alias_buffer(src, shape=(8, 8, 128), strides=(128, 1))


@tilelang.jit(target="musa", out_idx=[-1])
def global_alias_kernel():
    @T.prim_func
    def kernel(
        src: T.Tensor((64,), T.float32),
        dst: T.Tensor((4, 8), T.float32),
    ):
        with T.Kernel(1, threads=32):
            alias = T.alias_buffer(
                src,
                shape=(4, 8),
                strides=(4, 1),
                elem_offset=3,
            )
            for i, j in T.Parallel(4, 8):
                dst[i, j] = alias[i, j]

    return kernel


@tilelang.testing.requires_musa
def test_alias_buffer_global_data_consistency():
    src_cpu = torch.arange(64, dtype=torch.float32)
    actual = global_alias_kernel()(src_cpu.to("musa")).cpu()

    row = torch.arange(4)[:, None]
    col = torch.arange(8)[None, :]
    expected = src_cpu[3 + row * 4 + col]
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


@tilelang.jit(target="musa", out_idx=[-1])
def shared_alias_kernel():
    @T.prim_func
    def kernel(
        src: T.Tensor((8, 8, 16), T.float32),
        dst: T.Tensor((64, 16), T.float32),
    ):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((64, 16), T.float32)
            alias = T.alias_buffer(
                smem,
                shape=(8, 8, 16),
                strides=(128, 16, 1),
            )

            for inner, group, col in T.Parallel(8, 8, 16):
                alias[inner, group, col] = src[inner, group, col]
            T.sync_threads()
            for row, col in T.Parallel(64, 16):
                dst[row, col] = smem[row, col]

    return kernel


@tilelang.testing.requires_musa
def test_alias_buffer_shared_data_consistency():
    src_cpu = torch.arange(8 * 8 * 16, dtype=torch.float32).reshape(8, 8, 16)
    actual = shared_alias_kernel()(src_cpu.to("musa")).cpu()
    expected = src_cpu.reshape(64, 16)
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
