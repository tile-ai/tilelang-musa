"""Regression tests for packed integer vector constructors (issue #9)."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@pytest.mark.parametrize(
    ("tl_dtype", "torch_dtype", "lo", "hi", "vec_extent"),
    [
        ("int8", torch.int8, -128, 128, 1024),   # 8 lanes/thread -> make_int2(8 x signed char)
        ("int8", torch.int8, -128, 128, 2048),   # 16 lanes/thread -> make_int4(16 x signed char)
        ("uint8", torch.uint8, 0, 256, 1024),    # 8 lanes/thread -> make_uint2(8 x unsigned char)
        ("uint8", torch.uint8, 0, 256, 2048),    # 16 lanes/thread -> make_uint4(16 x unsigned char)
        ("int16", torch.int16, -1000, 1000, 1024),  # 8 lanes/thread -> make_int4(8 x short)
    ],
)
def test_musa_masked_packed_vector_load(tl_dtype, torch_dtype, lo, hi, vec_extent):
    @tilelang.jit(out_idx=[1], target="musa")
    def build():
        @T.prim_func
        def main(x: T.Tensor((4, 128), tl_dtype), out: T.Tensor((128,), tl_dtype)):
            with T.Kernel(1, threads=128):
                acc = T.alloc_fragment((vec_extent,), "float32")
                T.fill(acc, -T.infinity("float32"))
                for step in T.serial(4):
                    for j in T.Parallel(vec_extent):
                        val = T.if_then_else(
                            j < 128,
                            T.cast(x[step, j], "float32"),
                            -T.infinity("float32"),
                        )
                        acc[j] = T.max(acc[j], val)
                for j in T.Parallel(vec_extent):
                    if j < 128:
                        out[j] = T.cast(acc[j], tl_dtype)

        return main

    kernel = build()
    x = torch.randint(lo, hi, (4, 128), dtype=torch_dtype, device="musa")
    out = kernel(x)
    torch.musa.synchronize()
    ref = x.max(dim=0).values
    assert torch.equal(out, ref)


if __name__ == "__main__":
    tilelang.testing.main()
