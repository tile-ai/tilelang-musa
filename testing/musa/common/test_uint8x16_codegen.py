import tilelang
import tilelang.language as T
import tilelang.testing


tilelang.disable_cache()


@tilelang.jit(target="musa", out_idx=[2])
def uint8x16_codegen():
    @T.prim_func
    def main(
        a: T.Tensor((256 * 16,), "uint8"),
        b: T.Tensor((256 * 16,), "uint8"),
        y: T.Tensor((256 * 16,), "uint8"),
    ):
        with T.Kernel(1, threads=256):
            for index in T.Parallel(256 * 16):
                y[index] = T.bitwise_and(a[index], b[index])

    return main


@tilelang.testing.requires_musa
def test_uint8x16_codegen():
    source = uint8x16_codegen().get_kernel_source()
    assert "uint4" in source, source
    assert "tl::robust_param_desc_t" not in source, source
