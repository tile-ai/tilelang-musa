import tilelang
import tilelang.testing
import tilelang.language as T


@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_issue_1719_layout_1():
    @tilelang.jit
    def _buggy_kernel():
        with T.Kernel(threads=128):
            tmp1 = T.alloc_shared([32, 32], T.float16)
            tmp2 = T.alloc_shared([32, 32], T.float16)
            tmp3 = T.alloc_fragment([32, 32], T.float32)
            tmp4 = T.alloc_fragment([32], T.float32)
            T.gemm(tmp1, tmp2, tmp3, transpose_B=True)
            T.reduce_max(tmp3, tmp4)
            for i in T.Parallel(32):
                tmp4[i] = 1

    kernel = _buggy_kernel.compile()
    print(kernel.get_kernel_source())


if __name__ == "__main__":
    tilelang.testing.main()
