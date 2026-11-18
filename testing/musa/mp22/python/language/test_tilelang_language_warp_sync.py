import tilelang
import tilelang.language as T
import tilelang.testing
import torch


@tilelang.jit
def kernel_with_warp_sync():
    @T.prim_func
    def main(
        A: T.Tensor((1,), "int32"),
        B: T.Tensor((1,), "int32"),
    ):
        with T.Kernel(1, threads=128):
            tx = T.get_thread_binding()
            if tx == 0:
                A[0] = -1
            T.sync_warp()
            if tx == 1:
                B[0] = A[0]

    return main


@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_warp_sync():
    a = torch.empty((1,), device="musa", dtype=torch.int32)
    b = torch.empty((1,), device="musa", dtype=torch.int32)
    kernel = kernel_with_warp_sync()
    assert "__syncwarp" in kernel.get_kernel_source()
    kernel(a, b)
    assert b[0] == -1


if __name__ == "__main__":
    tilelang.testing.main()
