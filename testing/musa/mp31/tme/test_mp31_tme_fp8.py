import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


FP8_DTYPES = [
    (torch.float8_e4m3fn, "float8_e4m3fn"),
    (torch.float8_e5m2, "float8_e5m2"),
]


def _make_tme_fp8_copy(dtype):
    @T.prim_func
    def kernel(A: T.Tensor((128,), dtype), C: T.Tensor((128,), dtype)):
        with T.Kernel(1, threads=128):
            shared = T.alloc_shared((128,), dtype)
            barrier = T.alloc_barrier(128)
            T.tma_copy(A[0], shared, barrier=barrier)
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)
            T.tma_copy(shared, C[0])
            T.tma_store_wait(0)

    return kernel


@pytest.mark.parametrize("torch_dtype, tilelang_dtype", FP8_DTYPES)
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_tme_fp8_load_store(torch_dtype, tilelang_dtype):
    kernel = tilelang.compile(
        _make_tme_fp8_copy(tilelang_dtype),
        out_idx=-1,
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    values = torch.randn((128,), dtype=torch.float16, device="musa").to(torch_dtype)
    result = kernel(values)
    torch.musa.synchronize()
    torch.testing.assert_close(
        result.cpu().to(torch.float32),
        values.cpu().to(torch.float32),
        rtol=0.0,
        atol=0.0,
    )
