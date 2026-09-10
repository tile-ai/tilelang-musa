import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


@T.prim_func
def _copy_on_explicit_stream(
    src: T.Tensor((128,), T.float32),
    dst: T.Tensor((128,), T.float32),
):
    with T.Kernel(1, threads=128):
        i = T.get_thread_binding()
        dst[i] = src[i] + 1.0


@tilelang.testing.requires_musa
def test_musa_tvm_ffi_explicit_raw_stream():
    kernel = tilelang.compile(
        _copy_on_explicit_stream,
        out_idx=[1],
        target="musa",
        execution_backend="tvm_ffi",
    )
    src = torch.arange(128, dtype=torch.float32, device="musa")
    stream = torch.musa.Stream()
    result = kernel(src, stream=int(stream.musa_stream))
    stream.synchronize()
    torch.testing.assert_close(result.cpu(), src.cpu() + 1.0)

    with pytest.raises(TypeError, match="raw stream handle"):
        kernel(src, stream=stream)
