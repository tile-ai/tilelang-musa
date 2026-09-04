import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

_M = 32
_N = 32


def _torch_dtype(tilelang_dtype):
    return {
        T.float32: torch.float32,
        T.float16: torch.float16,
        T.bfloat16: torch.bfloat16,
        T.float8_e4m3fn: torch.float8_e4m3fn,
        T.int32: torch.int32,
    }[tilelang_dtype]


def _make_copy_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((_M, _N), dtype),
        dst: T.Tensor((_M, _N), dtype),
    ):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((_M, _N), dtype)
            T.copy(src, smem)
            T.copy(smem, dst)

    return kernel


def _copy_prefer_async_kernel(dtype):
    @T.prim_func
    def kernel(
        src: T.Tensor((_M,), dtype),
        dst: T.Tensor((_M,), dtype),
    ):
        with T.Kernel(1, threads=128):
            smem = T.alloc_shared((_M,), dtype)
            T.copy(src, smem, prefer_instruction="cp_async")
            T.copy(smem, dst)

    return kernel


@T.prim_func
def _pipeline_copy_kernel(
    src: T.Tensor((4, _M), T.float32),
    dst: T.Tensor((4, _M), T.float32),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((_M,), T.float32)
        for k in T.Pipelined(4, num_stages=2):
            T.copy(src[k, :], smem)
            T.copy(smem, dst[k, :])


@T.prim_func
def _prefer_async_unsupported_kernel(
    src: T.Tensor((128,), T.float16),
    dst: T.Tensor((128,), T.float16),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float16)
        T.copy(src, smem, prefer_instruction="cp_async")
        T.copy(smem, dst, disable_tma=True)


@T.prim_func
def _pipeline_async_fallback_kernel(
    src: T.Tensor((2, 128), T.float16),
    dst: T.Tensor((2, 128), T.float16),
):
    with T.Kernel(1, threads=128):
        smem = T.alloc_shared((128,), T.float16)
        for k in T.Pipelined(2, num_stages=2):
            T.copy(src[k, :], smem, disable_tma=True)
            T.copy(smem, dst[k, :], disable_tma=True)


@pytest.mark.parametrize("dtype", [T.float32, T.float16, T.bfloat16, T.float8_e4m3fn, T.int32])
@tilelang.testing.requires_musa
def test_musa_copy_codegen(dtype):
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _make_copy_kernel(dtype),
            target=target,
            enable_device_compile=False,
        )

    assert artifact.kernel_source is not None
    assert "__global__" in artifact.kernel_source


@pytest.mark.parametrize("dtype", [T.float32, T.float16, T.bfloat16, T.float8_e4m3fn, T.int32])
@tilelang.testing.requires_musa
def test_musa_copy_runtime_values(dtype):
    kernel = tilelang.compile(
        _make_copy_kernel(dtype),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )

    src_cpu = torch.arange(_M * _N, dtype=torch.float32).reshape(_M, _N)
    if dtype is T.float8_e4m3fn:
        src_cpu = (src_cpu % 8).to(torch.float8_e4m3fn)
    else:
        src_cpu = src_cpu.to(_torch_dtype(dtype))
    dst = kernel(src_cpu.to("musa"))
    torch.musa.synchronize()

    torch.testing.assert_close(dst.cpu(), src_cpu, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_copy_prefer_async_codegen_and_runtime():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _copy_prefer_async_kernel(T.float32),
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert "tl::cp_async_gs<4>" in source
    assert "tl::cp_async_commit()" in source
    assert "tl::cp_async_wait<0>()" in source

    kernel = tilelang.compile(
        _copy_prefer_async_kernel(T.float32),
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(_M, dtype=torch.float32)
    dst = kernel(src.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(dst.cpu(), src, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_pipeline_copy_selects_async_codegen():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _pipeline_copy_kernel,
            target=target,
            enable_device_compile=False,
        )

    source = artifact.kernel_source
    assert "tl::cp_async_gs<4>" in source
    assert "tl::cp_async_commit()" in source
    assert "tl::cp_async_wait<0>()" in source

    kernel = tilelang.compile(
        _pipeline_copy_kernel,
        out_idx=[1],
        target={"kind": "musa"},
        execution_backend="tvm_ffi",
    )
    src = torch.arange(4 * _M, dtype=torch.float32).reshape(4, _M)
    dst = kernel(src.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(dst.cpu(), src, atol=0, rtol=0)


@tilelang.testing.requires_musa
def test_musa_copy_prefer_async_rejects_unsupported_copy():
    target = tvm.target.Target({"kind": "musa"})
    with (
        target,
        pytest.raises(
            tvm.TVMError,
            match='T.copy\\(prefer_instruction="cp_async"\\).*no SIMT fallback',
        ),
    ):
        tilelang.lower(
            _prefer_async_unsupported_kernel,
            target=target,
            enable_device_compile=False,
        )


@tilelang.testing.requires_musa
def test_musa_pipeline_async_falls_back_to_simt():
    target = tvm.target.Target({"kind": "musa"})
    with target:
        artifact = tilelang.lower(
            _pipeline_async_fallback_kernel,
            target=target,
            enable_device_compile=False,
        )
    assert "cp_async_gs" not in artifact.kernel_source

    kernel = tilelang.compile(_pipeline_async_fallback_kernel, out_idx=[1])
    src = torch.randn((2, 128), dtype=torch.float16)
    dst = kernel(src.to("musa"))
    torch.musa.synchronize()
    torch.testing.assert_close(dst.cpu(), src, atol=0, rtol=0)
