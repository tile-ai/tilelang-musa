import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import torch


@T.prim_func
def _peer_lsu128_copy(
    source: T.Tensor((128,), T.float32),
    destination: T.Tensor((128,), T.float32),
):
    with T.Kernel(32, threads=32) as block:
        offset = block * 4
        packed = T.ldg128_peer(source[offset : offset + 4])
        T.stg128_peer(destination[offset : offset + 4], packed)


@T.prim_func
def _peer_signal_kernel(signal: T.Tensor((1,), T.int32)):
    with T.Kernel(1, threads=32):
        thread = T.get_thread_binding()
        if thread == 0:
            T.peer_signal_store(signal[0], 17)
        T.peer_release_fence()


def _make_indexed_peer_signal_kernel(use_extern):
    @T.prim_func
    def kernel(signal_ptrs: T.Tensor((1,), T.ptr), index: T.int32, value: T.int32):
        with T.Kernel(1, threads=32):
            if T.get_thread_binding() == 0:
                if use_extern:
                    T.call_extern("handle", "tl::peer_signal_store", signal_ptrs[0], index, value)
                else:
                    signal = T.make_tensor(signal_ptrs[0], (16,), T.int32)
                    T.peer_signal_store(signal[index], value)

    return kernel


def _make_peer_warp_reduce_kernel(dtype):
    vector_width = 4 if dtype == "float32" else 8

    @T.prim_func
    def kernel(
        source: T.Tensor((64, vector_width), dtype),
        destination: T.Tensor((64, vector_width), dtype),
    ):
        with T.Kernel(1, threads=64):
            thread = T.get_thread_binding()
            warp = thread // 32
            lane = thread % 32
            scratch = T.alloc_shared((1024,), T.float32)
            packed = T.ldg128_peer(source[thread, 0:vector_width])
            reduced = T.peer_warp_reduce128(packed, scratch, dtype)
            if warp % 2 == 0 and lane < 8:
                T.stg128_peer(destination[thread, 0:vector_width], reduced)

    return kernel


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_peer_lsu128_codegen_and_correctness():
    kernel = tilelang.compile(
        _peer_lsu128_copy,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    source = torch.randn(128, dtype=torch.float32, device="musa")
    result = kernel(source)
    torch.musa.synchronize()
    generated = kernel.get_kernel_source()
    assert "tl::load_global_128_peer_robust(" in generated
    assert "tl::store_global_128_peer_streaming(" in generated
    torch.testing.assert_close(result, source, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_peer_lsu128_frontend_rejects_non_buffer_arguments():
    with pytest.raises(TypeError, match="T.ldg128_peer expects"):
        T.ldg128_peer(1)
    with pytest.raises(TypeError, match="T.stg128_peer expects"):
        T.stg128_peer(1, T.uint32(0))
    with pytest.raises(TypeError, match="T.peer_signal_store expects"):
        T.peer_signal_store(1, 1)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_peer_release_fence_and_signal_store():
    kernel = tilelang.compile(
        _peer_signal_kernel,
        out_idx=[0],
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    result = kernel()
    torch.musa.synchronize()
    generated = kernel.get_kernel_source()
    assert "tl::peer_signal_store(" in generated
    assert "tl::peer_release_fence()" in generated
    torch.testing.assert_close(result, torch.tensor([17], dtype=torch.int32, device="musa"), atol=0, rtol=0)


@pytest.mark.parametrize("use_extern", [False, True])
@pytest.mark.parametrize("index", [0, 3, 15])
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_peer_signal_store_runtime_pointer_and_index(use_extern, index):
    kernel = tilelang.compile(
        _make_indexed_peer_signal_kernel(use_extern),
        target={"kind": "musa", "arch": "mp_31"},
        execution_backend="tvm_ffi",
    )
    signal = torch.full((16,), -77, dtype=torch.int32, device="musa")
    pointers = torch.tensor([signal.data_ptr()], dtype=torch.int64, device="musa")
    kernel(pointers, index, 12345)
    torch.musa.synchronize()
    assert "#include <tl_templates/musa/mp31/lsu.h>" in kernel.get_kernel_source()
    expected = torch.full((16,), -77, dtype=torch.int32)
    expected[index] = 12345
    torch.testing.assert_close(signal.cpu(), expected, atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_peer_warp_reduce128_codegen_and_correctness():
    torch_dtypes = {
        "float16": torch.float16,
        "bfloat16": torch.bfloat16,
        "float32": torch.float32,
    }
    for dtype, torch_dtype in torch_dtypes.items():
        vector_width = 4 if dtype == "float32" else 8
        kernel = tilelang.compile(
            _make_peer_warp_reduce_kernel(dtype),
            out_idx=[1],
            target={"kind": "musa", "arch": "mp_31"},
            execution_backend="tvm_ffi",
        )
        source = torch.ones((64, vector_width), dtype=torch_dtype, device="musa")
        result = kernel(source)
        torch.musa.synchronize()
        helper = {
            "float16": "tl::peer_warp_reduce_fp16x8(",
            "bfloat16": "tl::peer_warp_reduce_bfloat16x8(",
            "float32": "tl::peer_warp_reduce_float32x4(",
        }[dtype]
        assert helper in kernel.get_kernel_source()
        torch.testing.assert_close(result[:8], torch.full_like(result[:8], 8), atol=0, rtol=0)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_peer_warp_reduce128_frontend_rejects_invalid_dtype():
    with pytest.raises(ValueError, match="does not support dtype"):

        @T.prim_func
        def _invalid_dtype(
            source: T.Tensor((64, 4), T.float32),
        ):
            with T.Kernel(1, threads=64):
                thread = T.get_thread_binding()
                scratch = T.alloc_shared((1024,), T.float32)
                packed = T.ldg128_peer(source[thread, 0:4])
                T.peer_warp_reduce128(packed, scratch, "int32")


if __name__ == "__main__":
    tilelang.testing.main()
