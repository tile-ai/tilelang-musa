import tilelang
import tilelang.testing
from tilelang import language as T


def _has_float4_variable(source: str, name: str) -> bool:
    return f"float4 {name}" in source or f"tl_f4 {name}" in source


@tilelang.testing.requires_musa
def let_vectorize_load():
    @T.prim_func
    def main(A_ptr: T.handle):
        A = T.match_buffer(A_ptr, (16, 16), dtype=T.float32, align=16)

        for _blockIdx in T.thread_binding(1, thread="blockIdx.x"):
            for _threadIdx in T.thread_binding(128, thread="threadIdx.x"):
                b = A[0, 0:4]
                A[0, 4:8] = b

    return main


def test_let_vectorize_load():
    program = let_vectorize_load()
    kernel = tilelang.compile(program, target="musa")
    assert _has_float4_variable(kernel.get_kernel_source(), "b")


@tilelang.jit
def bind_kernel(A: T.Tensor((64,), T.float16), B: T.Tensor((64,), T.float16)):
    with T.Kernel(1, threads=32):
        A_shared = T.alloc_shared((32,), T.float16)
        for k in T.Pipelined(2, num_stages=2):
            T.copy(A[k * 32], A_shared)
            for i in T.Parallel(32):
                x = A_shared[i] + 1.0
                B[k * 32 + i] = x


@tilelang.testing.requires_cuda
def test_bind_kernel():
    kernel_source = bind_kernel.get_kernel_source()
    assert "float x" in kernel_source


@tilelang.jit
def producer_bind_kernel(A: T.Tensor((64,), T.float16), B: T.Tensor((64,), T.float16)):
    with T.Kernel(1, threads=256):
        index_shared = T.alloc_shared((1,), T.int32)
        A_shared = T.alloc_shared((32,), T.float16)
        for k in T.Pipelined(2, num_stages=2):
            # Producer-side bind depends on a shared value produced in the same pipeline body.
            index_shared[0] = k * 32
            offset = index_shared[0]
            T.copy(A[offset], A_shared)
            for i in T.Parallel(32):
                B[k * 32 + i] = A_shared[i]


@tilelang.testing.requires_cuda
def test_producer_bind_kernel():
    kernel_source = producer_bind_kernel.get_kernel_source()
    assert "tl::tma_load" in kernel_source


if __name__ == "__main__":
    tilelang.testing.main()
