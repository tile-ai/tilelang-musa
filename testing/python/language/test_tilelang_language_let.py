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


if __name__ == "__main__":
    tilelang.testing.main()
