import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


def _has_float_vector_type(source: str, lanes: int) -> bool:
    return f"float{lanes}" in source or f"tl_f{lanes}" in source


def test_assume_remove_boundary_check():
    @tilelang.jit
    def kernel_with_assume():
        N = T.dynamic("N")

        @T.prim_func
        def main(A: T.Tensor((N,), T.float32), l: T.int32, r: T.int32):
            with T.Kernel(1, threads=32) as _:
                for i in T.serial(r - l + 1):
                    T.assume(l + i >= 0 and l + i < N)
                    A[l + i] = 0

        return main

    jit_kernel = kernel_with_assume()
    source = jit_kernel.get_kernel_source()

    assert "if (" not in source


def test_assume_enable_vectorization():
    @tilelang.jit
    def kernel_vectorize(M):
        N = T.dynamic("N")
        vectorize_size = 4

        @T.prim_func
        def main(
            A: T.Tensor((M, N), T.float32),
            B: T.Tensor((M, N), T.float32),
        ):
            with T.Kernel(1, threads=32) as _:
                tid = T.get_thread_binding()

                base_idx = tid * 4
                T.assume(N % vectorize_size == 0)

                for i in T.vectorized(vectorize_size):
                    T.assume(base_idx + i < N)
                    B[tid, base_idx + i] = A[tid, base_idx + i]

        return main

    jit_kernel = kernel_vectorize(128)
    source = jit_kernel.get_kernel_source()

    assert _has_float_vector_type(source, 4) and ("if (" not in source)


def test_assume_complex_indexing():
    @tilelang.jit
    def kernel_complex():
        M = T.dynamic("M")
        N = T.dynamic("N")

        @T.prim_func
        def main(
            A: T.Tensor((M, N), T.float32),
            B: T.Tensor((M, N), T.float32),
        ):
            with T.Kernel(1, threads=32) as _:
                tid = T.get_thread_binding()
                for j in T.serial(N):
                    i_src = T.min(j + 233, tid + 2)
                    j_src = j * T.ceildiv(j, i_src) * j - 1

                    T.assume(i_src >= 0 and i_src < M)
                    T.assume(j_src >= 0 and j_src < N)

                    B[tid, j] = A[i_src, j_src]

        return main

    jit_kernel = kernel_complex()
    source = jit_kernel.get_kernel_source()

    assert "if (" not in source


def test_assume_on_alloc_var():
    # T.assume on a mutable T.alloc_var (local.var) variable is treated as an
    # axiom and must be propagated, removing the out-of-bounds check on A[x].
    @tilelang.jit
    def kernel_alloc_var():
        N = T.dynamic("N")

        @T.prim_func
        def main(A: T.Tensor((N,), T.float32), out: T.Tensor((1,), T.float32), l: T.int32):
            with T.Kernel(1, threads=32) as _:
                x = T.alloc_var(T.int32)
                x = l
                T.assume(x >= 0 and x < N)
                out[0] = A[x]

        return main

    jit_kernel = kernel_alloc_var()
    source = jit_kernel.get_kernel_source()

    assert "if (" not in source


@pytest.mark.parametrize("mode", ["unchanged", "no_assume", "overwrite", "branch_overwrite", "opaque_write"])
def test_assume_local_var_guard_invalidation(mode):
    N = T.dynamic("N")

    @T.prim_func
    def main(A: T.Tensor((N,), T.float32), out: T.Tensor((1,), T.float32), l: T.int32):
        with T.Kernel(1, threads=32):
            x = T.alloc_var(T.int32)
            x = l
            if mode != "no_assume":
                T.assume(x >= 0 and x < N)
            if mode == "overwrite":
                x = l - N
            if mode == "branch_overwrite":
                if l > 5:
                    x = l - N
            if mode == "opaque_write":
                T.evaluate(T.call_extern("int32", "unknown_mutator", T.address_of(x)))
            out[0] = A[x]

    source = tilelang.lower(main, target={"kind": "musa", "arch": "mp_31"}, target_host="c").kernel_source
    # Only the unchanged value is covered by the assumption. A later write,
    # including a conditional write, must retain the lower-bound guard.
    assert ("0 <=" in source) == (mode != "unchanged")


if __name__ == "__main__":
    tilelang.testing.main()
