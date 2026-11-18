import tilelang
import tilelang.testing
from tilelang import language as T


def test_issue_1734():
    """Test that loop-invariant if statements are hoisted out of loops."""

    @tilelang.jit()
    def kernel():
        @T.prim_func
        def main(
            A: T.Tensor[(2, 512), T.float32],
            B: T.Tensor[(2, 512), T.float32],
            C: T.Tensor[(2,), T.float32],
        ):
            with T.Kernel(1, threads=256):
                A_local = T.alloc_fragment((2, 512), T.float32)
                B_local = T.alloc_fragment((2, 512), T.float32)
                C_local = T.alloc_fragment((2,), T.float32)

                T.copy(A, A_local)
                T.copy(C, C_local)

                for i, j in T.Parallel(2, 512):
                    if C_local[i] >= 0:
                        B_local[i, j] = A_local[i, j]

                T.copy(B_local, B)

        return main

    mod = kernel.compile()
    source = mod.get_kernel_source()
    # Verify that the loop-invariant if is hoisted.  The loop may be kept in
    # the generated source or fully eliminated by vectorization.
    if_pos = source.find("if (")
    for_pos = source.find("for (")
    assert if_pos >= 0, "Loop-invariant if should be present"
    assert for_pos < 0 or if_pos < for_pos, "Loop-invariant if should be hoisted"


if __name__ == "__main__":
    tilelang.testing.main()
