import tilelang
import tilelang.language as T
import tilelang.testing


M = 64
N = 128
CONSUMER_THREADS = 256
PRODUCER_START = 512
THREADS = 640


def permute_n(index):
    local = index % 64
    return index // 64 * 64 + local % 8 * 8 + local // 8


@tilelang.jit(
    target="musa",
    pass_configs={tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True},
)
def swizzled_shared_store(A, K):
    A: T.Tensor[(M, N), T.bfloat16]
    K: T.Tensor[(N, N), T.bfloat16]
    B = T.empty((M, N), T.bfloat16)

    with T.Kernel(1, threads=THREADS):
        A_operand = T.alloc_shared((M, N), T.bfloat16)
        K_operand = T.alloc_shared((N, N), T.bfloat16)
        A_shared = A_operand
        A_accum = T.alloc_fragment((M, N), T.float32)
        A_fragment = T.alloc_fragment((M, N), T.bfloat16)
        tx = T.get_thread_binding()

        if tx >= PRODUCER_START:
            T.annotate_layout(
                {A_operand: tilelang.layout.make_sqmma_swizzled_layout(A_operand, k_major=True)},
                allow_reannotation=True,
                allow_buffer_region=True,
            )
            T.copy(A, A_operand)
            T.copy(K, K_operand)
        elif tx < CONSUMER_THREADS:
            T.gemm(
                A_operand,
                K_operand,
                A_accum,
                transpose_B=True,
                policy=T.GemmWarpPolicy.FullCol,
            )
            for i, j in T.Parallel(M, N):
                A_fragment[i, j] = T.cast(A_accum[i, j], T.bfloat16)

            T.annotate_layout(
                {A_shared: tilelang.layout.make_sqmma_swizzled_layout(A_shared, k_major=False, continuity=64)},
                allow_reannotation=True,
                allow_buffer_region=True,
            )
            for i, j in T.Parallel(M, N):
                A_shared[i, permute_n(j)] = A_fragment[i, j]
            for i, j in T.Parallel(M, N):
                B[i, j] = A_shared[i, permute_n(j)]

    return B


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_swizzled_shared_store_is_vectorized():
    kernel = swizzled_shared_store.compile()
    source = kernel.get_kernel_source()

    assert "*(tl_bf8*)(((bfloat16_t*)A_shared)" in source
    assert "= *(tl_bf8*)(A_fragment" in source


if __name__ == "__main__":
    tilelang.testing.main()
