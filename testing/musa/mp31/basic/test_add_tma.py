import tilelang
import tilelang.testing
import tilelang.language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def tma_add_one(A, block_M, block_N, dtype="float32"):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], dtype]
    C = T.empty((M, N), dtype)

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
        tile = T.alloc_shared((block_M, block_N), dtype)

        # TMA load: GMEM -> SMEM.
        T.copy(A[by * block_M, bx * block_N], tile)

        # Simple elementwise compute on shared memory.
        for i, j in T.Parallel(block_M, block_N):
            tile[i, j] = tile[i, j] + 1

        # TMA store: SMEM -> GMEM.
        T.copy(tile, C[by * block_M, bx * block_N])

    return C


def ref_program(x):
    return x + 1


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_tma_add_one():
    M = 4096
    N = 4096
    BLOCK_M = 128
    BLOCK_N = 128
    kernel = tma_add_one.compile(M=M, N=N, block_M=BLOCK_M, block_N=BLOCK_N)
    a = torch.randn(M, N, device="musa", dtype=torch.float32)
    c = kernel(a)
    torch.testing.assert_close(c, ref_program(a), rtol=1e-6, atol=1e-6)


def main():
    M = 4096
    N = 4096
    BLOCK_M = 128
    BLOCK_N = 128
    kernel = tma_add_one.compile(M=M, N=N, block_M=BLOCK_M, block_N=BLOCK_N)
    print(kernel.get_kernel_source())

    a = torch.randn(M, N, device="musa", dtype=torch.float32)
    c = kernel(a)
    torch.testing.assert_close(c, ref_program(a), rtol=1e-6, atol=1e-6)
    print("pass")


if __name__ == "__main__":
    main()
