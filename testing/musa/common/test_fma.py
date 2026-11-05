import tilelang
import tilelang.language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa")
def matmul_fma_relu(
    A,
    B,
    block_M,
    block_N,
    block_K,
    thread_tile_m=8,
    thread_tile_n=8,
    dtype="float16",
    accum_dtype="float32",
):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], dtype]
    B: T.Tensor[[K, N], dtype]
    C = T.empty((M, N), dtype)

    threads_m = block_M // thread_tile_m
    threads_n = block_N // thread_tile_n
    assert block_M % thread_tile_m == 0 and block_N % thread_tile_n == 0
    threads = threads_m * threads_n

    with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
        a_shared = T.alloc_shared((block_M, block_K), dtype)
        b_shared = T.alloc_shared((block_K, block_N), dtype)
        c_local = T.alloc_local((thread_tile_m, thread_tile_n), accum_dtype)

        tid = T.get_thread_binding()
        lane_m = tid % threads_m
        lane_n = tid // threads_m

        T.clear(c_local)
        for ko in T.Pipelined(T.ceildiv(K, block_K), num_stages=2):
            T.copy(A[by * block_M, ko * block_K], a_shared)
            T.copy(B[ko * block_K, bx * block_N], b_shared)

            k_tile = T.min(block_K, K - ko * block_K)
            for kk in T.serial(k_tile):
                for i in T.serial(thread_tile_m):
                    a_val = T.cast(a_shared[lane_m * thread_tile_m + i, kk], accum_dtype)
                    for j in T.serial(thread_tile_n):
                        b_val = T.cast(b_shared[kk, lane_n * thread_tile_n + j], accum_dtype)
                        c_local[i, j] += a_val * b_val

        zero = T.cast(0, accum_dtype)
        for i, j in T.grid(thread_tile_m, thread_tile_n):
            val = T.max(c_local[i, j], zero)
            row = by * block_M + lane_m * thread_tile_m + i
            col = bx * block_N + lane_n * thread_tile_n + j
            if row < M and col < N:
                C[row, col] = val

    return C


def _assert_case(M, N, K, block_M, block_N, block_K, thread_tile_m, thread_tile_n):
    kernel = matmul_fma_relu.compile(
        M=M,
        N=N,
        K=K,
        block_M=block_M,
        block_N=block_N,
        block_K=block_K,
        thread_tile_m=thread_tile_m,
        thread_tile_n=thread_tile_n,
        dtype="float16",
        accum_dtype="float32",
    )

    a = torch.randn(M, K, device="musa", dtype=torch.float16)
    b = torch.randn(K, N, device="musa", dtype=torch.float16)
    c = kernel(a, b)
    ref = torch.relu(a @ b)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)
    return kernel


def test_fma():
    _assert_case(
        M=1024,
        N=1024,
        K=1024,
        block_M=128,
        block_N=128,
        block_K=32,
        thread_tile_m=8,
        thread_tile_n=8,
    )


def main():
    kernel = _assert_case(
        M=1024,
        N=1024,
        K=1024,
        block_M=128,
        block_N=128,
        block_K=32,
        thread_tile_m=8,
        thread_tile_n=8,
    )
    print(kernel.get_kernel_source())
    print("pass")


if __name__ == "__main__":
    main()
