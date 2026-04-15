import tilelang
import tilelang.testing
from tilelang import language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def gemm_reduce_sum(A, B, threads=256):
    M, N, K = T.const("M N K")
    A: T.Tensor[[M, K], T.float16]
    B: T.Tensor[[N, K], T.float16]
    out = T.empty((M,), "float32")

    with T.Kernel(1, threads=threads) as _:
        a_shared = T.alloc_shared((M, K), "float16")
        b_shared = T.alloc_shared((N, K), "float16")
        acc = T.alloc_fragment((M, N), "float32")
        out_local = T.alloc_fragment((M,), "float32")

        T.clear(acc)
        T.copy(A, a_shared)
        T.copy(B, b_shared)

        T.gemm(a_shared, b_shared, acc, transpose_B=True)
        T.reduce_sum(acc, out_local, dim=1, clear=True)
        T.copy(out_local, out)

    return out


def gemm_reduce_sum_ref(A, B):
    scores = torch.matmul(A.float(), B.float().transpose(0, 1))
    return scores.sum(dim=1)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_gemm_reduce_sum():
    torch.random.manual_seed(2026)

    M, N, K = 128, 128, 64
    kernel = gemm_reduce_sum.compile(M=M, N=N, K=K)

    A = torch.randn(M, K, device="musa", dtype=torch.float16)
    B = torch.randn(N, K, device="musa", dtype=torch.float16)
    out = kernel(A, B)
    out_ref = gemm_reduce_sum_ref(A, B).float()

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)


def main():
    torch.random.manual_seed(2026)

    M, N, K = 128, 128, 64
    kernel = gemm_reduce_sum.compile(M=M, N=N, K=K)
    print(kernel.get_kernel_source())

    A = torch.randn(M, K, device="musa", dtype=torch.float16)
    B = torch.randn(N, K, device="musa", dtype=torch.float16)
    out = kernel(A, B)
    out_ref = gemm_reduce_sum_ref(A, B).float()

    torch.testing.assert_close(out, out_ref, rtol=1e-2, atol=1e-2)
    print("pass")


if __name__ == "__main__":
    main()
