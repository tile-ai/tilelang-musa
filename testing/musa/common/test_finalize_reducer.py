import re

import tilelang
import tilelang.testing
import tilelang.language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def finalize_reducer_sum_cross_warp(A, threads=128):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], T.float32]
    B = T.empty((M,), "float32")

    with T.Kernel(1, threads=threads) as _:
        A_local = T.alloc_fragment((M, N), "float32")
        R = T.alloc_reducer(M, "float32", replication="all")

        T.copy(A, A_local)
        T.clear(R)
        for i, j in T.Parallel(M, N):
            R[i] += A_local[i, j]
        T.finalize_reducer(R)
        T.copy(R, B)

    return B


def _extract_named_barrier_base(source):
    match = re.search(r"tl::NamedBarrier<([0-9]+)>", source)
    assert match is not None, source
    return int(match.group(1))


def _assert_consecutive_barrier_init(source, base_id):
    init0 = re.search(rf"__musa_async_init_arrival\(\s*{base_id}\s*,", source)
    init1 = re.search(rf"__musa_async_init_arrival\(\s*{base_id + 1}\s*,", source)
    assert init0 is not None, source
    assert init1 is not None, source


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_finalize_reducer_named_barrier_cross_warp():
    M = 4
    N = 128
    threads = 128

    kernel = finalize_reducer_sum_cross_warp.compile(M=M, N=N, threads=threads)
    source = kernel.get_kernel_source()

    assert "tl::AllReduce<" in source, source
    assert "tl::AllReduceWS" not in source, source
    barrier_base = _extract_named_barrier_base(source)
    _assert_consecutive_barrier_init(source, barrier_base)

    a = torch.randn(M, N, dtype=torch.float32, device="musa")
    b = kernel(a)
    ref = torch.sum(a, dim=1)
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


def main():
    M = 4
    N = 128
    threads = 128

    kernel = finalize_reducer_sum_cross_warp.compile(M=M, N=N, threads=threads)
    source = kernel.get_kernel_source()
    print(source)

    a = torch.randn(M, N, dtype=torch.float32, device="musa")
    b = kernel(a)
    ref = torch.sum(a, dim=1)
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)
    print("pass")


if __name__ == "__main__":
    main()
