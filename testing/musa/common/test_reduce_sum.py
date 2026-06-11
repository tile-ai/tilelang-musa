import re

import tilelang
import tilelang.testing
import tilelang.language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa", verbose=True)
def reduce_sum_cross_warp(A, threads=128):
    M, N = T.const("M N")
    A: T.Tensor[[M, N], T.float32]
    B = T.empty((M,), "float32")

    with T.Kernel(1, threads=threads) as _:
        a_local = T.alloc_fragment((M, N), "float32")
        b_local = T.alloc_fragment((M,), "float32")

        T.copy(A, a_local)
        T.reduce_sum(a_local, b_local, dim=1)
        T.copy(b_local, B)

    return B


def _extract_named_barrier_base(source):
    match = re.search(r"tl::NamedBarrier<([0-9]+)>", source)
    assert match is not None, source
    return int(match.group(1))


def _extract_allreduce_threads(source):
    match = re.search(r"AllReduce<[^,]+,\s*([0-9]+),", source)
    assert match is not None, source
    return int(match.group(1))


def _assert_consecutive_barrier_init(source, base_id):
    init0 = re.search(rf"__musa_async_init_arrival\(\s*{base_id}\s*,", source)
    init1 = re.search(rf"__musa_async_init_arrival\(\s*{base_id + 1}\s*,", source)
    assert init0 is not None, source
    assert init1 is not None, source


def ref_program(x):
    return torch.sum(x, dim=1)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_reduce_sum_cross_warp():
    M = 32
    N = 512

    kernel = reduce_sum_cross_warp.compile(M=M, N=N)
    source = kernel.get_kernel_source()

    assert "tl::AllReduce<" in source, source
    allreduce_threads = _extract_allreduce_threads(source)
    assert allreduce_threads > 32, source
    barrier_base = _extract_named_barrier_base(source)
    _assert_consecutive_barrier_init(source, barrier_base)

    a = torch.randn(M, N, dtype=torch.float32, device="musa")
    b = kernel(a)
    ref = ref_program(a)
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)


def main():
    M = 32
    N = 512
    kernel = reduce_sum_cross_warp.compile(M=M, N=N)
    source = kernel.get_kernel_source()
    print(source)

    a = torch.randn(M, N, dtype=torch.float32, device="musa")
    b = kernel(a)
    ref = ref_program(a)
    torch.testing.assert_close(b, ref, rtol=1e-2, atol=1e-2)
    print("pass")


if __name__ == "__main__":
    main()
