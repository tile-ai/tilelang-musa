import re

import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
import torch

tilelang.disable_cache()


@tilelang.jit(target="musa")
def tma_copy_1d(A, block_N, dtype):
    N = T.const("N")
    A: T.Tensor[[N], dtype]
    C = T.empty((N,), dtype)

    with T.Kernel(T.ceildiv(N, block_N), threads=128) as bx:
        tile = T.alloc_shared((block_N,), dtype)
        T.copy(A[bx * block_N], tile, disable_tma=False)
        T.copy(tile, C[bx * block_N], disable_tma=True)

    return C


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "N, block_N",
    [
        (8192, 128),
        (4096, 128),
        (16384, 256),
    ],
)
def test_tma_1d(N, block_N):
    kernel = tma_copy_1d.compile(N=N, block_N=block_N, dtype="float32")
    code = kernel.get_kernel_source()
    tma_load_pattern = rf"tl::tma_load.*{block_N}"
    assert re.search(tma_load_pattern, code), f"tl::tma_load with block_N={block_N} not found in generated code"

    a = torch.randn(N, device="musa", dtype=torch.float32)
    c = kernel(a)
    torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)


def main():
    N = 8192
    block_N = 128
    kernel = tma_copy_1d.compile(N=N, block_N=block_N, dtype="float32")
    print(kernel.get_kernel_source())

    a = torch.randn(N, device="musa", dtype=torch.float32)
    c = kernel(a)
    torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)
    print("pass")


if __name__ == "__main__":
    main()
