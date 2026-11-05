import re

import tilelang
import tilelang.language as T
import tilelang.testing
import torch

tilelang.disable_cache()

CONSUMER_THREADS = 128
BLOCK_ELEMS = 128
NUM_STEPS = 4
NUM_ELEMS = BLOCK_ELEMS * NUM_STEPS


@tilelang.jit(target="musa")
def simt_copy_kernel(A, producer_threads=None):
    A: T.Tensor[[NUM_ELEMS], T.float16]
    B = T.empty((NUM_ELEMS,), T.float16)

    with T.Kernel(1, threads=CONSUMER_THREADS, producer_threads=producer_threads) as _:
        simt_shared = T.alloc_shared((BLOCK_ELEMS,), T.float16)
        simt_output_shared = T.alloc_shared((BLOCK_ELEMS,), T.float16)
        for stage in T.Pipelined(NUM_STEPS, num_stages=2):
            # Producer path: explicit SIMT copy from global to shared.
            # This path is intentionally non-TMA to validate producer_threads override for SIMT producer.
            for i in T.Parallel(BLOCK_ELEMS):
                simt_shared[i] = A[stage * BLOCK_ELEMS + i]
            # Consumer path: do actual compute (B = A + 1) in shared.
            for i in T.Parallel(BLOCK_ELEMS):
                simt_output_shared[i] = simt_shared[i] + T.float16(1.0)
            T.copy(simt_output_shared, B[stage * BLOCK_ELEMS : (stage + 1) * BLOCK_ELEMS])

    return B


def get_launch_bounds_threads(source: str) -> int:
    match = re.search(r"__launch_bounds__\((\d+)\s*,\s*1\)", source)
    assert match is not None, "Cannot find __launch_bounds__ in generated source."
    return int(match.group(1))


def get_producer_threads_from_source(source: str) -> int:
    return get_launch_bounds_threads(source) - CONSUMER_THREADS


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_producer_threads_simt_copy_codegen_partition():
    source_default = simt_copy_kernel.compile().get_kernel_source()
    source_override = simt_copy_kernel.compile(producer_threads=32).get_kernel_source()

    # Producer/consumer partitioning is skipped for non-TMA SIMT-only
    # pipelines, so producer_threads does not change the launch shape here.
    assert get_producer_threads_from_source(source_default) == 0
    assert get_producer_threads_from_source(source_override) == 0

    assert "vthread" not in source_default
    assert "vthread" not in source_override


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_producer_threads_simt_copy_correctness():
    kernel = simt_copy_kernel.compile(producer_threads=32)

    a = torch.randn((NUM_ELEMS,), device="musa", dtype=torch.float16)
    b = kernel(a)
    ref = a + 1.0
    torch.testing.assert_close(b, ref, rtol=0.0, atol=0.0)


def main():
    kernel = simt_copy_kernel.compile(producer_threads=32)
    print(kernel.get_kernel_source())

    a = torch.randn((NUM_ELEMS,), device="musa", dtype=torch.float16)
    b = kernel(a)
    ref = a + 1.0
    torch.testing.assert_close(b, ref, rtol=0.0, atol=0.0)
    print("pass")


if __name__ == "__main__":
    main()
