import re

import torch

import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_copy_with_manual_tma_barrier_end_to_end():
    n = 4096
    block_n = 128
    expected_bytes = block_n * 4

    @T.prim_func
    def kernel(A: T.Tensor((n,), "float32"), C: T.Tensor((n,), "float32")):
        with T.Kernel(T.ceildiv(n, block_n), threads=128) as bx:
            tile = T.alloc_shared((block_n,), "float32")
            barrier = T.alloc_barrier(128)
            T.copy(A[bx * block_n], tile, barrier=barrier)
            T.barrier_arrive(barrier)
            T.barrier_wait(barrier, 0)
            T.copy(tile, C[bx * block_n])

    compiled_kernel = tilelang.compile(
        kernel,
        out_idx=-1,
        target="musa",
        execution_backend="cython",
        verbose=False,
        pass_configs={
            tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
            tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        },
    )

    source = compiled_kernel.get_kernel_source()
    flat_source = " ".join(source.split())
    expect_tx = re.search(rf"__musa_async_add_trans\(([^,]+),\s*{expected_bytes}\)", flat_source)
    assert expect_tx, source
    barrier_id = re.escape(expect_tx.group(1))
    tma_load_pattern = rf"tl::tma_load\([^;]*,\s*{barrier_id}\s*,\s*{expected_bytes}\)"
    assert re.search(tma_load_pattern, flat_source), source

    a = torch.randn((n,), device="musa", dtype=torch.float32)
    c = compiled_kernel(a)

    torch.testing.assert_close(c, a, rtol=1e-6, atol=1e-6)


if __name__ == "__main__":
    tilelang.testing.main()
