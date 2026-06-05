import re

import tilelang
from tilelang import tvm
import tilelang.language as T
import tilelang.testing
from tilelang.engine.lower import lower


def _lower_tma_copy(enable_prefetch=False):
    M, K = 16, 256
    block_m, block_k = 4, 128

    @T.prim_func
    def tma_copy(x: T.Tensor((M, K), T.float16)):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(K, block_k), threads=32) as (
            pid_m,
            pid_k,
        ):
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            mbar = T.alloc_barrier(32)
            T.tma_copy(
                x[
                    pid_m * block_m : (pid_m + 1) * block_m,
                    pid_k * block_k : (pid_k + 1) * block_k,
                ],
                x_shared,
                barrier=mbar,
            )
            T.barrier_arrive(mbar)
            T.mbarrier_wait_parity(mbar, 0)

    pass_configs = {
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: enable_prefetch,
    }
    with tvm.transform.PassContext(config=pass_configs):
        return lower(tma_copy, target="musa", enable_device_compile=False)


def test_musa_tme_descriptor_load_prefetch_disabled_by_default():
    src = _lower_tma_copy().kernel_source
    assert "tl::prefetch_tma_descriptor" not in src
    assert "tl::tma_load" in src


def test_musa_tme_descriptor_load_emits_prefetch_when_enabled():
    src = _lower_tma_copy(enable_prefetch=True).kernel_source
    flat_src = " ".join(src.split())
    assert "tl::prefetch_tma_descriptor" in src
    assert "tl::tma_load" in src
    assert src.index("tl::prefetch_tma_descriptor") < src.index("tl::tma_load")
    assert re.search(
        r"tl::prefetch_tma_descriptor\(x_desc_0\)",
        flat_src,
    )


if __name__ == "__main__":
    tilelang.testing.main()
