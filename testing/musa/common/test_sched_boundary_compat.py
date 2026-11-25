import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def require_musa():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA is not available")


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def sched_boundary_copy_kernel():
    @T.prim_func
    def main(
        src: T.Tensor((8,), T.float32),
        out: T.Tensor((8,), T.float32),
    ):
        with T.Kernel(1, threads=1):
            src_shared = T.alloc_shared((4,), T.float32)
            for stage in T.serial(2):
                T.copy(
                    src[stage * 4 : (stage + 1) * 4],
                    src_shared,
                    force_async_copy=True,
                )
                T.sched_boundary()
                T.copy(src_shared, out[stage * 4 : (stage + 1) * 4])

    return main


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_sched_boundary_async_copy_numerical():
    require_musa()

    src = torch.tensor(
        [1.0, -2.0, 3.5, -4.5, 5.0, -6.0, 7.25, -8.25],
        device="musa",
        dtype=torch.float32,
    )
    out = sched_boundary_copy_kernel()(src)

    if isinstance(out, (tuple, list)):
        out = out[0]

    torch.testing.assert_close(out, src, rtol=0.0, atol=0.0)
