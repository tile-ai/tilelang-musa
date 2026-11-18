import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm

tilelang.disable_cache()


PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def fp16_broadcast(lanes):
    @T.prim_func
    def main(
        pad: T.Tensor((1,), T.float32),
        output: T.Tensor((lanes,), T.float16),
    ):
        with T.Kernel(1, threads=1):
            value = T.alloc_var(T.float16)
            value = pad[0]
            for i in T.vectorized(lanes):
                output[i] = value

    return main


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("lanes", [2, 4, 8])
def test_fp16_broadcast(lanes):
    kernel = tilelang.compile(
        fp16_broadcast(lanes),
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_22"},
        pass_configs=PASS_CONFIGS,
    )
    source = kernel.get_kernel_source()
    assert f"tl_h{lanes}" in source
    assert "make_tl_h_elem(half_t(static_cast<float>(value)))" not in source
    assert f"make_tl_h{lanes}_splat(value)" in source

    pad = torch.tensor([10.0], device="musa", dtype=torch.float32)
    output = kernel(pad)
    torch.testing.assert_close(
        output,
        torch.full_like(output, 10.0),
        rtol=0,
        atol=0,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_fp16_broadcast_codegen_uses_bitwise_splat_on_musa():
    program = fp16_broadcast(8)

    qy2_target = tvm.target.Target({"kind": "musa", "arch": "mp_22"})
    with tvm.transform.PassContext(config=PASS_CONFIGS), qy2_target:
        qy2_source = tilelang.lower(
            program,
            target=qy2_target,
            enable_device_compile=False,
        ).kernel_source
    assert "make_tl_h8_splat(value)" in qy2_source

    mp31_target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with tvm.transform.PassContext(config=PASS_CONFIGS), mp31_target:
        mp31_source = tilelang.lower(
            program,
            target=mp31_target,
            enable_device_compile=False,
        ).kernel_source
    assert "make_tl_h8_splat(value)" in mp31_source
    assert "make_tl_h_elem(half_t(static_cast<float>(value)))" not in mp31_source


if __name__ == "__main__":
    tilelang.testing.main()
