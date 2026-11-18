import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing


tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


@tilelang.jit(target={"kind": "musa", "arch": "mp_31"}, pass_configs=PASS_CONFIGS)
def strided_operand_panel(Q, identity, operand="B", panel_index=0):
    Q: T.Tensor((64, 128), T.bfloat16)
    identity: T.Tensor((64, 64), T.bfloat16)
    output = T.empty((64, 64), T.float32)

    is_a = operand.startswith("A")
    transpose = operand.endswith("T")
    k_major = (is_a and not transpose) or (not is_a and transpose)

    with T.Kernel(1, threads=128):
        Q_shared = T.alloc_shared((64, 128), T.bfloat16)
        Q_panels = T.alias_buffer(
            Q_shared,
            shape=(2, 64, 64),
            strides=(64, 128, 1),
        )
        identity_shared = T.alloc_shared((64, 64), T.bfloat16)
        accumulator = T.alloc_fragment((64, 64), T.float32)

        T.annotate_layout(
            {
                Q_shared: tilelang.layout.make_sqmma_swizzled_layout(
                    Q_shared,
                    continuity=128,
                    k_major=k_major,
                )
            },
            allow_reannotation=True,
        )
        T.copy(Q, Q_shared, disable_tma=True)
        T.copy(identity, identity_shared, disable_tma=True)
        T.sync_threads()

        if is_a:
            T.gemm(
                Q_panels[panel_index, :, :],
                identity_shared,
                accumulator,
                transpose_A=transpose,
                clear_accum=True,
            )
        else:
            T.gemm(
                identity_shared,
                Q_panels[panel_index, :, :],
                accumulator,
                transpose_A=True,
                transpose_B=transpose,
                clear_accum=True,
            )
        T.copy(accumulator, output)

    return output


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
@pytest.mark.parametrize("operand", ["A", "AT", "B", "BT"])
@pytest.mark.parametrize("panel_index", [0, 1])
def test_sqmma_strided_alias_data_correctness(operand, panel_index):
    # Distinct rows and panels expose a compact-stride or panel-offset error.
    row = torch.arange(64, dtype=torch.float32)[:, None]
    panel = torch.arange(128, dtype=torch.float32)[None, :] // 64
    q_cpu = (row * 2 + panel).to(torch.bfloat16)
    identity_cpu = torch.eye(64, dtype=torch.bfloat16)

    compiled = strided_operand_panel.compile(operand=operand, panel_index=panel_index)
    source = compiled.get_kernel_source()
    match = re.search(r"tl::gemm_ss<([^>]*)>", source)
    assert match is not None
    params = [value.strip() for value in match.group(1).split(",")]
    stride_index = 8 if operand.startswith("A") else 9
    assert params[stride_index] == "128"
    assert params[12] == "true"

    actual = compiled(q_cpu.to("musa"), identity_cpu.to("musa")).cpu()
    expected = q_cpu[:, panel_index * 64 : (panel_index + 1) * 64].float()
    if operand.endswith("T"):
        expected = expected.T
    torch.testing.assert_close(actual, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
