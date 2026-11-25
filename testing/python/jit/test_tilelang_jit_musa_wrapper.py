import pytest

from tilelang import tvm
import tilelang.language as T
from tilelang.jit.adapter.musa_wrapper import TLMUSASourceWrapper


@pytest.mark.parametrize("qualifier", ["", "const ", "__attribute__((grid_constant)) const "])
def test_musa_wrapper_descriptor_qualifiers(monkeypatch, qualifier):
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), B: T.Tensor((16,), "float32"), C: T.Tensor((16,), "float32")):
        T.evaluate(0)

    wrapper = TLMUSASourceWrapper.__new__(TLMUSASourceWrapper)
    wrapper.mod = tvm.IRModule({"main": main})
    wrapper.l2_persistent_map = {}
    wrapper.use_cooperative_groups = {"kernel": False}
    wrapper.tma_im2col_output_dim_args = None
    wrapper.fast_divmod_host_args = None
    wrapper.robust_descriptor_args = None
    captured = {}

    def capture_descriptors(names, variables):
        captured.update(names)
        assert len(variables) == 2
        return ""

    monkeypatch.setattr(wrapper, "generate_tma_descriptor_args", capture_descriptors)
    source = (
        f'extern "C" __global__ void kernel({qualifier}MUtensorDescriptor A_desc_0, '
        f'{qualifier}MUtensorDescriptor B_desc_1, float* __restrict__ C) {{}}'
    )
    info = {"kernel": {"block_info": [128, 1, 1], "grid_info": [1, 1, 1],
                       "dynamic_smem_buf": 0, "function_params": list(main.params)}}
    dispatch = wrapper.create_dispatch_func(source, info)
    assert ">>>(A_desc_0, B_desc_1, C);" in dispatch
    assert captured == {"A_desc_0": "A", "B_desc_1": "B"}


def test_musa_wrapper_does_not_treat_adjacent_scalar_as_descriptor(monkeypatch):
    @T.prim_func
    def main(A: T.Tensor((16,), "float32"), A_desc_output_dim: T.int32):
        T.evaluate(0)

    wrapper = TLMUSASourceWrapper.__new__(TLMUSASourceWrapper)
    wrapper.mod = tvm.IRModule({"main": main})
    wrapper.l2_persistent_map = {}
    wrapper.use_cooperative_groups = {"kernel": False}
    wrapper.tma_im2col_output_dim_args = None
    wrapper.fast_divmod_host_args = None
    wrapper.robust_descriptor_args = None
    captured = {}
    monkeypatch.setattr(wrapper, "generate_tma_descriptor_args", lambda names, variables: captured.update(names) or "")
    source = 'extern "C" __global__ void kernel(int A_desc_output_dim, const MUtensorDescriptor A_desc) {}'
    info = {"kernel": {"block_info": [128, 1, 1], "grid_info": [1, 1, 1],
                       "dynamic_smem_buf": 0, "function_params": [main.params[1], main.params[0]]}}
    dispatch = wrapper.create_dispatch_func(source, info)
    assert ">>>(A_desc_output_dim, A_desc);" in dispatch
    assert captured == {"A_desc": "A"}
