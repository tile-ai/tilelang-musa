import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm


@T.prim_func
def _system_fence_kernel(
    source: T.Tensor((1,), T.int32),
    destination: T.Tensor((1,), T.int32),
):
    with T.Kernel(1, threads=32):
        if T.get_thread_binding() == 0:
            destination[0] = source[0]
            T.fence_sys()


@tilelang.testing.requires_musa
def test_system_fence_is_registered_as_opaque():
    op = tvm.ir.Op.get("tl.musa.fence_sys")
    effect = op.get_attr("TCallEffectKind")
    assert int(effect.value) == tvm.tirx.CallEffectKind.Opaque.value


@tilelang.testing.requires_musa
def test_system_fence_codegen_and_execution():
    kernel = tilelang.compile(
        _system_fence_kernel,
        out_idx=[1],
        target="musa",
        execution_backend="tvm_ffi",
    )
    source = torch.tensor([17], dtype=torch.int32, device="musa")
    result = kernel(source)
    torch.musa.synchronize()
    assert "__threadfence_system()" in kernel.get_kernel_source()
    torch.testing.assert_close(result, source, atol=0, rtol=0)


if __name__ == "__main__":
    tilelang.testing.main()
