import pytest
import tilelang
import tilelang.testing
import torch
from tilelang import tvm
from tilelang import language as T

tilelang.disable_cache()


def _get_test_target_and_device() -> tuple[str, str]:
    if hasattr(torch, "musa") and torch.musa.is_available():
        return "musa", "musa"
    if torch.cuda.is_available():
        return "cuda", "cuda"
    pytest.skip("Neither MUSA nor CUDA is available")


def _sync(device: str) -> None:
    if device == "musa":
        torch.musa.synchronize()
    elif device == "cuda":
        torch.cuda.synchronize()


def _make_pipeline_copy_kernel(producer_threads: int | None):
    @T.prim_func
    def main(
        A: T.Tensor((256,), T.float16),
        B: T.Tensor((256,), T.float16),
    ):
        with T.Kernel(1, threads=128, producer_threads=producer_threads):
            A_shared = T.alloc_shared((128,), T.float16)
            for k in T.Pipelined(2, num_stages=2):
                T.copy(A[k * 128 : (k + 1) * 128], A_shared)
                T.copy(A_shared, B[k * 128 : (k + 1) * 128])

    return main


def _thread_idx_x_extent(func: tvm.tir.PrimFunc) -> int:
    extents: list[int] = []

    def visitor(node):
        if not isinstance(node, tvm.tir.AttrStmt):
            return
        if node.attr_key != "thread_extent":
            return
        if not isinstance(node.node, tvm.tir.IterVar):
            return
        if node.node.thread_tag != "threadIdx.x":
            return
        assert isinstance(node.value, tvm.tir.IntImm)
        extents.append(int(node.value.value))

    tvm.tir.stmt_functor.post_order_visit(func.body, visitor)
    assert extents, "Cannot find threadIdx.x thread_extent in lowered TIR."
    return max(extents)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_producer_threads_override_changes_thread_partition():
    target, _ = _get_test_target_and_device()
    base_kernel = _make_pipeline_copy_kernel(None)
    overridden_kernel = _make_pipeline_copy_kernel(32)

    base_artifact = tilelang.lower(base_kernel, target=target)
    overridden_artifact = tilelang.lower(overridden_kernel, target=target)

    base_extent = _thread_idx_x_extent(base_artifact.device_mod["main"])
    overridden_extent = _thread_idx_x_extent(overridden_artifact.device_mod["main"])

    # Default: producer=consumer=128 => threadIdx.x extent is 256.
    assert base_extent == 256
    # With override: producer=32, consumer=128 => threadIdx.x extent is 160.
    assert overridden_extent == 160


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_producer_threads_runtime_copy():
    target, device = _get_test_target_and_device()
    kernel = tilelang.compile(_make_pipeline_copy_kernel(32), target=target)

    A = torch.randn((256,), device=device, dtype=torch.float16)
    B = torch.empty_like(A)
    kernel(A, B)
    _sync(device)
    torch.testing.assert_close(B, A)


if __name__ == "__main__":
    tilelang.testing.main()
