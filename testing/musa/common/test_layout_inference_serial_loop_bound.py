"""Regression test for analyzer scope in LayoutInference serial loops."""

import tilelang
import tilelang.language as T
import tilelang.testing


def _make_runtime_pointer_tma_kernel(
    rows=8192,
    hidden=3584,
    num_peers=7,
    persistent_blocks=16,
):
    row_steps = (rows + persistent_blocks - 1) // persistent_blocks
    threads = num_peers * 32

    @T.prim_func
    def kernel(
        source: T.Tensor((rows, hidden), T.bfloat16),
        peer_ptrs: T.Tensor((num_peers,), T.ptr),
    ):
        with T.Kernel(1, persistent_blocks, threads=threads) as (_, cta):
            tx = T.get_thread_binding(0)
            warp_id = tx // 32
            staged = T.alloc_shared((1, hidden), T.bfloat16, scope="shared.dyn")
            barrier = T.alloc_barrier(threads)

            for row_step in T.serial(row_steps):
                row = cta + row_step * persistent_blocks
                if row < rows:
                    T.tma_copy(source[row, 0], staged, barrier=barrier)
                    T.barrier_arrive(barrier)
                    T.mbarrier_wait_parity(barrier, row_step % 2)
                    peer = T.make_tensor(peer_ptrs[warp_id], (rows, hidden), T.bfloat16)
                    T.tma_copy(staged, peer[row, 0])
                    T.tma_store_wait()

    return kernel


@tilelang.testing.requires_musa
def test_layout_inference_preserves_serial_loop_bound():
    artifact = tilelang.lower(
        _make_runtime_pointer_tma_kernel(),
        target="musa",
        enable_device_compile=False,
    )
    source = str(artifact.kernel_source)
    assert "tvm_stack_alloca" not in source
    assert "tl::tma_load" in source
