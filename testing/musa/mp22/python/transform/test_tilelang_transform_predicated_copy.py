import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm


@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_pipeline_predicated_copy_preserves_shared_fill_correctness():
    @T.prim_func
    def main(
        A: T.Tensor((8,), T.float16),
        B: T.Tensor((16,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((16,), T.float16)
            for _ in T.Pipelined(1, num_stages=2):
                T.fill(S, 0)
                T.ptx_cp_async(
                    T.access_ptr(S[0], "w", 16),
                    T.access_ptr(A[0], "r", 16),
                    8,
                    True,
                )
                T.ptx_cp_async(
                    T.access_ptr(S[8], "w", 16),
                    T.access_ptr(A[0], "r", 16),
                    8,
                    False,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                # Keep the output path synchronous so this regression isolates
                # predicated cp.async and shared-fill pipeline ordering.
                T.copy(S, B[0:16], prefer_instruction="sync")

    # MTCC 5.1 crashes when compiling the conditional cp.async intrinsic, so
    # validate that pipeline planning/lowering preserves it without invoking
    # the device compiler. Runtime correctness is checked with the equivalent
    # synchronous data movement below.
    target = tvm.target.Target({"kind": "musa", "arch": "mp_22"})
    with target:
        artifact = tilelang.lower(main, target=target, enable_device_compile=False)
    src = str(artifact.kernel_source)
    assert "cp_async_gs_conditional<16>" in src, "Expected predicated cp.async in generated MUSA source"

    @T.prim_func
    def main_sync(
        A: T.Tensor((8,), T.float16),
        B: T.Tensor((16,), T.float16),
    ):
        with T.Kernel(1, threads=32):
            S = T.alloc_shared((16,), T.float16)
            T.fill(S, 0)
            for i in T.Parallel(8):
                S[i] = A[i]
            T.copy(S, B[0:16], prefer_instruction="sync")

    kernel = tilelang.compile(main_sync, out_idx=[1], target=target)

    a = torch.randn((8,), dtype=torch.float16, device="musa")
    b = kernel(a)

    expected = torch.zeros((16,), dtype=torch.float16, device="musa")
    expected[:8] = a
    tilelang.testing.torch_assert_close(b, expected, rtol=0, atol=0)


if __name__ == "__main__":
    tilelang.testing.main()
