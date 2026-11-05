"""Regression tests for LowerLDGSTG around MUSA copy-site guard attrs."""

import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()

PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG: True,
    tilelang.PassConfigKey.TL_ENABLE_LOWER_LDGSTG_PREDICATED: True,
}


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_force_async_copy_and_ldgstg():
    @T.prim_func
    def main(
        src: T.Tensor([4], T.float32),
        out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=1) as _:
            src_shared = T.alloc_shared([4], T.float32)
            for v in T.vectorized(4):
                T.copy(src[v], src_shared[v], force_async_copy=True)
            T.copy(src_shared, out)

    return main


@tilelang.jit(target="musa", out_idx=[1], pass_configs=PASS_CONFIGS)
def kernel_with_robust_load_and_ldgstg():
    @T.prim_func
    def main(
        src: T.Tensor([4], T.float32),
        out: T.Tensor([4], T.float32),
    ):
        with T.Kernel(1, threads=4) as _:
            tid = T.get_thread_binding()
            src_local = T.alloc_local([1], T.float32)
            robust_desc = T.make_robust_desc(T.address_of(src[1]), 8)
            T.copy(src[tid], src_local, src_robust_desc=robust_desc)
            out[tid] = src_local[0]

    return main


@tilelang.jit(
    target="musa",
    out_idx=[1],
    pass_configs={
        **PASS_CONFIGS,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    },
)
def kernel_with_tail_robust_force_async_copy_and_ldgstg():
    @T.prim_func
    def main(
        src: T.Tensor((18, 8), T.float16),
        out: T.Tensor((18, 8), T.float16),
    ):
        with T.Kernel(2, threads=16) as pid_m:
            src_shared = T.alloc_shared((16, 8), T.float16)
            robust_desc = T.make_robust_desc(T.address_of(src[0, 0]), 18 * 8 * 2)
            T.copy(
                src[pid_m * 16 : (pid_m + 1) * 16, 0:8],
                src_shared,
                force_async_copy=True,
                src_robust_desc=robust_desc,
            )
            T.copy(src_shared, out[pid_m * 16 : (pid_m + 1) * 16, 0:8])

    return main


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_force_async_copy_source_not_rewritten_to_ldgstg():
    code = kernel_with_force_async_copy_and_ldgstg().get_kernel_source()

    assert "tl::cp_async_gs<16>" in code
    assert "load_global_128" not in code


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_robust_load_source_not_rewritten_to_ldgstg():
    code = kernel_with_robust_load_and_ldgstg().get_kernel_source()

    assert "tl::robust_load<float>" in code
    assert "load_global_32" not in code


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_robust_force_async_copy_source_not_rewritten_to_ldgstg():
    code = kernel_with_tail_robust_force_async_copy_and_ldgstg().get_kernel_source()

    assert "tl::cp_async_gs_robust_conditional<16>" in code
    assert "load_global_128_conditional" not in code


if __name__ == "__main__":
    tilelang.testing.main()
