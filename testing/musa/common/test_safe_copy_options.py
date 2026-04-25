import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()

BASE_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: True,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}


def make_tail_robust_force_async_copy_kernel(pass_configs):
    @tilelang.jit(target="musa", out_idx=[1], pass_configs=pass_configs)
    def kernel():
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

    return kernel


KERNEL_DISABLE_SAFE_MEMORY_ACCESS = make_tail_robust_force_async_copy_kernel(
    {
        **BASE_PASS_CONFIGS,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    }
)

KERNEL_DISABLE_SAFE_COPY_PREDICATION = make_tail_robust_force_async_copy_kernel(
    {
        **BASE_PASS_CONFIGS,
        # Keep the general safe-memory pass out of the way so this case only
        # observes T.copy-generated predication.
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_COPY_PREDICATION: True,
    }
)

KERNEL_DISABLE_SAFE_ROBUST_COPY_PREDICATION = make_tail_robust_force_async_copy_kernel(
    {
        **BASE_PASS_CONFIGS,
        # Keep the general safe-memory pass out of the way so this case only
        # observes the robust async-copy emission toggle.
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_ROBUST_COPY_PREDICATION: True,
    }
)


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_disable_safe_memory_access_does_not_remove_copy_generated_robust_conditional():
    code = KERNEL_DISABLE_SAFE_MEMORY_ACCESS().get_kernel_source()

    assert "tl::cp_async_gs_robust_conditional<16>" in code


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_disable_safe_copy_predication_removes_copy_generated_robust_conditional():
    code = KERNEL_DISABLE_SAFE_COPY_PREDICATION().get_kernel_source()

    assert "tl::cp_async_gs_robust<16>" in code
    assert "tl::cp_async_gs_robust_conditional<16>" not in code


@tilelang.testing.requires_musa_compute_version_ge(2, 2)
def test_disable_safe_robust_copy_predication_removes_robust_conditional_emission():
    code = KERNEL_DISABLE_SAFE_ROBUST_COPY_PREDICATION().get_kernel_source()

    assert "tl::cp_async_gs_robust<16>" in code
    assert "tl::cp_async_gs_robust_conditional<16>" not in code
