import pytest
import tilelang
import tilelang.testing
import tilelang.language as T
import torch

tilelang.disable_cache()

COMMON_PASS_CONFIGS = {
    tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
    tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
    tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
}

SQMMA_PASS_CONFIGS = {
    **COMMON_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
}

WMMA_PASS_CONFIGS = {
    **COMMON_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_DISABLE_SQMMA: True,
}

FMA_PASS_CONFIGS = {
    **COMMON_PASS_CONFIGS,
    tilelang.PassConfigKey.TL_DISABLE_SQMMA: True,
    tilelang.PassConfigKey.TL_DISABLE_PH1_WMMA: True,
}


def _make_identity_gemm_ws_offset(
    *,
    pass_configs,
    threads: int,
    ws_groups: tuple[int, ...],
    wg_wait: int = 0,
    wait_wgmma: bool = False,
    compile_flags: list[str] | None = None,
):
    jit_kwargs = {
        "target": "musa",
        "out_idx": [],
        "pass_configs": pass_configs,
    }
    if compile_flags is not None:
        jit_kwargs["compile_flags"] = compile_flags

    @tilelang.jit(**jit_kwargs)
    def identity_gemm_ws_offset():
        M = N = K = 64

        @T.prim_func
        def kernel(
            a: T.Tensor([M, K], "float16"),
            b: T.Tensor([K, N], "float16"),
            c: T.Tensor([M, N], "float32"),
        ):
            with T.Kernel(1, threads=threads) as _:
                a_shared = T.alloc_shared([M, K], "float16")
                b_shared = T.alloc_shared([K, N], "float16")
                c_frag = T.alloc_fragment([M, N], "float32")

                with T.ws(*ws_groups):
                    T.copy(a, a_shared)
                    T.copy(b, b_shared)
                    T.gemm(a_shared, b_shared, c_frag, clear_accum=True, wg_wait=wg_wait)
                    if wait_wgmma:
                        T.wait_wgmma(0)
                    T.copy(c_frag, c)

        return kernel

    return identity_gemm_ws_offset


identity_gemm_sqmma_ws_offset = _make_identity_gemm_ws_offset(
    pass_configs=SQMMA_PASS_CONFIGS,
    threads=512,
    ws_groups=(1, 2),
    wg_wait=-1,
    wait_wgmma=True,
    compile_flags=["-O3", "-DENABLE_BF16"],
)

identity_gemm_wmma_ws_offset = _make_identity_gemm_ws_offset(
    pass_configs=WMMA_PASS_CONFIGS,
    threads=256,
    ws_groups=(1,),
)

identity_gemm_fma_ws_offset = _make_identity_gemm_ws_offset(
    pass_configs=FMA_PASS_CONFIGS,
    threads=256,
    ws_groups=(1,),
)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "kernel_factory",
    [
        identity_gemm_sqmma_ws_offset,
        identity_gemm_wmma_ws_offset,
        identity_gemm_fma_ws_offset,
    ],
    ids=["sqmma", "wmma", "fma"],
)
def test_gemm_ws_thread_offset_identity(kernel_factory):
    M = N = K = 64
    a = torch.eye(M, K, device="musa", dtype=torch.float16)
    b = torch.eye(K, N, device="musa", dtype=torch.float16)
    c = torch.zeros(M, N, device="musa", dtype=torch.float32)

    kernel = kernel_factory()
    kernel(a, b, c)

    ref = torch.mm(a.float(), b.float())
    torch.testing.assert_close(c.cpu(), ref.cpu(), rtol=0.0, atol=0.0)
