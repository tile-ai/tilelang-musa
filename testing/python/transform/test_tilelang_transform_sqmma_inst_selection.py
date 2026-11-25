import re

import pytest
import tilelang
import tilelang.language as T
import tilelang.testing


def _transposed_a_matmul(dtype="float16", threads=512):
    M, N, K = 512, 512, 512
    block_M, block_N, block_K = 64, 64, 128

    @T.prim_func
    def gemm(
        A: T.Tensor((K, M), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_K, block_M), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[k * block_K, by * block_M], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, transpose_A=True)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def _nn_matmul(dtype="float32", block_M=256, block_N=256, block_K=32, threads=128, N=512):
    M, K = 512, 512

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), dtype),
        B: T.Tensor((K, N), dtype),
        C: T.Tensor((M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (
            bx,
            by,
        ):
            A_shared = T.alloc_shared((block_M, block_K), dtype)
            B_shared = T.alloc_shared((block_K, block_N), dtype)
            C_local = T.alloc_fragment((block_M, block_N), "float32")

            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=0):
                T.copy(A[by * block_M, k * block_K], A_shared)
                T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return gemm


def _compile_source(func):
    tilelang.disable_cache()
    try:
        kernel = tilelang.compile(
            func,
            target="musa",
            execution_backend="tvm_ffi",
            out_idx=[2],
        )
        return kernel.get_kernel_source()
    finally:
        tilelang.enable_cache()


def _has_ph1_sqmma_gemm_ss(source, m=64, n=64, k=128, warp_m=None, warp_n=None):
    for match in re.finditer(r"tl::gemm_ss<([^>]*)>", source):
        params = [param.strip() for param in match.group(1).split(",")]
        if len(params) <= 12 or params[:3] != [str(m), str(n), str(k)]:
            continue
        if warp_m is not None and params[3] != str(warp_m):
            continue
        if warp_n is not None and params[4] != str(warp_n):
            continue
        if params[12] == "true":
            return True
    return False


@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
def test_ph1_sqmma_keeps_transposed_a_multi_warp_group(dtype):
    source = _compile_source(_transposed_a_matmul(dtype=dtype, threads=512))

    assert _has_ph1_sqmma_gemm_ss(source, warp_m=8, warp_n=2)


@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_ph1_sqmma_keeps_transposed_a_single_warp_group():
    source = _compile_source(_transposed_a_matmul(dtype="float16", threads=128))

    assert _has_ph1_sqmma_gemm_ss(source)


@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize(
    "block_M, block_N, block_K",
    [
        (256, 256, 32),
        (64, 64, 128),
    ],
)
def test_ph1_sqmma_falls_back_for_tf32_multi_inst_tiles(block_M, block_N, block_K):
    source = _compile_source(
        _nn_matmul(
            dtype=T.tfloat32,
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
        )
    )

    assert not _has_ph1_sqmma_gemm_ss(source, block_M, block_N, block_K)


@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_ph1_sqmma_keeps_tf32_basic_tile():
    source = _compile_source(
        _nn_matmul(
            dtype=T.tfloat32,
            block_M=64,
            block_N=64,
            block_K=32,
        )
    )

    assert _has_ph1_sqmma_gemm_ss(source, 64, 64, 32)


@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("threads", [32, 128, 256])
@pytest.mark.parametrize("shape", [(32, 32, 32), (64, 64, 32), (64, 64, 128), (128, 64, 32)])
def test_ph1_float32_uses_full_precision_accumulation(threads, shape):
    block_M, block_N, block_K = shape
    source = _compile_source(
        _nn_matmul(
            dtype="float32",
            block_M=block_M,
            block_N=block_N,
            block_K=block_K,
            threads=threads,
        )
    )

    # FP32 must not use the reduced precision TF32 matrix instructions.
    assert not _has_ph1_sqmma_gemm_ss(source, block_M, block_N, block_K)
    assert "tl::sqmma" not in source and "tl::wmma" not in source
    assert "accum[0] = (accum[0] +" in source and "k_iter" in source


@tilelang.testing.requires_musa
@tilelang.testing.requires_musa_compute_version_ge(3, 1)
@pytest.mark.parametrize("dtype", ["float16", "bfloat16"])
@pytest.mark.parametrize("block_N", [512, 528])
def test_ph1_sqmma_compiles_n16_aligned_warp_tiles(dtype, block_N):
    # N528 needs M64 instructions although the warp tile has M128 rows.
    # Device compilation catches constraints hidden by source-only lowering.
    tilelang.lower(
        _nn_matmul(dtype=dtype, block_M=128, block_N=block_N, block_K=32, N=block_N),
        target={"kind": "musa", "arch": "mp_31"},
        enable_device_compile=True,
    )
