import re

import pytest
import tilelang
from tilelang import tvm as tvm
import tilelang.testing


@pytest.fixture(autouse=True)
def disable_tilelang_cache(monkeypatch):
    cache_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    yield
    if cache_enabled:
        tilelang.enable_cache()


THREADS = [128, 256, 512, 1024]
TRANSPOSE_CASES = [
    (False, False, "nn"),
    (False, True, "nt"),
    (True, False, "tn"),
    (True, True, "tt"),
]


def _uses_inst_shape(kernel_source, op_name, inst_shape):
    shape = r",\s*".join(str(value) for value in inst_shape)
    return re.search(rf"tl::{op_name}<[^;]*,\s*{shape}>", kernel_source)


def _uses_warp_partition(kernel_source, op_name, block_shape, warp_partition):
    block = r",\s*".join(str(value) for value in block_shape)
    warp = r",\s*".join(str(value) for value in warp_partition)
    return re.search(rf"tl::{op_name}<\s*{block},\s*{warp},", kernel_source)


def _uses_qy2_subgroup(kernel_source, op_name, inst_shape, subgroup_size):
    shape = r",\s*".join(str(value) for value in inst_shape)
    return re.search(
        rf"tl::{op_name}<[^;]*,\s*{shape},\s*{subgroup_size},\s*0>",
        kernel_source,
    )


@pytest.mark.parametrize(
    "mma_shape", [None, "m8n32k16", "m16n16k16", "m32n32k16"]
)
@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_qy2_target_uses_default_wave128(monkeypatch, mma_shape):
    if mma_shape is None:
        monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    else:
        monkeypatch.setenv("TILELANG_MUSA_MP22_MMA_SHAPE", mma_shape)
    target = tvm.target.Target({"kind": "musa", "arch": "mp_22"})
    assert int(tilelang._ffi_api.TargetGetWarpSize(target)) == 128


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_rejects_invalid_mma_shape(monkeypatch):
    monkeypatch.setenv("TILELANG_MUSA_MP22_MMA_SHAPE", "m16n16k61")
    program = matmul(
        128,
        128,
        16,
        128,
        128,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        128,
    )
    with pytest.raises(
        Exception,
        match="Invalid TILELANG_MUSA_MP22_MMA_SHAPE: m16n16k61",
    ):
        tilelang.compile(program, out_idx=[2])


def matmul(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_shared, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=0,
    num_threads=128,
):
    program = matmul(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )
    kernel = tilelang.compile(program, out_idx=[2])
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        if dtypeAccum in ("float16", "bfloat16") or in_dtype == "int8":
            return tilelang.testing.matmul_naive(A, B, getattr(torch, dtypeAccum), getattr(torch, out_dtype))
        if in_dtype == "float32":
            # Convert float32 to tfloat32 because tfloat32 mma cannot truncate
            # float32 automatically, -0x1000 meas
            A = (A.view(torch.int32) - 0x1000).view(torch.float32)
            B = (B.view(torch.int32) - 0x1000).view(torch.float32)
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_i8i8i32(trans_A, trans_B, layout_tag, num_threads):
    run_gemm(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "int8",
        "int8",
        "int32",
        128,
        128,
        32,
        0,
        num_threads,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f16f16f32(trans_A, trans_B, layout_tag, num_threads):
    run_gemm(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_auto_m8_fallback(monkeypatch):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        8,
        128,
        16,
        8,
        128,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        128,
    )
    kernel_source = tilelang.compile(program, out_idx=[2]).get_kernel_source()
    assert _uses_warp_partition(kernel_source, "gemm_ss", (8, 128, 16), (1, 4))
    assert _uses_inst_shape(kernel_source, "gemm_ss", (8, 32, 16))


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("block_m, block_n", [(16, 64), (64, 16)])
def test_gemm_auto_m16_fallback(monkeypatch, block_m, block_n):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        block_m,
        block_n,
        16,
        block_m,
        block_n,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        128,
    )
    kernel_source = tilelang.compile(program, out_idx=[2]).get_kernel_source()
    expected_partition = (1, 4) if block_m == 16 else (4, 1)
    assert _uses_warp_partition(
        kernel_source,
        "gemm_ss",
        (block_m, block_n, 16),
        expected_partition,
    )
    assert _uses_inst_shape(kernel_source, "gemm_ss", (16, 16, 16))


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", [32, 64])
def test_gemm_auto_m16_fallback_for_small_thread_block(monkeypatch, num_threads):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        128,
        128,
        16,
        128,
        128,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        num_threads,
    )
    kernel_source = tilelang.compile(program, out_idx=[2]).get_kernel_source()
    assert _uses_inst_shape(kernel_source, "gemm_ss", (16, 16, 16))


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_auto_m16_fallback_when_m32_cannot_partition(monkeypatch):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        32,
        32,
        16,
        32,
        32,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        64,
    )
    kernel_source = tilelang.compile(program, out_idx=[2]).get_kernel_source()
    assert _uses_inst_shape(kernel_source, "gemm_ss", (16, 16, 16))


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", [32, 64])
def test_gemm_auto_m8_fallback_for_small_thread_block(monkeypatch, num_threads):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        8,
        128,
        16,
        8,
        128,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        num_threads,
    )
    kernel_source = tilelang.compile(program, out_idx=[2]).get_kernel_source()
    assert _uses_inst_shape(kernel_source, "gemm_ss", (8, 32, 16))


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize(
    "block_m, expected_shape",
    [(16, (16, 16, 16)), (8, (8, 32, 16))],
)
def test_gemm_bf16_auto_fallback(monkeypatch, block_m, expected_shape):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        block_m,
        128,
        16,
        block_m,
        128,
        16,
        False,
        False,
        "bfloat16",
        "bfloat16",
        "float32",
        0,
        32,
    )
    kernel_source = tilelang.compile(program, out_idx=[2]).get_kernel_source()
    assert _uses_inst_shape(kernel_source, "gemm_ss", expected_shape)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_auto_rejects_unsupported_shape(monkeypatch):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        8,
        16,
        16,
        8,
        16,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        32,
    )
    with pytest.raises(Exception, match="No QY2 MMA instruction shape supports"):
        tilelang.compile(program, out_idx=[2])


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize(
    "block_m, block_n, num_threads",
    [
        (16, 16, 128),
        (32, 32, 256),
    ],
)
def test_gemm_auto_rejects_unpartitionable_thread_block(
    monkeypatch, block_m, block_n, num_threads
):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        block_m,
        block_n,
        16,
        block_m,
        block_n,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        num_threads,
    )
    with pytest.raises(Exception, match="No QY2 MMA instruction shape supports"):
        tilelang.compile(program, out_idx=[2])


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("mma_shape", ["m16n16k16", "m8n32k16"])
def test_gemm_explicit_wave32_rejects_incompatible_threads(monkeypatch, mma_shape):
    monkeypatch.setenv("TILELANG_MUSA_MP22_MMA_SHAPE", mma_shape)
    program = matmul(
        32,
        32,
        16,
        32,
        32,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        16,
    )
    with pytest.raises(Exception, match=rf"Explicit QY2 MMA shape {mma_shape} is incompatible"):
        tilelang.compile(program, out_idx=[2])


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize(
    "mma_shape, block_m, block_n, num_threads",
    [
        ("m32n32k16", 32, 32, 256),
        ("m16n16k16", 16, 16, 128),
        ("m8n32k16", 8, 32, 128),
    ],
)
def test_gemm_explicit_shape_rejects_unpartitionable_thread_block(
    monkeypatch, mma_shape, block_m, block_n, num_threads
):
    monkeypatch.setenv("TILELANG_MUSA_MP22_MMA_SHAPE", mma_shape)
    program = matmul(
        block_m,
        block_n,
        16,
        block_m,
        block_n,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        num_threads,
    )
    with pytest.raises(
        Exception,
        match=rf"Explicit QY2 MMA shape {mma_shape} is incompatible",
    ):
        tilelang.compile(program, out_idx=[2])


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_uses_qy2_python_lowering(monkeypatch):
    from tilelang.musa.op.gemm.gemm_qy2_mma import GemmQY2MMA

    calls = 0
    original_lower = GemmQY2MMA.lower

    def traced_lower(self, *args, **kwargs):
        nonlocal calls
        calls += 1
        return original_lower(self, *args, **kwargs)

    monkeypatch.setattr(GemmQY2MMA, "lower", traced_lower)
    program = matmul(
        32,
        32,
        16,
        32,
        32,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        128,
    )
    tilelang.compile(program, out_idx=[2])
    assert calls > 0


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_qy2_thread_y_gemm_uses_subgroup_lowering(monkeypatch):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        128,
        128,
        16,
        128,
        128,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        (128, 4),
    )
    target = tvm.target.Target({"kind": "musa", "arch": "mp_22"})
    with target:
        kernel_source = tilelang.lower(program, target=target).kernel_source

    assert _uses_qy2_subgroup(
        kernel_source,
        "gemm_ss",
        (32, 32, 16),
        128,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_ph1_thread_y_gemm_does_not_use_qy2_subgroup_lowering():
    program = matmul(
        128,
        128,
        16,
        128,
        128,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        (128, 4),
    )
    target = tvm.target.Target({"kind": "musa", "arch": "mp_31"})
    with target:
        kernel_source = tilelang.lower(program, target=target).kernel_source

    assert not _uses_qy2_subgroup(
        kernel_source,
        "gemm_ss",
        (32, 32, 16),
        128,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_m32_uses_wave128(monkeypatch):
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    program = matmul(
        128,
        128,
        16,
        128,
        128,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        128,
    )
    kernel_source = tilelang.compile(program, out_idx=[2]).get_kernel_source()
    assert _uses_warp_partition(
        kernel_source, "gemm_ss", (128, 128, 16), (1, 1)
    )
    assert not _uses_inst_shape(kernel_source, "gemm_ss", (8, 32, 16))
    assert not _uses_inst_shape(kernel_source, "gemm_ss", (16, 16, 16))


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_explicit_m32_disables_m16_fallback(monkeypatch):
    monkeypatch.setenv("TILELANG_MUSA_MP22_MMA_SHAPE", "m32n32k16")
    program = matmul(
        16,
        64,
        16,
        16,
        64,
        16,
        False,
        False,
        "float16",
        "float16",
        "float32",
        0,
        128,
    )
    with pytest.raises(Exception, match="Explicit QY2 MMA shape m32n32k16 is incompatible"):
        tilelang.compile(program, out_idx=[2])


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f32f32f32(trans_A, trans_B, layout_tag, num_threads):
    run_gemm(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float32",
        "float32",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


def matmul_sr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_shared = T.alloc_shared(A_shared_shape, in_dtype)
            B_local = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_shared)
                else:
                    T.copy(A[by * block_M, k * block_K], A_shared)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_local)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_local)
                T.gemm(A_shared, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_sr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
):

    program = matmul_sr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2], verbose=True)
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        A = A.to(torch.float)
        B = B.to(torch.float)
        C = torch.matmul(A, B)
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_i8i8i32_sr(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_sr(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "int8",
        "int8",
        "int32",
        128,
        128,
        32,
        0,
        num_threads,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f16f16f32_sr(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_sr(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f32f32f32_sr(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_sr(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float32",
        "float32",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


def matmul_rr(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_local = T.alloc_fragment(B_shared_shape, in_dtype)
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_local)
                else:
                    T.copy(A[by * block_M, k * block_K], A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_local)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_local)
                T.gemm(A_local, B_local, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rr(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
):
    program = matmul_rr(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_i8i8i32_rr(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_rr(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "int8",
        "int8",
        "int",
        128,
        128,
        32,
        0,
        num_threads,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f16f16f32_rr(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_rr(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f32f32f32f32_rr(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_rr(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float32",
        "float32",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


def matmul_rs(
    M,
    N,
    K,
    block_M,
    block_N,
    block_K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    accum_dtype,
    num_stages,
    threads,
):
    A_shape = (K, M) if trans_A else (M, K)
    B_shape = (N, K) if trans_B else (K, N)
    A_shared_shape = (block_K, block_M) if trans_A else (block_M, block_K)
    B_shared_shape = (block_N, block_K) if trans_B else (block_K, block_N)

    import tilelang.language as T

    @T.prim_func
    def main(
        A: T.Tensor(A_shape, in_dtype),
        B: T.Tensor(B_shape, in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=threads) as (bx, by):
            A_local = T.alloc_fragment(A_shared_shape, in_dtype)
            B_shared = T.alloc_shared(B_shared_shape, in_dtype, scope="shared")
            C_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            T.clear(C_local)
            for k in T.Pipelined(T.ceildiv(K, block_K), num_stages=num_stages):
                if trans_A:
                    T.copy(A[k * block_K, by * block_M], A_local)
                else:
                    T.copy(A[by * block_M, k * block_K], A_local)
                if trans_B:
                    T.copy(B[bx * block_N, k * block_K], B_shared)
                else:
                    T.copy(B[k * block_K, bx * block_N], B_shared)
                T.gemm(A_local, B_shared, C_local, trans_A, trans_B)
            T.copy(C_local, C[by * block_M, bx * block_N])

    return main


def run_gemm_rs(
    M,
    N,
    K,
    trans_A,
    trans_B,
    in_dtype,
    out_dtype,
    dtypeAccum,
    block_M,
    block_N,
    block_K,
    num_stages=1,
    num_threads=128,
):
    program = matmul_rs(
        M,
        N,
        K,
        block_M,
        block_N,
        block_K,
        trans_A,
        trans_B,
        in_dtype,
        out_dtype,
        dtypeAccum,
        num_stages,
        num_threads,
    )

    kernel = tilelang.compile(program, out_idx=[2])
    print(kernel.get_kernel_source())
    profiler = kernel.get_profiler()

    def ref_program(A, B):
        import torch

        if trans_A:
            A = A.T
        if trans_B:
            B = B.T
        C = torch.matmul(A.to(torch.float), B.to(torch.float))
        C = C.to(torch.__getattribute__(out_dtype))
        return C

    profiler.assert_allclose(ref_program, atol=1e-2, rtol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_i8i8i32_rs(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_rs(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "int8",
        "int8",
        "int32",
        128,
        128,
        32,
        0,
        num_threads,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f16f16f32_rs(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_rs(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float16",
        "float16",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize("num_threads", THREADS)
@pytest.mark.parametrize(
    "trans_A, trans_B, layout_tag",
    TRANSPOSE_CASES,
    ids=[case[2] for case in TRANSPOSE_CASES],
)
def test_gemm_f32f32f32_rs(trans_A, trans_B, layout_tag, num_threads):
    run_gemm_rs(
        512,
        1024,
        768,
        trans_A,
        trans_B,
        "float32",
        "float32",
        "float32",
        128,
        128,
        32,
        0,
        num_threads,
    )


if __name__ == "__main__":
    tilelang.testing.main()
