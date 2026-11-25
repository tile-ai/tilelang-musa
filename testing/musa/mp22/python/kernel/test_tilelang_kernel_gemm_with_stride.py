import pytest
import tilelang
import tilelang.language as T
import tilelang.testing
import torch
from tilelang import tvm
from tilelang.layout import make_gemm_qy2_fragment_b_col, make_gemm_qy2_fragment_c
from tilelang.musa.op.gemm.gemm_qy2_mma import GemmQY2MMA


M = N = 32
BLOCK_M = BLOCK_N = BLOCK_K = 32
THREADS = 128


@pytest.fixture(autouse=True)
def disable_tilelang_cache(monkeypatch):
    cache_enabled = tilelang.is_cache_enabled()
    tilelang.disable_cache()
    monkeypatch.delenv("TILELANG_MUSA_MP22_MMA_SHAPE", raising=False)
    yield
    if cache_enabled:
        tilelang.enable_cache()


def matmul_with_stride(
    a_is_fragment: bool,
    b_is_fragment: bool,
    transpose_a: bool = False,
    transpose_b: bool = False,
    k_extent: int = BLOCK_K,
    c_is_sliced: bool = False,
):
    a_shape = (k_extent, M) if transpose_a else (M, k_extent)
    b_shape = (N, k_extent) if transpose_b else (k_extent, N)
    a_tile_shape = (BLOCK_K, BLOCK_M) if transpose_a else (BLOCK_M, BLOCK_K)
    b_tile_shape = (BLOCK_N, BLOCK_K) if transpose_b else (BLOCK_K, BLOCK_N)
    a_slice_offset = a_tile_shape[-1]
    b_slice_offset = b_tile_shape[-1]
    a_operand_shape = (a_tile_shape[0], a_slice_offset * 2)
    b_operand_shape = (b_tile_shape[0], b_slice_offset * 2)
    c_operand_shape = (BLOCK_M * 2, BLOCK_N * 2) if c_is_sliced else (BLOCK_M, BLOCK_N)
    c_row_offset = BLOCK_M if c_is_sliced else 0
    c_col_offset = BLOCK_N if c_is_sliced else 0

    @T.prim_func
    def main(
        A: T.Tensor(a_shape, "float16"),
        B: T.Tensor(b_shape, "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=THREADS):
            if a_is_fragment:
                A_shared = T.alloc_shared(a_tile_shape, "float16")
                A_operand = T.alloc_fragment(a_operand_shape, "float16")
            else:
                A_operand = T.alloc_shared(a_operand_shape, "float16")

            if b_is_fragment:
                B_shared = T.alloc_shared(b_tile_shape, "float16")
                B_operand = T.alloc_fragment(b_operand_shape, "float16")
            else:
                B_operand = T.alloc_shared(b_operand_shape, "float16")

            C_local = T.alloc_fragment(c_operand_shape, "float32")
            if not a_is_fragment:
                T.clear(A_operand)
            if not b_is_fragment:
                T.clear(B_operand)
            T.clear(C_local)

            for ko in T.Pipelined(T.ceildiv(k_extent, BLOCK_K), num_stages=0):
                if transpose_a:
                    A_global = A[ko * BLOCK_K, 0]
                else:
                    A_global = A[0, ko * BLOCK_K]
                if a_is_fragment:
                    T.copy(A_global, A_shared)
                    T.copy(A_shared, A_operand[:, a_slice_offset:])
                else:
                    T.copy(A_global, A_operand[:, a_slice_offset:])

                if transpose_b:
                    B_global = B[0, ko * BLOCK_K]
                else:
                    B_global = B[ko * BLOCK_K, 0]
                if b_is_fragment:
                    T.copy(B_global, B_shared)
                    T.copy(B_shared, B_operand[:, b_slice_offset:])
                else:
                    T.copy(B_global, B_operand[:, b_slice_offset:])

                T.gemm(
                    A_operand[:, a_slice_offset:],
                    B_operand[:, b_slice_offset:],
                    C_local[c_row_offset:, c_col_offset:],
                    transpose_a,
                    transpose_b,
                )
            T.copy(C_local[c_row_offset:, c_col_offset:], C)

    return main


def run_gemm_with_stride(
    a_is_fragment: bool,
    b_is_fragment: bool,
    transpose_a: bool = False,
    transpose_b: bool = False,
    k_extent: int = BLOCK_K,
    c_is_sliced: bool = False,
):
    kernel = tilelang.compile(
        matmul_with_stride(
            a_is_fragment,
            b_is_fragment,
            transpose_a,
            transpose_b,
            k_extent,
            c_is_sliced,
        ),
        out_idx=[2],
    )
    a_shape = (k_extent, M) if transpose_a else (M, k_extent)
    b_shape = (N, k_extent) if transpose_b else (k_extent, N)
    a = torch.randn(a_shape, device="musa", dtype=torch.float16)
    b = torch.randn(b_shape, device="musa", dtype=torch.float16)
    ref_a = a.T if transpose_a else a
    ref_b = b.T if transpose_b else b
    torch.testing.assert_close(kernel(a, b), ref_a @ ref_b, rtol=1e-2, atol=1e-2)


def run_gemm_with_stride_ss(**kwargs):
    run_gemm_with_stride(False, False, **kwargs)


def run_gemm_with_stride_sr(**kwargs):
    run_gemm_with_stride(False, True, **kwargs)


def run_gemm_with_stride_rs(**kwargs):
    run_gemm_with_stride(True, False, **kwargs)


def run_gemm_with_stride_rr(**kwargs):
    run_gemm_with_stride(True, True, **kwargs)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_with_stride_ss():
    run_gemm_with_stride_ss()


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_with_stride_sr():
    run_gemm_with_stride_sr()


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_with_stride_rs():
    run_gemm_with_stride_rs()


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_with_stride_rr():
    run_gemm_with_stride_rr()


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize(
    "a_is_fragment,b_is_fragment",
    [(False, False), (False, True), (True, False), (True, True)],
    ids=["ss", "sr", "rs", "rr"],
)
def test_gemm_with_accumulator_stride(a_is_fragment, b_is_fragment):
    run_gemm_with_stride(a_is_fragment, b_is_fragment, c_is_sliced=True)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize(
    "transpose_a, transpose_b",
    [(False, True), (True, False)],
    ids=["nt", "tn"],
)
def test_gemm_with_stride_transposed_multi_k(transpose_a, transpose_b):
    run_gemm_with_stride_ss(
        transpose_a=transpose_a,
        transpose_b=transpose_b,
        k_extent=BLOCK_K * 2,
    )


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_with_stride_m16_rr(monkeypatch):
    monkeypatch.setenv("TILELANG_MUSA_MP22_MMA_SHAPE", "m16n16k16")
    run_gemm_with_stride_rr(c_is_sliced=True)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize(
    "run_case",
    [run_gemm_with_stride_sr, run_gemm_with_stride_rs],
    ids=["sr", "rs"],
)
def test_gemm_with_stride_m8_mixed(monkeypatch, run_case):
    monkeypatch.setenv("TILELANG_MUSA_MP22_MMA_SHAPE", "m8n32k16")
    run_case(c_is_sliced=True)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_full_fragment_region_keeps_original_layout():
    buffer = tvm.tirx.decl_buffer(
        (BLOCK_N, 128),
        "float16",
        name="B_fragment",
        scope="local.fragment",
    )
    region = tvm.tirx.BufferRegion(
        buffer,
        [
            tvm.ir.Range.from_min_extent(0, BLOCK_N),
            tvm.ir.Range.from_min_extent(0, 128),
        ],
    )
    layout = make_gemm_qy2_fragment_b_col(
        BLOCK_M,
        BLOCK_N,
        128,
        BLOCK_M,
        BLOCK_N,
        16,
    )

    fitted_layout = GemmQY2MMA._fit_fragment_layout_to_region(layout, region)

    assert fitted_layout.same_as(layout)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_full_accumulator_region_keeps_original_layout():
    buffer = tvm.tirx.decl_buffer(
        (BLOCK_M, BLOCK_N),
        "float32",
        name="C_fragment",
        scope="local.fragment",
    )
    region = tvm.tirx.BufferRegion(
        buffer,
        [
            tvm.ir.Range.from_min_extent(0, BLOCK_M),
            tvm.ir.Range.from_min_extent(0, BLOCK_N),
        ],
    )
    layout = make_gemm_qy2_fragment_c(
        BLOCK_M,
        BLOCK_N,
        BLOCK_M,
        BLOCK_N,
        32,
    )

    fitted_layout = GemmQY2MMA._fit_fragment_layout_to_region(layout, region)

    assert fitted_layout.same_as(layout)


def matmul_with_leading_fragment_dims():
    @T.prim_func
    def main(
        A: T.Tensor((M, BLOCK_K), "float16"),
        B: T.Tensor((BLOCK_K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=THREADS):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            A_fragment = T.alloc_fragment((1, BLOCK_M, BLOCK_K * 2), "float16")
            B_fragment = T.alloc_fragment((1, BLOCK_K, BLOCK_N * 2), "float16")
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.copy(A, A_shared)
            T.copy(B, B_shared)
            T.copy(A_shared, A_fragment[0, :, BLOCK_K:])
            T.copy(B_shared, B_fragment[0, :, BLOCK_N:])
            T.clear(C_local)
            T.gemm(
                A_fragment[0, :, BLOCK_K:],
                B_fragment[0, :, BLOCK_N:],
                C_local,
            )
            T.copy(C_local, C)

    return main


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_with_stride_leading_fragment_dims():
    kernel = tilelang.compile(matmul_with_leading_fragment_dims(), out_idx=[2])
    a = torch.randn((M, BLOCK_K), device="musa", dtype=torch.float16)
    b = torch.randn((BLOCK_K, N), device="musa", dtype=torch.float16)
    torch.testing.assert_close(kernel(a, b), a @ b, rtol=1e-2, atol=1e-2)


def matmul_with_invalid_fragment_extent():
    @T.prim_func
    def main(
        A: T.Tensor((M, BLOCK_K), "float16"),
        B: T.Tensor((BLOCK_K, N), "float16"),
        C: T.Tensor((M, N), "float16"),
    ):
        with T.Kernel(1, threads=THREADS):
            A_shared = T.alloc_shared((BLOCK_M, BLOCK_K), "float16")
            A_fragment = T.alloc_fragment((BLOCK_M, BLOCK_K + 16), "float16")
            B_shared = T.alloc_shared((BLOCK_K, BLOCK_N), "float16")
            C_local = T.alloc_fragment((BLOCK_M, BLOCK_N), "float32")
            T.copy(A, A_shared)
            T.copy(A_shared, A_fragment[:, 16:])
            T.copy(B, B_shared)
            T.clear(C_local)
            T.gemm(A_fragment[:, 16:], B_shared, C_local)
            T.copy(C_local, C)

    return main


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_gemm_rejects_non_multiple_fragment_extent():
    with pytest.raises(
        Exception,
        match="fragment buffer extents must be multiples of the MMA layout extents",
    ):
        tilelang.compile(matmul_with_invalid_fragment_extent(), out_idx=[2])


if __name__ == "__main__":
    tilelang.testing.main()
