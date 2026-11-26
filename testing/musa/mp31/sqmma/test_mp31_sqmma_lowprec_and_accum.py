"""Tests for MP31 SQMMA W4A16 (fp16/bf16 x int4) and int8/uint8 -> float32 accum legalization."""

import pytest
import torch
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


def _pack_signed_int4_k_major(w_i8: torch.Tensor) -> torch.Tensor:
    """Pack signed int4 values in [-8, 7] along the last dim (2 nibbles/byte, low nibble = even)."""
    assert w_i8.shape[-1] % 2 == 0
    lo = (w_i8[..., 0::2].to(torch.int16) & 0x0F).to(torch.uint8)
    hi = ((w_i8[..., 1::2].to(torch.int16) & 0x0F) << 4).to(torch.uint8)
    return (lo | hi).contiguous()


@pytest.mark.parametrize("in_dtype,torch_in,low,high", [
    ("int8", torch.int8, -8, 8),
    ("uint8", torch.uint8, 0, 12),
])
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_sqmma_int8_uint8_fp32_accum(in_dtype, torch_in, low, high):
    M, N, K = 256, 256, 256
    BM, BN, BK = 128, 128, 128

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), in_dtype),
        B: T.Tensor((N, K), in_dtype),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=256) as (bx, by):
            A_s = T.alloc_shared((BM, BK), in_dtype)
            B_s = T.alloc_shared((BN, BK), in_dtype)
            C_l = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[bx * BN, k * BK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            T.copy(C_l, C[by * BM, bx * BN])

    kernel = tilelang.compile(gemm, out_idx=[2], target={"kind": "musa", "arch": "mp_31"})
    src = kernel.get_kernel_source()
    assert "tl::sqmma::s32" in src

    A = torch.randint(low, high, (M, K), device="musa", dtype=torch_in)
    B = torch.randint(low, high, (N, K), device="musa", dtype=torch_in)
    out = kernel(A, B)
    torch.musa.synchronize()
    ref = torch.matmul(A.float().cpu(), B.float().cpu().T)
    torch.testing.assert_close(out.cpu(), ref, rtol=0.0, atol=0.0)


@pytest.mark.parametrize("a_dtype,torch_a,atol", [
    ("float16", torch.float16, 1e-2),
    ("bfloat16", torch.bfloat16, 5e-2),
])
@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_sqmma_w4a16_native_int4(a_dtype, torch_a, atol):
    M, N, K = 256, 256, 128
    BM, BN, BK = 128, 128, 64

    @T.prim_func
    def gemm(
        A: T.Tensor((M, K), a_dtype),
        B: T.Tensor((N, K), "int4"),
        C: T.Tensor((M, N), "float32"),
    ):
        with T.Kernel(T.ceildiv(N, BN), T.ceildiv(M, BM), threads=256) as (bx, by):
            A_s = T.alloc_shared((BM, BK), a_dtype)
            B_s = T.alloc_shared((BN, BK), "int4")
            C_l = T.alloc_fragment((BM, BN), "float32")
            T.clear(C_l)
            for k in T.Pipelined(T.ceildiv(K, BK), num_stages=2):
                T.copy(A[by * BM, k * BK], A_s)
                T.copy(B[bx * BN, k * BK], B_s)
                T.gemm(A_s, B_s, C_l, transpose_B=True)
            T.copy(C_l, C[by * BM, bx * BN])

    kernel = tilelang.compile(gemm, out_idx=[2], target={"kind": "musa", "arch": "mp_31"})
    src = kernel.get_kernel_source()
    assert "tl::sqmma::s4" in src

    A = torch.randint(-3, 4, (M, K), device="musa", dtype=torch.int8).to(torch_a)
    B_unpacked = torch.randint(-7, 7, (N, K), device="musa", dtype=torch.int8)
    B_packed = _pack_signed_int4_k_major(B_unpacked)
    out = kernel(A, B_packed)
    torch.musa.synchronize()
    ref = torch.matmul(A.float().cpu(), B_unpacked.float().cpu().T)
    torch.testing.assert_close(out.cpu(), ref, rtol=atol, atol=atol)


if __name__ == "__main__":
    tilelang.testing.main()
