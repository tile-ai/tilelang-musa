import tilelang
import tilelang.language as T
from tilelang.utils.sparse import get_e_factor


def _sparse_matmul_source():
    M = 16
    N = 8
    K = 32
    block_M = 16
    block_N = 8
    block_K = 32
    in_dtype = T.float16
    out_dtype = T.float16
    accum_dtype = T.float32
    metadata_dtype = T.int16
    e_factor = get_e_factor(in_dtype, metadata_dtype)

    @T.prim_func
    def main(
        A_sparse: T.Tensor((M, K // 2), in_dtype),
        E: T.Tensor((M, K // e_factor), metadata_dtype),
        B: T.Tensor((K, N), in_dtype),
        C: T.Tensor((M, N), out_dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=32) as (bx, by):
            A_shared = T.alloc_shared((block_M, block_K // 2), in_dtype)
            E_shared = T.alloc_shared((block_M, block_K // e_factor), metadata_dtype)
            B_shared = T.alloc_shared((block_K, block_N), in_dtype)
            C_frag = T.alloc_fragment((block_M, block_N), accum_dtype)

            T.clear(C_frag)
            T.copy(A_sparse[by * block_M, 0], A_shared)
            T.copy(E[by * block_M, 0], E_shared)
            T.copy(B[0, bx * block_N], B_shared)
            T.gemm_sp(A_shared, E_shared, B_shared, C_frag, False, False, False)
            T.copy(C_frag, C[by * block_M, bx * block_N])

    return main


def test_gemm_sp_lowers_to_musa_mma_sp_source():
    artifact = tilelang.lower(_sparse_matmul_source(), target="musa -arch=mp_22")
    source = artifact.kernel_source

    assert "mma.sp.sync.aligned.m16n8k32" in source
    assert "ptx_ldmatrix" in source
