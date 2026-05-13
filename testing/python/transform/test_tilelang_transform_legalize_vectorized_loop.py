from tilelang import tvm as tvm
import tilelang as tl
import tilelang.language as T
import tilelang.testing


def vectorize_access_legalize(M: int = 64, N: int = 64):
    dtype = T.float32
    vec_len = 8

    @T.prim_func
    def main(
        A: T.Tensor((M, N, vec_len), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j in T.serial(N):
                for v in T.vectorized(vec_len):
                    A_shared[tid, j, v] = A[tid, j, v]

    @T.prim_func
    def expected(
        A: T.Tensor((M, N, vec_len), dtype=T.float32),
    ):
        with T.Kernel(1, 1, threads=M) as (bx, by):
            A_shared = T.alloc_shared((M, N, vec_len), dtype=dtype)
            tid = T.get_thread_binding()
            for j, v_2 in T.grid(M, vec_len // 4):
                for vec in T.vectorized(4):
                    A_shared[tid, j, v_2 * 4 + vec] = A[tid, j, v_2 * 4 + vec]

    return main, expected


def assert_vectorize_access(M: int = 64, N: int = 64):
    func, expected = vectorize_access_legalize(M, N)
    mod = tvm.IRModule({func.attrs["global_symbol"]: func})
    with tvm.target.Target("musa"):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)
    tvm.ir.assert_structural_equal(transformed["main"].body, expected.body)


def test_vectorize_access():
    assert_vectorize_access(64, 64)


def test_swizzled_tail_access_keeps_vectorization():
    @T.prim_func
    def main(K: T.handle):
        Kb = T.match_buffer(K, (1024, 1, 292), "bfloat16", strides=(292, 292, 1))
        S = T.alloc_shared((64, 256), "bfloat16")
        p = T.alloc_local((1,), "int32")
        ty = T.alloc_local((1,), "int32")
        kk = T.alloc_local((4,), "int32")
        p[0] = 0
        ty[0] = 0
        kk[0] = 0
        for r in T.unroll(4):
            for v in T.vectorized(8):
                S[
                    (p[0] * 8 + v + 192) // 128 * 32 + (r * 16 + ty[0]) // 2,
                    (r * 16 + ty[0]) % 2 * 128
                    + T.bitwise_xor(
                        (p[0] * 8 + v + 192) % 128 // 8,
                        (r * 16 + ty[0]) % 16,
                    )
                    * 8
                    + (p[0] * 8 + v + 192) % 8,
                ] = Kb[kk[r], 0, p[0] * 8 + v + 224]

    mod = tvm.IRModule.from_expr(main)
    with tvm.target.Target("musa -arch=mp_31"):
        transformed = tl.transform.LegalizeVectorizedLoop()(mod)

    script = transformed.script()
    assert "T.vectorized" in script
    assert "for v in range(8):" not in script


if __name__ == "__main__":
    tilelang.testing.main()
