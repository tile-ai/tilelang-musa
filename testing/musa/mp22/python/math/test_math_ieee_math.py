import tilelang
import tilelang.language as T
import tilelang.testing
import pytest
import torch
import struct


MP22_IEEE_ARITHMETIC_OPS = [
    ("ieee_add", T.ieee_add),
    ("ieee_sub", T.ieee_sub),
    ("ieee_mul", T.ieee_mul),
    ("ieee_fmaf", T.ieee_fmaf),
]



# Columns are A, B, [C], expected output bits; None denotes any quiet/signaling NaN.
IEEE_RN_CASES = {
    "ieee_add": [
        (0x3F800000, 0x33800000, 0x3F800000),  # halfway, even lower
        (0x3F800001, 0x33800000, 0x3F800002),  # halfway, even upper
        (0x00000001, 0x00000001, 0x00000002),
        (0x80000000, 0x80000000, 0x80000000),
        (0x3F800000, 0xBF800000, 0x00000000),
        (0x7F800000, 0xFF800000, None),
        (0x7FC00001, 0x3F800000, None),
        (0x7F7FFFFF, 0x7F7FFFFF, 0x7F800000),
    ],
    "ieee_sub": [
        (0x3F800001, 0x33800000, 0x3F800000),
        (0x00800000, 0x007FFFFF, 0x00000001),
        (0x80000000, 0x00000000, 0x80000000),
        (0x3F800000, 0x3F800000, 0x00000000),
        (0x7F800000, 0x7F800000, None),
        (0x7FC00001, 0x3F800000, None),
    ],
    "ieee_mul": [
        (0x3F800001, 0x3F7FFFFE, 0x3F800000),
        (0x00800000, 0x3F000000, 0x00400000),
        (0x00000001, 0x3F000000, 0x00000000),
        (0x80000000, 0x40000000, 0x80000000),
        (0x7F800000, 0x00000000, None),
        (0xFF800000, 0x40000000, 0xFF800000),
        (0x7FC00001, 0x3F800000, None),
    ],
    "ieee_fmaf": [
        (0x3F800001, 0x3F7FFFFE, 0xBF800000, 0xA8800000),
        (0x3F800000, 0x33800000, 0x3F800000, 0x3F800000),
        (0x00800000, 0x3F000000, 0x00000001, 0x00400001),
        (0x80000000, 0x40000000, 0x80000000, 0x80000000),
        (0x3F800000, 0x3F800000, 0xBF800000, 0x00000000),
        (0x7F800000, 0x00000000, 0x3F800000, None),
        (0x7FC00001, 0x3F800000, 0x3F800000, None),
    ],
}


def _float32_from_bits(bits):
    return struct.unpack("!f", struct.pack("!I", bits))[0]


def run_ieee_arithmetic_rn(
    mathop_name,
    mathop_func,
    M=32,
    N=32,
    block_M=16,
    block_N=16,
    dtype=T.float32,
):
    if mathop_name == "ieee_fmaf":

        @T.prim_func
        def main_func(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
            D: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                for i, j in T.Parallel(block_M, block_N):
                    D[by * block_M + i, bx * block_N + j] = mathop_func(
                        A[by * block_M + i, bx * block_N + j],
                        B[by * block_M + i, bx * block_N + j],
                        C[by * block_M + i, bx * block_N + j],
                        "rn",
                    )

        out_idx = [3]
        num_inputs = 3
    else:

        @T.prim_func
        def main_func(
            A: T.Tensor((M, N), dtype),
            B: T.Tensor((M, N), dtype),
            C: T.Tensor((M, N), dtype),
        ):
            with T.Kernel(T.ceildiv(N, block_N), T.ceildiv(M, block_M), threads=128) as (bx, by):
                for i, j in T.Parallel(block_M, block_N):
                    C[by * block_M + i, bx * block_N + j] = mathop_func(
                        A[by * block_M + i, bx * block_N + j],
                        B[by * block_M + i, bx * block_N + j],
                        "rn",
                    )

        out_idx = [2]
        num_inputs = 2

    kernel = tilelang.compile(
        main_func,
        out_idx=out_idx,
        target="musa",
        pass_configs={tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False},
    )

    # Exact binary32 fixtures, including rounding ties and exceptional values.
    # NaN payloads are unspecified; all other outputs must match every bit.
    cases = IEEE_RN_CASES[mathop_name]
    rows = [cases[i % len(cases)] for i in range(M * N)]
    inputs = []
    for column in range(num_inputs):
        values = [_float32_from_bits(row[column]) for row in rows]
        inputs.append(torch.tensor(values, dtype=torch.float32).reshape(M, N).to("musa"))
    result = kernel(*inputs).cpu().contiguous()
    assert result.dtype == torch.float32
    assert tuple(result.shape) == (M, N)
    actual = result.view(torch.int32).reshape(-1).tolist()
    for index, (bits, row) in enumerate(zip(actual, rows)):
        expected = row[-1]
        bits &= 0xFFFFFFFF
        if expected is None:
            assert bits & 0x7F800000 == 0x7F800000 and bits & 0x007FFFFF, (index, hex(bits))
        else:
            assert bits == expected, (mathop_name, index, hex(bits), hex(expected))


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
@pytest.mark.parametrize(
    "mathop_name,mathop_func",
    MP22_IEEE_ARITHMETIC_OPS,
    ids=[name for name, _ in MP22_IEEE_ARITHMETIC_OPS],
)
def test_mp22_ieee_arithmetic_rn(mathop_name, mathop_func):
    run_ieee_arithmetic_rn(mathop_name, mathop_func)


if __name__ == "__main__":
    tilelang.testing.main()
