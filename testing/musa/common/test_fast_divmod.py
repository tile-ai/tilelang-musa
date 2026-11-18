import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()

NUM_VALUES = 256
NUM_RESULTS = 6
EXECUTION_BACKENDS = ["tvm_ffi", "cython"]


def host_divisor_program(num_values):
    @T.prim_func
    def main(
        dividends: T.Tensor([num_values], T.int32),
        divisor: T.int32,
        output: T.Tensor([NUM_RESULTS, num_values], T.int32),
    ):
        with T.Kernel(1, threads=num_values):
            i = T.get_thread_binding()
            quotient, remainder = T.fast_divmod(dividends[i], divisor)
            output[0, i] = T.fast_div(dividends[i], divisor)
            output[1, i] = T.fast_mod(dividends[i], divisor)
            output[2, i] = quotient
            output[3, i] = remainder
            output[4, i] = dividends[i] // divisor
            output[5, i] = dividends[i] % divisor

    return main


def constant_divisor_program(num_values, divisor):
    @T.prim_func
    def main(
        dividends: T.Tensor([num_values], T.int32),
        output: T.Tensor([NUM_RESULTS, num_values], T.int32),
    ):
        with T.Kernel(1, threads=num_values):
            i = T.get_thread_binding()
            quotient, remainder = T.fast_divmod(dividends[i], T.int32(divisor))
            output[0, i] = T.fast_div(dividends[i], T.int32(divisor))
            output[1, i] = T.fast_mod(dividends[i], T.int32(divisor))
            output[2, i] = quotient
            output[3, i] = remainder
            output[4, i] = dividends[i] // T.int32(divisor)
            output[5, i] = dividends[i] % T.int32(divisor)

    return main


def device_divisor_program(num_values):
    @T.prim_func
    def main(
        dividends: T.Tensor([num_values], T.int32),
        divisors: T.Tensor([num_values], T.int32),
        output: T.Tensor([NUM_RESULTS, num_values], T.int32),
    ):
        with T.Kernel(1, threads=num_values):
            i = T.get_thread_binding()
            quotient, remainder = T.fast_divmod(dividends[i], divisors[i])
            output[0, i] = T.fast_div(dividends[i], divisors[i])
            output[1, i] = T.fast_mod(dividends[i], divisors[i])
            output[2, i] = quotient
            output[3, i] = remainder
            output[4, i] = dividends[i] // divisors[i]
            output[5, i] = dividends[i] % divisors[i]

    return main


def make_dividends():
    edge_values = torch.tensor(
        [
            0,
            1,
            2,
            3,
            7,
            15,
            16,
            17,
            31,
            32,
            33,
            127,
            255,
            256,
            257,
            65535,
            65536,
            1_000_003,
            1_073_741_823,
            1_073_741_824,
            2_147_483_646,
            2_147_483_647,
        ],
        dtype=torch.int32,
    )
    generator = torch.Generator().manual_seed(0)
    random_values = torch.randint(
        0,
        2**31,
        (NUM_VALUES - edge_values.numel(),),
        dtype=torch.int64,
        generator=generator,
    ).to(torch.int32)
    return torch.cat([edge_values, random_values])


def make_divisors():
    values = torch.tensor(
        [
            1,
            2,
            3,
            4,
            5,
            7,
            8,
            9,
            15,
            16,
            17,
            31,
            32,
            33,
            127,
            255,
            256,
            257,
            65535,
            65536,
            1_000_003,
            2_147_483_647,
        ],
        dtype=torch.int32,
    )
    repeats = (NUM_VALUES + values.numel() - 1) // values.numel()
    return values.repeat(repeats)[:NUM_VALUES]


def reference(dividends, divisors):
    quotient = torch.div(dividends, divisors, rounding_mode="floor")
    remainder = torch.remainder(dividends, divisors)
    return torch.stack([quotient, remainder, quotient, remainder, quotient, remainder])


def require_musa_device():
    if not hasattr(torch, "musa") or not torch.musa.is_available():
        pytest.skip("MUSA device is not available")


def run_and_check(kernel, arguments, expected):
    output = kernel(*arguments)
    if isinstance(output, (tuple, list)):
        output = output[0]
    torch.testing.assert_close(output.cpu(), expected, rtol=0.0, atol=0.0)


@tilelang.testing.requires_musa
@pytest.mark.parametrize("execution_backend", EXECUTION_BACKENDS)
def test_fast_divmod_host_divisor_numerical(execution_backend):
    require_musa_device()
    dividends = make_dividends()
    kernel = tilelang.compile(
        host_divisor_program(NUM_VALUES),
        out_idx=[2],
        target="musa",
        execution_backend=execution_backend,
    )

    for divisor in [1, 2, 3, 7, 16, 31, 65535, 1_000_003, 2_147_483_647]:
        expected = reference(dividends, divisor)
        run_and_check(kernel, (dividends.to("musa"), divisor), expected)


@tilelang.testing.requires_musa
@pytest.mark.parametrize("execution_backend", EXECUTION_BACKENDS)
def test_fast_divmod_constant_divisor_numerical(execution_backend):
    require_musa_device()
    divisor = 7
    dividends = make_dividends()
    kernel = tilelang.compile(
        constant_divisor_program(NUM_VALUES, divisor),
        out_idx=[1],
        target="musa",
        execution_backend=execution_backend,
    )

    run_and_check(kernel, (dividends.to("musa"),), reference(dividends, divisor))


@tilelang.testing.requires_musa
@pytest.mark.parametrize("execution_backend", EXECUTION_BACKENDS)
def test_fast_divmod_device_divisor_numerical(execution_backend):
    require_musa_device()
    dividends = make_dividends()
    divisors = make_divisors()
    kernel = tilelang.compile(
        device_divisor_program(NUM_VALUES),
        out_idx=[2],
        target="musa",
        execution_backend=execution_backend,
    )

    run_and_check(
        kernel,
        (dividends.to("musa"), divisors.to("musa")),
        reference(dividends, divisors),
    )
