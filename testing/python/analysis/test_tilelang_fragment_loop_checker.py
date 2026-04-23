import tilelang
import tilelang.testing
import tilelang.language as T
import pytest


@tilelang.jit
def simple_invalid_loop(A):
    with T.Kernel(1, threads=128) as _:
        data_frag = T.alloc_fragment([128], T.float32)
        for i in T.Parallel(A):
            data_frag[i] = 0


@tilelang.jit
def nested_invalid_loop(A):
    with T.Kernel(1, threads=128) as _:
        data_frag = T.alloc_fragment([128], T.float32)

        for i in T.Parallel(A // 64):
            for j in T.Parallel(64):
                data_frag[i * 64 + j] = 0


@tilelang.jit
def invalid_loop_with_complex_dataflow(A):
    with T.Kernel(1, threads=128) as _:
        data_frag = T.alloc_fragment([128], T.float32)

        for i in T.Parallel(A):
            data_frag[i // 64 + i % 64] = 0


@tilelang.jit
def invalid_fragment_load(A):
    with T.Kernel(1, threads=128) as _:
        data_frag = T.alloc_fragment([128], T.float32)
        data_shared = T.alloc_shared([128], T.float32)
        for i in T.serial(128):
            data_frag[i] = 0
        for i in T.Parallel(A):
            data_shared[i] = data_frag[i]


@tilelang.jit
def valid_loop_not_use_loop_var(A):
    with T.Kernel(1, threads=128) as _:
        data_frag = T.alloc_fragment([128], T.float32)

        for i in T.Parallel(A):  # noqa: B007
            for j in T.Parallel(64):
                data_frag[j] = 0  # This is valid because we don't use i


@tilelang.jit
def valid_loop_use_shared(A):
    with T.Kernel(1, threads=128) as _:
        data_shared = T.alloc_shared([128], T.float32)

        for i in T.Parallel(A):
            data_shared[i] = 0  # Valid because this is shared memory


@tilelang.jit
def valid_loop_use_local(A):
    with T.Kernel(1, threads=128) as _:
        data_local = T.alloc_local([128], T.float32)

        for i in T.Parallel(A):
            data_local[i] = 0  # Valid because this is local memory


@tilelang.jit
def valid_loop_serial(A):
    with T.Kernel(1, threads=128) as _:
        data_frag = T.alloc_fragment([128], T.float32)

        for i in T.serial(A):
            data_frag[i] = 0  # Valid because this is serial


def test_invalid_loop():
    for case in [
        simple_invalid_loop,
        nested_invalid_loop,
        invalid_loop_with_complex_dataflow,
        invalid_fragment_load,
    ]:
        with pytest.raises(ValueError, match="fragment buffer"):
            case.compile(A=T.dynamic("A"))


def test_valid_loop():
    for case in [
        valid_loop_not_use_loop_var,
        valid_loop_use_shared,
        valid_loop_use_local,
        valid_loop_serial,
    ]:
        case.compile(A=T.dynamic("A"))


if __name__ == "__main__":
    tilelang.testing.main()
