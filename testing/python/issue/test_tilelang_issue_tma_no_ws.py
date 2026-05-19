import re

import tilelang
import tilelang.testing
from tilelang import language as T
import torch


def _compile_tvm_ffi(func, pass_configs, **kwargs):
    tilelang.disable_cache()
    try:
        return tilelang.compile(
            func,
            target="musa",
            execution_backend="tvm_ffi",
            pass_configs=pass_configs,
            **kwargs,
        )
    finally:
        tilelang.enable_cache()


def test_tma_lower_no_warp_specialized_injects_mbarrier():
    """Regression for TMA lowering when warp specialization is disabled.

    When `tl.disable_tma_lower=False` but `tl.disable_warp_specialized=True`, the
    optimization pipeline must still run the TMA barrier allocation/injection
    passes so generated MUSA source initializes and uses async named barrier
    correctly.
    """

    M, K = 16, 128
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def tma_copy(x: T.Tensor((M, K), T.float16)):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(K, block_k), threads=threads) as (
            pid_m,
            pid_k,
        ):
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            T.fill(x_shared, 0)
            T.copy(
                x[
                    pid_m * block_m : (pid_m + 1) * block_m,
                    pid_k * block_k : (pid_k + 1) * block_k,
                ],
                x_shared,
            )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }
    kernel = _compile_tvm_ffi(tma_copy, pass_configs)

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "__musa_async_bar_record(1)" in src
    assert "__musa_async_init_arrival(1" in src
    assert "tl::mbarrier_arrive_expect_tx(1" in src


def test_tma_lower_no_warp_specialized_2d_descriptor_uses_args1_barrier():
    """Cover the 2D-descriptor TMA barrier rewrite path (barrier at args[1])."""

    M, K = 16, 256
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def tma_copy_2d_desc(x: T.Tensor((M, K), T.float16)):
        with T.Kernel(T.ceildiv(M, block_m), T.ceildiv(K, block_k), threads=threads) as (
            pid_m,
            pid_k,
        ):
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            T.fill(x_shared, 0)
            T.copy(
                x[
                    pid_m * block_m : (pid_m + 1) * block_m,
                    pid_k * block_k : (pid_k + 1) * block_k,
                ],
                x_shared,
            )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
    }

    kernel = _compile_tvm_ffi(tma_copy_2d_desc, pass_configs)

    src = kernel.get_kernel_source()
    assert "MUtensorDescriptor" in src
    assert "tl::tma_load" in src

    flat_src = " ".join(src.split())
    pattern = r"tl::tma_load(?:<[^>]+>)?\([^,]+,\s*1\s*,"
    assert re.search(pattern, flat_src), (
        f"Expected regex {pattern!r} to match flattened MUSA source. Generated source (truncated):\n{src[:1000]}"
    )


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_num_stages_zero_pure_tma_does_not_auto_warp_specialize():
    """num_stages=0 should keep pure TMA loops out of auto-WS."""

    M, K = 8, 256
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def copy_loop_num_stages_zero(
        x: T.Tensor((M, K), T.float16),
        y: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=0):
                T.copy(
                    x[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                    x_shared,
                )
                T.copy(
                    x_shared,
                    y[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(copy_loop_num_stages_zero, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "__launch_bounds__(160, 1)" not in src
    assert "if (32 <= ((int)threadIdx.x))" not in src

    x = torch.randn((M, K), device="musa", dtype=torch.float16)
    y = kernel(x)
    torch.testing.assert_close(y, x)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_num_stages_one_pure_tma_keeps_auto_warp_specialize():
    """Pure TMA loops should auto-WS when num_stages is explicitly enabled."""

    M, K = 8, 256
    block_m, block_k = 4, 128
    threads = 32

    @T.prim_func
    def copy_loop_num_stages_one(
        x: T.Tensor((M, K), T.float16),
        y: T.Tensor((M, K), T.float16),
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=1):
                T.copy(
                    x[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                    x_shared,
                )
                T.copy(
                    x_shared,
                    y[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                )

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(copy_loop_num_stages_one, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "__launch_bounds__(160, 1)" in src
    assert "if (32 <= ((int)threadIdx.x))" in src
    x = torch.randn((M, K), device="musa", dtype=torch.float16)
    y = kernel(x)
    torch.testing.assert_close(y, x)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_num_stages_zero_cp_async_only_does_not_auto_warp_specialize():
    """num_stages=0 should keep cp.async-only loops out of auto-WS."""

    bytes_per_copy = 16
    threads = 32

    @T.prim_func
    def cp_async_only_num_stages_zero(
        x: T.Tensor((4 * bytes_per_copy,), T.uint8),
        y: T.Tensor((4 * bytes_per_copy,), T.uint8),
    ):
        with T.Kernel(1, threads=threads):
            x_shared = T.alloc_shared((bytes_per_copy,), dtype=T.uint8)
            for ko in T.Pipelined(4, num_stages=0):
                T.ptx_cp_async(
                    T.access_ptr(x_shared[0], "w", bytes_per_copy),
                    T.access_ptr(x[ko * bytes_per_copy], "r", bytes_per_copy),
                    bytes_per_copy,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                for i in T.serial(bytes_per_copy):
                    y[ko * bytes_per_copy + i] = x_shared[i]

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(cp_async_only_num_stages_zero, pass_configs, out_idx=[1])

    src = kernel.get_kernel_source()
    assert "cp_async_gs<16>" in src
    assert "__launch_bounds__(32, 1)" in src
    assert "__launch_bounds__(160, 1)" not in src
    assert "if (32 <= ((int)threadIdx.x))" not in src
    x = torch.randint(0, 256, (4 * bytes_per_copy,), device="musa", dtype=torch.uint8)
    y = kernel(x)
    torch.testing.assert_close(y, x)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_num_stages_one_mixed_tma_cp_async_keeps_auto_ws():
    """Mixed TMA+cp.async should keep both async paths valid on MUSA."""

    M, K = 8, 256
    block_m, block_k = 4, 128
    threads = 128
    cp_async_bytes = 16

    @T.prim_func
    def mixed_async_num_stages_one(
        x: T.Tensor((M, K), T.float16),
        meta: T.Tensor((2 * cp_async_bytes,), T.uint8),
        y: T.Tensor((M, K), T.float16),
        meta_out: T.Tensor((2 * cp_async_bytes,), T.uint8),
    ):
        with T.Kernel(T.ceildiv(M, block_m), threads=threads) as pid_m:
            x_shared = T.alloc_shared((block_m, block_k), dtype=T.float16)
            meta_shared = T.alloc_shared((cp_async_bytes,), dtype=T.uint8)

            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=1):
                T.copy(
                    x[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                    x_shared,
                )
                T.ptx_cp_async(
                    T.access_ptr(meta_shared[0], "w", cp_async_bytes),
                    T.access_ptr(meta[ko * cp_async_bytes], "r", cp_async_bytes),
                    cp_async_bytes,
                )
                T.ptx_commit_group()
                T.ptx_wait_group(0)
                T.copy(
                    x_shared,
                    y[
                        pid_m * block_m : (pid_m + 1) * block_m,
                        ko * block_k : (ko + 1) * block_k,
                    ],
                )
                for i in T.serial(cp_async_bytes):
                    meta_out[ko * cp_async_bytes + i] = meta_shared[i]

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(mixed_async_num_stages_one, pass_configs, out_idx=[2, 3])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "cp_async_gs<16>" in src
    assert "__launch_bounds__(256, 1)" in src
    assert "__launch_bounds__(160, 1)" not in src
    producer_idx = src.index("if (128 <= ((int)threadIdx.x)) {")
    consumer_idx = src.index("} else {", producer_idx)
    cp_async_idx = src.index("cp_async_gs<16>")
    tma_idx = src.index("tl::tma_load")
    assert producer_idx < cp_async_idx < consumer_idx
    assert producer_idx < tma_idx < consumer_idx


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_mixed_tma_cp_async_shared_stage_barriers():
    """Mixed TMA+cp.async groups should share one MUSA producer barrier path."""

    M = N = K = 256
    block_m = block_n = 128
    block_k = 32
    num_stages = 3
    threads = 128

    @T.prim_func
    def mixed_gemm_shared_barrier(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            C_local = T.alloc_fragment((block_m, block_n), T.float32)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[by * block_m, ko * block_k], A_shared)
                for k, j in T.Parallel(block_k, block_n):
                    B_shared[k, j] = B[ko * block_k + k, bx * block_n + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_m, bx * block_n])

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(mixed_gemm_shared_barrier, pass_configs, out_idx=[2])

    src = kernel.get_kernel_source()
    assert "tl::tma_load" in src
    assert "cp_async_gs<16>" in src
    assert "__launch_bounds__(256, 1)" in src
    assert "__musa_async_bar_record(8)" in src
    assert "mbarrier_cp_async_arrive_noinc" not in src
    assert "expect_transaction" not in src
    for barrier_id in range(1, 9):
        assert f"__musa_async_init_arrival({barrier_id}," in src

    producer_idx = src.index("if (128 <= ((int)threadIdx.x)) {")
    consumer_idx = src.index("} else {", producer_idx)
    cp_async_idx = src.index("cp_async_gs<16>")
    tma_idx = src.index("tl::tma_load")
    assert producer_idx < cp_async_idx < consumer_idx
    assert producer_idx < tma_idx < consumer_idx
    assert "cp_async_gs<16>" not in src[consumer_idx:]
    assert "tl::tma_load" not in src[consumer_idx:]

    a = torch.randn((M, K), device="musa", dtype=torch.float16)
    b = torch.randn((K, N), device="musa", dtype=torch.float16)
    ref = a @ b
    c = kernel(a, b)
    torch.testing.assert_close(c, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_sparse_ws_regular_metadata_copy_stays_in_producer():
    """Metadata global->shared copy stays in producer after WS split on MUSA."""

    M = N = 128
    K = 256
    block_m = block_n = 128
    block_k = 128
    num_stages = 2
    threads = 128

    @T.prim_func
    def metadata_copy_pipeline(
        A: T.Tensor((M, K), T.float16),
        E: T.Tensor((M, K // 8), T.uint8),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
        MetaOut: T.Tensor((M, K // 8), T.uint8),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            E_shared = T.alloc_shared((block_m, block_k // 8), T.uint8)
            C_local = T.alloc_fragment((block_m, block_n), T.float32)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(E[by * block_m, ko * block_k // 8], E_shared)
                T.copy(A[by * block_m, ko * block_k], A_shared)
                for k, j in T.Parallel(block_k, block_n):
                    B_shared[k, j] = B[ko * block_k + k, bx * block_n + j]
                T.gemm(A_shared, B_shared, C_local)
                T.copy(E_shared, MetaOut[by * block_m, ko * block_k // 8])

            T.copy(C_local, C[by * block_m, bx * block_n])

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(metadata_copy_pipeline, pass_configs, out_idx=[3, 4])

    src = kernel.get_kernel_source()
    producer_idx = src.index("if (128 <= ((int)threadIdx.x)) {")
    consumer_idx = src.index("} else {", producer_idx)
    metadata_tma_idx = src.index("tl::tma_load<SmemSwizzleGranularity::B16>(E_desc_0")
    compute_tma_idx = src.index("tl::tma_load<SmemSwizzleGranularity::B16>(A_desc_1")
    metadata_store_idx = src.index("tl::tma_store<SmemSwizzleGranularity::B16>(MetaOut_desc_2")

    assert producer_idx < metadata_tma_idx < consumer_idx
    assert producer_idx < compute_tma_idx < consumer_idx
    assert metadata_store_idx > consumer_idx
    assert "E_desc_0" not in src[consumer_idx:]


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_pure_tma_consumer_local_init_does_not_leak_into_producer():
    """Consumer local init should stay out of producer in MUSA WS split."""

    M = N = K = 256
    block_m = block_n = 128
    block_k = 32
    num_stages = 3
    threads = 128

    @T.prim_func
    def mixed_gemm_consumer_local_init(
        A: T.Tensor((M, K), T.float16),
        B: T.Tensor((K, N), T.float16),
        C: T.Tensor((M, N), T.float16),
    ):
        with T.Kernel(T.ceildiv(N, block_n), T.ceildiv(M, block_m), threads=threads) as (bx, by):
            A_shared = T.alloc_shared((block_m, block_k), T.float16)
            B_shared = T.alloc_shared((block_k, block_n), T.float16)
            C_local = T.alloc_fragment((block_m, block_n), T.float32)

            T.clear(C_local)
            for ko in T.Pipelined(T.ceildiv(K, block_k), num_stages=num_stages):
                T.copy(A[by * block_m, ko * block_k], A_shared)
                for k, j in T.Parallel(block_k, block_n):
                    B_shared[k, j] = B[ko * block_k + k, bx * block_n + j]
                T.gemm(A_shared, B_shared, C_local)

            T.copy(C_local, C[by * block_m, bx * block_n])

    pass_configs = {
        tilelang.PassConfigKey.TL_ENABLE_FAST_MATH: False,
        tilelang.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    }
    kernel = _compile_tvm_ffi(mixed_gemm_consumer_local_init, pass_configs, out_idx=[2])

    src = kernel.get_kernel_source()
    producer_idx = src.index("if (128 <= ((int)threadIdx.x)) {")
    consumer_idx = src.index("} else {", producer_idx)
    prelude_src = src[:producer_idx]
    producer_src = src[producer_idx:consumer_idx]
    consumer_src = src[consumer_idx:]

    init_pattern = r"\*\(float4\*\)\(C_local \+ \(i_\d+ \* 4\)\) = make_float4"
    assert re.search(init_pattern, consumer_src)
    assert not re.search(init_pattern, prelude_src)
    assert not re.search(init_pattern, producer_src)


if __name__ == "__main__":
    tilelang.testing.main()
