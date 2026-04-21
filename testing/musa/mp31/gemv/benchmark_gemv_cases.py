"""TileLang MUSA GEMV/small-M GEMM device-time benchmark.

The benchmark targets decode-style shapes such as ``A[M, K] @ B[N, K].T`` for
small M. Timing uses torch profiler device events and only counts TileLang
``main_kernel`` time. Python dispatch, TileLang host wrapper time, explicit
synchronization, and optional cache-flush kernels are excluded.
"""

import argparse
import csv
from dataclasses import dataclass
from typing import Callable

import torch
import tilelang as tl
import tilelang.language as T
from tvm import DataType

from tilelang.tileop.base import GemmWarpPolicy
from tilelang.utils.device import synchronize


M1_CASES = [
    (1, 6144, 4096),
    (1, 4096, 4096),
    (1, 24576, 4096),
    (1, 4096, 12288),
    (1, 1280, 2048),
    (1, 2048, 1024),
    (1, 128, 2048),
]

SMALL_M_CASES = [
    (5, 6144, 4096),
    (5, 4096, 4096),
    (5, 24576, 4096),
]

# Keep all tuning knobs in data tables. BM16 TME configs explicitly separate
# A/B cache hints: A is small and reused across output tiles, while B is mostly
# streaming through each CTA.
M1_SIMT_CONFIGS = [
    ("m1_128", 2, 64),
    ("m1_1280", 8, 32),
    ("m1_default", 8, 32),
    ("m1_4096", 4, 16),
]

M1_BM16_PC_CONFIGS = [
    (
        "m1_bm16_default",
        16,
        64,
        128,
        256,
        2,
        "default",
        "cache_persist",
        "cache_persist",
        "cache_once",
        "cache_none",
    ),
    (
        "m1_bm16_bn128",
        16,
        128,
        128,
        256,
        2,
        "default",
        "cache_persist",
        "cache_persist",
        "cache_once",
        "cache_none",
    ),
]

M5_BM16_PC_CONFIGS = [
    (
        "m5_default",
        16,
        64,
        128,
        256,
        2,
        "default",
        "cache_persist",
        "cache_persist",
        "cache_once",
        "cache_none",
    ),
    (
        "m5_n4096",
        16,
        128,
        64,
        256,
        3,
        "default",
        "cache_persist",
        "cache_persist",
        "cache_once",
        "cache_none",
    ),
    (
        "m5_large_n",
        16,
        64,
        128,
        256,
        2,
        "m",
        "cache_persist",
        "cache_persist",
        "cache_once",
        "cache_none",
    ),
]

PASS_CONFIGS_TME_WS = {
    tl.PassConfigKey.TL_ENABLE_MUSA_TMA_PREFETCH: True,
    tl.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: False,
    tl.PassConfigKey.TL_DISABLE_TMA_LOWER: False,
    tl.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
}


@dataclass(frozen=True)
class BenchResult:
    kind: str
    M: int
    N: int
    K: int
    config: str
    ms: float
    tflops: float
    gbps: float
    bytes_rw: int
    correct: bool


Config = tuple[str, int, int, int, int, int, str, str, str, str, str]

# Only used by the fallback selector score. Common MP31 deployments expose
# 56/60/64 MPs, so use a representative estimate rather than a hardware claim.
SELECTOR_ESTIMATED_MP_COUNT = 64


def _ceil_div_int(a: int, b: int) -> int:
    return (a + b - 1) // b


def _score_bm16_config(M: int, N: int, K: int, config: Config, nr_mp: int = SELECTOR_ESTIMATED_MP_COUNT) -> float:
    """muDNN-style coarse score for unknown small-M shapes.

    This intentionally mirrors the high-level terms in muDNN's local
    ``scripts/gemm_choose.cc`` without trying to clone private constants: expose
    enough tiles to fill waves, penalize large epilogues on short reductions,
    and favor arithmetic intensity when two candidates have similar occupancy.
    """
    _, block_M, block_N, block_K, _, _, policy, _, _, _, _ = config
    m_tiles = _ceil_div_int(max(M, 1), block_M)
    n_tiles = _ceil_div_int(max(N, 1), block_N)
    total_tiles = max(1, m_tiles * n_tiles)
    waves = _ceil_div_int(total_tiles, nr_mp)
    wave_ratio = total_tiles / float(waves * nr_mp)

    dtype_size = 2.0
    arithmetic_intensity = (2.0 * M * N * K) / max(
        dtype_size * (M * K + N * K + M * N),
        1.0,
    )
    # MP31 bf16 GEMV is bandwidth-biased in these shapes. Use a soft cap so the
    # score still differentiates very small-output and very large-output cases.
    intensity_ratio = min(arithmetic_intensity / 96.0, 1.0)
    epilogue_ratio = 1.0 - 0.5 * block_K / float(max(K, block_K))
    standalone_ratio = 0.95 if (m_tiles == 1 or n_tiles == 1) else 1.0
    policy_ratio = 1.02 if policy == "m" and N >= 8192 else 1.0
    return wave_ratio * (0.35 + 0.65 * intensity_ratio) * epilogue_ratio * standalone_ratio * policy_ratio


def _pick_scored_configs(M: int, N: int, K: int, configs: list[Config], limit: int = 1) -> list[Config]:
    return [
        config
        for _, config in sorted(
            ((_score_bm16_config(M, N, K, config), config) for config in configs),
            key=lambda item: item[0],
            reverse=True,
        )[:limit]
    ]


def _make_cache_flush_buffers(flush_cache_mb: int):
    if flush_cache_mb <= 0:
        return None
    num_bytes = flush_cache_mb * 1024 * 1024
    num_elems = max(1, num_bytes // 4)
    src = torch.arange(num_elems, device="musa", dtype=torch.float32)
    dst = torch.empty_like(src)
    return src, dst


def _flush_cache(buffers) -> None:
    if buffers is None:
        return
    src, dst = buffers
    dst.copy_(src)
    synchronize()


def _dtype_bytes(dtype: str) -> int:
    return DataType(dtype).bits // 8


def _gemv_bytes(M: int, N: int, K: int, dtype: str) -> int:
    itemsize = _dtype_bytes(dtype)
    return (M * K + N * K + M * N) * itemsize


def _tflops(M: int, N: int, K: int, ms: float) -> float:
    return (2.0 * M * N * K) / (ms * 1e-3) / 1e12


def _gbps(bytes_rw: int, ms: float) -> float:
    return bytes_rw / (ms * 1e-3) / 1e9


def _measure_tilelang_kernel_kineto(fn: Callable[[], object], rep: int, flush_buffers=None) -> float:
    # Warm once outside the profiler so only steady-state main_kernel events are collected.
    if flush_buffers is not None:
        _flush_cache(flush_buffers)
    fn()
    synchronize()

    schedule = torch.profiler.schedule(wait=1, warmup=1, active=rep, repeat=1)
    profiler = torch.profiler.profile(
        activities=[torch.profiler.ProfilerActivity.MUSA],
        schedule=schedule,
        record_shapes=False,
    )
    with profiler:
        for _ in range(rep + 2):
            if flush_buffers is not None:
                _flush_cache(flush_buffers)
            fn()
            synchronize()
            profiler.step()

    kernel_events = [evt for evt in profiler.events() if evt.name == "main_kernel"]
    if not kernel_events:
        raise RuntimeError("No TileLang main_kernel event found in profiler output")
    return sum(evt.device_time for evt in kernel_events) / len(kernel_events) / 1000.0


def _check_close(out: torch.Tensor, ref: torch.Tensor) -> bool:
    try:
        torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)
        return True
    except AssertionError as exc:
        print(f"[warn] correctness mismatch: {str(exc).splitlines()[0]}", flush=True)
        return False


@tl.jit(out_idx=[-1], target="musa")
def gemv_m1_splitk(
    N: int,
    K: int,
    block_N: int,
    reduce_threads: int,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
    b_layout: str = "contiguous_transb",
):
    tile_k = 128 // DataType(dtype).bits
    block_k = reduce_threads * tile_k
    if b_layout == "contiguous_transb":
        b_type = T.Tensor((N, K), dtype)
    elif b_layout == "strided_transb":
        # Original torch weight is contiguous as W[K, N]. W.T has shape [N, K]
        # with element strides (1, N), avoiding an out-of-timing contiguous copy.
        b_type = T.StridedTensor((N, K), (1, N), dtype)
    else:
        raise ValueError(f"Unsupported B layout: {b_layout}")

    @T.prim_func
    def main(
        A: T.Tensor((K,), dtype),
        B: b_type,
        C: T.Tensor((N,), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=(block_N, reduce_threads)) as bn:
            tn = T.get_thread_binding(0)
            tk = T.get_thread_binding(1)
            a_local = T.alloc_local((tile_k,), dtype)
            b_local = T.alloc_local((tile_k,), dtype)
            accum = T.alloc_local((1,), accum_dtype)
            reduced = T.alloc_local((1,), accum_dtype)

            T.clear(accum)
            for bk in T.serial(T.ceildiv(K, block_k)):
                for kk in T.vectorized(tile_k):
                    k_idx = bk * block_k + tk * tile_k + kk
                    n_idx = bn * block_N + tn
                    if k_idx < K and n_idx < N:
                        a_local[kk] = A[k_idx]
                        b_local[kk] = B[n_idx, k_idx]
                    else:
                        a_local[kk] = T.cast(0, dtype)
                        b_local[kk] = T.cast(0, dtype)
                for kk in T.serial(tile_k):
                    accum[0] += a_local[kk].astype(accum_dtype) * b_local[kk].astype(accum_dtype)

            with T.attr(
                T.comm_reducer(lambda x, y: x + y, [T.cast(0, accum_dtype)]),
                "reduce_scope",
                T.reinterpret(T.uint64(0), dtype="handle"),
            ):
                T.evaluate(
                    T.tvm_thread_allreduce(
                        T.uint32(1),
                        accum[0],
                        True,
                        reduced[0],
                        tk,
                        dtype="handle",
                    )
                )
            if bn * block_N + tn < N:
                C[bn * block_N + tn] = reduced[0]

    return main


@tl.jit(out_idx=[-1], target="musa", pass_configs=PASS_CONFIGS_TME_WS)
def small_m_bm16_sqmma_tme_pc_transb(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    policy: str,
    a_inner_cache_policy: str,
    a_outer_cache_policy: str,
    b_inner_cache_policy: str,
    b_outer_cache_policy: str,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
    b_layout: str = "contiguous_transb",
):
    if policy == "default":
        warp_policy = None
    elif policy == "m":
        warp_policy = GemmWarpPolicy.FullRow
    elif policy == "n":
        warp_policy = GemmWarpPolicy.FullCol
    elif policy == "square":
        warp_policy = GemmWarpPolicy.Square
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    if b_layout == "contiguous_transb":
        b_type = T.Tensor((N, K), dtype)
    elif b_layout == "strided_transb":
        # B is W.T where original W is contiguous [K, N], so B strides are (1, N).
        b_type = T.StridedTensor((N, K), (1, N), dtype)
    else:
        raise ValueError(f"Unsupported B layout: {b_layout}")

    @T.prim_func
    def main(
        A: T.Tensor((block_M, K), dtype),
        B: b_type,
        C: T.Tensor((block_M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            a_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            b_shared = T.alloc_shared((num_stages, block_N, block_K), dtype)
            c_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            mbars = T.alloc_barrier([128, 128] * num_stages)

            with T.ws(0):
                T.clear(c_local)

            for ko in T.serial(T.ceildiv(K, block_K)):
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[ko % num_stages + num_stages],
                        parity=((ko // num_stages) % 2) ^ 1,
                    )
                    T.tma_copy(
                        A[0:block_M, ko * block_K : (ko + 1) * block_K],
                        a_shared[ko % num_stages, :, :],
                        barrier=mbars[ko % num_stages],
                        inner_cache_policy=a_inner_cache_policy,
                        outer_cache_policy=a_outer_cache_policy,
                    )
                    T.tma_copy(
                        B[bx * block_N : (bx + 1) * block_N, ko * block_K : (ko + 1) * block_K],
                        b_shared[ko % num_stages, :, :],
                        barrier=mbars[ko % num_stages],
                        inner_cache_policy=b_inner_cache_policy,
                        outer_cache_policy=b_outer_cache_policy,
                    )
                    T.mbarrier_arrive(mbarrier=mbars[ko % num_stages])
                with T.ws(0):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[ko % num_stages],
                        parity=(ko // num_stages) % 2,
                    )
                    if warp_policy is None:
                        T.gemm(
                            a_shared[ko % num_stages, :, :],
                            b_shared[ko % num_stages, :, :],
                            c_local,
                            transpose_B=True,
                        )
                    else:
                        T.gemm(
                            a_shared[ko % num_stages, :, :],
                            b_shared[ko % num_stages, :, :],
                            c_local,
                            transpose_B=True,
                            policy=warp_policy,
                        )
                    T.mbarrier_arrive(mbarrier=mbars[ko % num_stages + num_stages])

            with T.ws(0):
                for mi, ni in T.Parallel(block_M, block_N):
                    n_idx = bx * block_N + ni
                    if n_idx < N:
                        C[mi, n_idx] = c_local[mi, ni]

    return main


@tl.jit(out_idx=[-1], target="musa", pass_configs=PASS_CONFIGS_TME_WS)
def small_m_bm16_sqmma_tme_pc_w_kn(
    M: int,
    N: int,
    K: int,
    block_M: int,
    block_N: int,
    block_K: int,
    threads: int,
    num_stages: int,
    policy: str,
    a_inner_cache_policy: str,
    a_outer_cache_policy: str,
    b_inner_cache_policy: str,
    b_outer_cache_policy: str,
    dtype: T.dtype = T.float16,
    accum_dtype: T.dtype = T.float32,
):
    if policy == "default":
        warp_policy = None
    elif policy == "m":
        warp_policy = GemmWarpPolicy.FullRow
    elif policy == "n":
        warp_policy = GemmWarpPolicy.FullCol
    elif policy == "square":
        warp_policy = GemmWarpPolicy.Square
    else:
        raise ValueError(f"Unsupported policy: {policy}")

    @T.prim_func
    def main(
        A: T.Tensor((block_M, K), dtype),
        W: T.Tensor((K, N), dtype),
        C: T.Tensor((block_M, N), dtype),
    ):
        with T.Kernel(T.ceildiv(N, block_N), threads=threads) as bx:
            a_shared = T.alloc_shared((num_stages, block_M, block_K), dtype)
            w_shared = T.alloc_shared((num_stages, block_K, block_N), dtype)
            c_local = T.alloc_fragment((block_M, block_N), accum_dtype)
            mbars = T.alloc_barrier([128, 128] * num_stages)

            with T.ws(0):
                T.clear(c_local)

            for ko in T.serial(T.ceildiv(K, block_K)):
                with T.ws(1):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[ko % num_stages + num_stages],
                        parity=((ko // num_stages) % 2) ^ 1,
                    )
                    T.tma_copy(
                        A[0:block_M, ko * block_K : (ko + 1) * block_K],
                        a_shared[ko % num_stages, :, :],
                        barrier=mbars[ko % num_stages],
                        inner_cache_policy=a_inner_cache_policy,
                        outer_cache_policy=a_outer_cache_policy,
                    )
                    T.tma_copy(
                        W[ko * block_K : (ko + 1) * block_K, bx * block_N : (bx + 1) * block_N],
                        w_shared[ko % num_stages, :, :],
                        barrier=mbars[ko % num_stages],
                        inner_cache_policy=b_inner_cache_policy,
                        outer_cache_policy=b_outer_cache_policy,
                    )
                    T.mbarrier_arrive(mbarrier=mbars[ko % num_stages])
                with T.ws(0):
                    T.mbarrier_wait_parity(
                        mbarrier=mbars[ko % num_stages],
                        parity=(ko // num_stages) % 2,
                    )
                    if warp_policy is None:
                        T.gemm(a_shared[ko % num_stages, :, :], w_shared[ko % num_stages, :, :], c_local)
                    else:
                        T.gemm(
                            a_shared[ko % num_stages, :, :],
                            w_shared[ko % num_stages, :, :],
                            c_local,
                            policy=warp_policy,
                        )
                    T.mbarrier_arrive(mbarrier=mbars[ko % num_stages + num_stages])

            with T.ws(0):
                for mi, ni in T.Parallel(block_M, block_N):
                    n_idx = bx * block_N + ni
                    if n_idx < N:
                        C[mi, n_idx] = c_local[mi, ni]

    return main


def _format_bm16_config(config: Config) -> str:
    (
        name,
        block_M,
        block_N,
        block_K,
        threads,
        num_stages,
        policy,
        a_inner,
        a_outer,
        b_inner,
        b_outer,
    ) = config
    return (
        f"name={name},block_M={block_M},block_N={block_N},block_K={block_K},"
        f"threads={threads},stages={num_stages},policy={policy},"
        f"a_cache=({a_inner},{a_outer}),b_cache=({b_inner},{b_outer})"
    )


def _select_m1_simt_configs(N: int, K: int, mode: str) -> list[tuple[str, int, int]]:
    # Arguments use TileLang's internal convention: N is output columns and K is
    # reduction. bench.py passes original Shape(M, reduce, output) as (M, K, N).
    if mode == "all":
        return M1_SIMT_CONFIGS
    if mode == "simt":
        if N <= 1024 and K >= 1024:
            return [("m1_small_out", 2, 64)]
        if N == 1280 and K == 2048:
            return [("m1_1280", 8, 32)]
        if N == 4096 and K == 4096:
            return [("m1_4096", 4, 16)]
        return [("m1_default", 8, 32)]
    if mode == "auto" and not _select_m1_bm16_configs(N, K, "auto"):
        return _select_m1_simt_configs(N, K, "simt")
    return []


def _select_m1_bm16_configs(N: int, K: int, mode: str) -> list[Config]:
    if mode == "all":
        return M1_BM16_PC_CONFIGS
    if mode == "sqmma_pc" or mode == "auto":
        if mode == "auto" and N == 1024 and K == 2048:
            return []
        if mode == "auto" and N < 2048:
            return []
        # BN128 wins when the output tile count is modest or exactly covers the
        # large-output/short-reduce image-projection shape. BN64 remains better
        # for the streaming-B shapes with larger reduction.
        if (N == 4096 and K == 4096) or (N >= 12288 and K == 4096):
            return [M1_BM16_PC_CONFIGS[1]]
        return [M1_BM16_PC_CONFIGS[0]]
    return []


def _select_m5_bm16_configs(N: int, K: int, mode: str, M: int = 5) -> list[Config]:
    if mode in ("all", "pc"):
        return M5_BM16_PC_CONFIGS
    if mode == "auto":
        # Preserve measured hot-shape choices, then use a muDNN-inspired score
        # for shapes not covered by the screenshot benchmark. Arguments use the
        # internal convention: N is output columns and K is reduction.
        if N == 4096 and K == 4096:
            return [M5_BM16_PC_CONFIGS[1]]
        if N == 12288 and K == 4096:
            return [M5_BM16_PC_CONFIGS[1]]
        if N == 4096 and K >= 24576:
            return [M5_BM16_PC_CONFIGS[2]]
        return _pick_scored_configs(M, N, K, M5_BM16_PC_CONFIGS)
    raise ValueError(f"Unsupported small-M mode: {mode}")


def _bench_m1_case(
    M: int,
    N: int,
    K: int,
    dtype: str,
    rep: int,
    check: bool,
    flush_buffers=None,
    mode: str = "auto",
) -> list[BenchResult]:
    assert M == 1
    torch_dtype = T.dtype(dtype).as_torch()
    a = torch.randn(K, device="musa", dtype=torch_dtype)
    b = torch.randn(N, K, device="musa", dtype=torch_dtype)
    ref = None
    if check:
        ref = a @ b.T
        synchronize()

    bytes_rw = _gemv_bytes(M, N, K, dtype)
    results: list[BenchResult] = []

    for name, block_N, reduce_threads in _select_m1_simt_configs(N, K, mode):
        kernel = gemv_m1_splitk(N, K, block_N, reduce_threads, dtype=dtype)
        out = kernel(a, b)
        synchronize()
        correct = True if ref is None else _check_close(out, ref)
        ms = _measure_tilelang_kernel_kineto(lambda kernel=kernel: kernel(a, b), rep=rep, flush_buffers=flush_buffers)
        results.append(
            BenchResult(
                kind="m1_splitk",
                M=M,
                N=N,
                K=K,
                config=f"name={name},BLOCK_N={block_N},reduce_threads={reduce_threads}",
                ms=ms,
                tflops=_tflops(M, N, K, ms),
                gbps=_gbps(bytes_rw, ms),
                bytes_rw=bytes_rw,
                correct=correct,
            )
        )

    bm16_configs = _select_m1_bm16_configs(N, K, mode)
    if bm16_configs:
        a_pad16 = torch.zeros(16, K, device="musa", dtype=torch_dtype)
        a_pad16[0, :] = a
        synchronize()
        for config in bm16_configs:
            (
                _,
                block_M,
                block_N,
                block_K,
                threads,
                num_stages,
                policy,
                a_inner,
                a_outer,
                b_inner,
                b_outer,
            ) = config
            kernel = small_m_bm16_sqmma_tme_pc_transb(
                M,
                N,
                K,
                block_M,
                block_N,
                block_K,
                threads,
                num_stages,
                policy,
                a_inner,
                a_outer,
                b_inner,
                b_outer,
                dtype=dtype,
            )
            out = kernel(a_pad16, b)
            synchronize()
            correct = True if ref is None else _check_close(out[0, :], ref)
            ms = _measure_tilelang_kernel_kineto(lambda kernel=kernel: kernel(a_pad16, b), rep=rep, flush_buffers=flush_buffers)
            results.append(
                BenchResult(
                    kind="m1_bm16_sqmma_tme_pc",
                    M=M,
                    N=N,
                    K=K,
                    config=_format_bm16_config(config),
                    ms=ms,
                    tflops=_tflops(M, N, K, ms),
                    gbps=_gbps(bytes_rw, ms),
                    bytes_rw=bytes_rw,
                    correct=correct,
                )
            )
    return results


def _bench_small_m_case(
    M: int,
    N: int,
    K: int,
    dtype: str,
    rep: int,
    check: bool,
    flush_buffers=None,
    mode: str = "auto",
) -> list[BenchResult]:
    torch_dtype = T.dtype(dtype).as_torch()
    a = torch.randn(M, K, device="musa", dtype=torch_dtype)
    b = torch.randn(N, K, device="musa", dtype=torch_dtype)
    a_pad16 = torch.zeros(16, K, device="musa", dtype=torch_dtype)
    a_pad16[:M, :] = a
    ref = None
    if check:
        ref = a @ b.T
        synchronize()

    bytes_rw = _gemv_bytes(M, N, K, dtype)
    results: list[BenchResult] = []
    for config in _select_m5_bm16_configs(N, K, mode, M=M):
        (
            _,
            block_M,
            block_N,
            block_K,
            threads,
            num_stages,
            policy,
            a_inner,
            a_outer,
            b_inner,
            b_outer,
        ) = config
        kernel = small_m_bm16_sqmma_tme_pc_transb(
            M,
            N,
            K,
            block_M,
            block_N,
            block_K,
            threads,
            num_stages,
            policy,
            a_inner,
            a_outer,
            b_inner,
            b_outer,
            dtype=dtype,
        )
        out = kernel(a_pad16, b)
        synchronize()
        correct = True if ref is None else _check_close(out[:M, :], ref)
        ms = _measure_tilelang_kernel_kineto(lambda kernel=kernel: kernel(a_pad16, b), rep=rep, flush_buffers=flush_buffers)
        results.append(
            BenchResult(
                kind="bm16_sqmma_tme_pc",
                M=M,
                N=N,
                K=K,
                config=_format_bm16_config(config),
                ms=ms,
                tflops=_tflops(M, N, K, ms),
                gbps=_gbps(bytes_rw, ms),
                bytes_rw=bytes_rw,
                correct=correct,
            )
        )
    return results


def _select_cases(kind: str, case: str = ""):
    if case:
        parts = [int(x.strip()) for x in case.split(",")]
        if len(parts) != 3:
            raise ValueError("--case must be M,N,K")
        return [tuple(parts)]
    if kind == "m1":
        return M1_CASES
    if kind == "small_m":
        return SMALL_M_CASES
    return M1_CASES + SMALL_M_CASES


def _write_csv(path: str, results: list[BenchResult]) -> None:
    if not path:
        return
    with open(path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow(BenchResult.__dataclass_fields__.keys())
        for result in results:
            writer.writerow([getattr(result, field) for field in BenchResult.__dataclass_fields__.keys()])


def _format_result(result: BenchResult) -> str:
    return (
        f"{result.kind},{result.M},{result.N},{result.K},{result.config},"
        f"{result.ms:.6f},{result.tflops:.3f},{result.gbps:.2f},{result.bytes_rw},{result.correct}"
    )


def _print_result(result: BenchResult) -> None:
    print(_format_result(result), flush=True)


def _print_best(label: str, result: BenchResult) -> None:
    print(f"{label},{_format_result(result)}", flush=True)


def main() -> None:
    parser = argparse.ArgumentParser(description="TileLang MUSA GEMV device-time benchmark")
    parser.add_argument("--kind", choices=["m1", "small_m", "all"], default="all")
    parser.add_argument("--dtype", default="float16")
    parser.add_argument("--rep", type=int, default=20, help="Kineto active repetitions per measurement")
    parser.add_argument("--check", action="store_true", help="Compare each kernel against torch reference before timing")
    parser.add_argument("--csv", default="", help="Optional output CSV path")
    parser.add_argument(
        "--flush-cache-mb",
        type=int,
        default=0,
        help="Flush this many MiB before each profiled kernel launch; flush copy is not included in reported time",
    )
    parser.add_argument("--case", default="", help="Optional single case as M,N,K, for example 5,6144,4096")
    parser.add_argument(
        "--m1-mode",
        choices=["auto", "simt", "sqmma_pc", "all"],
        default="auto",
        help="Select M=1 implementation: auto dispatch, SIMT split-K, BM16 SQMMA/TME PC, or all configs.",
    )
    parser.add_argument(
        "--small-m-mode",
        choices=["auto", "pc", "all"],
        default="auto",
        help="Select M>1 implementation configs. auto runs the selected config; pc/all sweep BM16 PC configs.",
    )
    args = parser.parse_args()

    flush_buffers = _make_cache_flush_buffers(args.flush_cache_mb)
    results: list[BenchResult] = []
    print("kind,M,N,K,config,device_ms,TFLOPS,GBps,bytes_rw,correct")
    for M, N, K in _select_cases(args.kind, args.case):
        if M == 1:
            case_results = _bench_m1_case(
                M,
                N,
                K,
                args.dtype,
                args.rep,
                args.check,
                flush_buffers=flush_buffers,
                mode=args.m1_mode,
            )
        else:
            case_results = _bench_small_m_case(
                M,
                N,
                K,
                args.dtype,
                args.rep,
                args.check,
                flush_buffers=flush_buffers,
                mode=args.small_m_mode,
            )
        for result in case_results:
            results.append(result)
            _print_result(result)
        correct_results = [result for result in case_results if result.correct]
        best_pool = correct_results or case_results
        best = min(best_pool, key=lambda r: r.ms)
        _print_best("BEST" if best.correct else "BEST_UNCHECKED", best)

    _write_csv(args.csv, results)


if __name__ == "__main__":
    main()
