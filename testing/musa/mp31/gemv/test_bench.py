# Copyright (c) 2025, Ted Zadouri, Tri Dao.
"""Torch / TileLang benchmark for Qwen3-VL GEMV-like shapes on MUSA.

Shape convention in this file follows the original torchada script:

    x: (M, N), w: (N, K), y = x @ w, output: (M, K)

The TileLang kernels in ``benchmark_gemv_cases.py`` use the equivalent internal
layout ``A[M, reduce] @ B[out, reduce].T``. Therefore this script prepares
either benchmarks the historical ``B = w.T.contiguous()`` path or passes
the original bench input ``w[N, K]`` directly to a TileLang kernel. It reports
only TileLang ``main_kernel`` device time.
"""

import argparse
import csv
import sys
from pathlib import Path

import torch

try:
    from mate.testing.utils import bench_kineto
except ImportError:  # TileLang-only runs do not need mate.
    bench_kineto = None

try:
    if not __package__:
        raise ModuleNotFoundError(name="benchmark_gemv_cases")
    from .benchmark_gemv_cases import (
        _check_close,
        _gbps,
        _gemv_bytes,
        _make_cache_flush_buffers,
        _measure_tilelang_kernel_kineto,
        _select_m1_bm16_configs,
        _select_m1_simt_configs,
        _select_m5_bm16_configs,
        _tflops,
        gemv_m1_splitk,
        small_m_bm16_sqmma_tme_pc_w_kn,
        small_m_bm16_sqmma_tme_pc_transb,
    )
except ModuleNotFoundError as err:
    if err.name not in {"benchmark_gemv_cases", "musa"}:
        raise
    module_dir = str(Path(__file__).resolve().parent)
    if module_dir not in sys.path:
        sys.path.insert(0, module_dir)
    from benchmark_gemv_cases import (
        _check_close,
        _gbps,
        _gemv_bytes,
        _make_cache_flush_buffers,
        _measure_tilelang_kernel_kineto,
        _select_m1_bm16_configs,
        _select_m1_simt_configs,
        _select_m5_bm16_configs,
        _tflops,
        gemv_m1_splitk,
        small_m_bm16_sqmma_tme_pc_w_kn,
        small_m_bm16_sqmma_tme_pc_transb,
    )
from tilelang.utils.device import synchronize


DEVICE = "musa"
TFLOPS_SCALE = 10**12
GB_SCALE = 1024**3
S2US = 10**6

# (K, N) pairs from the original script. It prints Shape(M, N, K), where N is
# reduction dimension and K is output dimension.
KNS = [
    (4096, 6144),
    (4096, 4096),
    (4096, 24576),
    (12288, 4096),
    (2048, 1280),
    (1024, 2048),
    (2048, 128),
]

GENERALITY_CASES = [
    (M, N, K)
    for M in (1, 2, 5, 8, 16)
    for N, K in (
        (6144, 4096),
        (4096, 4096),
        (24576, 4096),
        (4096, 12288),
        (1280, 2048),
        (2048, 1024),
        (128, 2048),
        # Extra model-like rectangular shapes seen in the local muDNN selector
        # cases. They are opt-in because they are significantly more expensive.
        (5120, 25600),
        (7168, 18432),
    )
]

PRESET_CASES = {
    "screenshot": [(M, N, K) for M in (1, 5) for K, N in KNS],
    "generality": GENERALITY_CASES,
}


def _torch_dtype(name: str):
    if name in ("bf16", "bfloat16"):
        return torch.bfloat16
    if name in ("fp16", "float16", "half"):
        return torch.float16
    raise ValueError(f"Unsupported dtype: {name}")


def _tilelang_dtype_name(name: str) -> str:
    if name in ("bf16", "bfloat16"):
        return "bfloat16"
    if name in ("fp16", "float16", "half"):
        return "float16"
    raise ValueError(f"Unsupported dtype: {name}")


def _fmt_config(config) -> str:
    (
        name,
        block_m,
        block_n,
        block_k,
        threads,
        stages,
        policy,
        a_inner,
        a_outer,
        b_inner,
        b_outer,
    ) = config
    return (
        f"name={name},BM={block_m},BN={block_n},BK={block_k},"
        f"threads={threads},stages={stages},policy={policy},"
        f"a_cache=({a_inner},{a_outer}),b_cache=({b_inner},{b_outer})"
    )


def _w_kn_configs(configs):
    # The original W[K, N] NN SQMMA path is validated with BN64. Preserve the
    # selector decision in the name, but make the temporary cap explicit.
    capped_configs = []
    for (
        name,
        block_m,
        block_n,
        block_k,
        threads,
        stages,
        policy,
        a_inner,
        a_outer,
        b_inner,
        b_outer,
    ) in configs:
        capped_block_n = min(block_n, 64)
        suffix = "_wkn" if capped_block_n == block_n else "_wkn_bn64cap"
        capped_configs.append(
            (
                f"{name}{suffix}",
                block_m,
                capped_block_n,
                block_k,
                threads,
                stages,
                policy,
                a_inner,
                a_outer,
                b_inner,
                b_outer,
            )
        )
    return capped_configs


def _check_bm16_supported(M: int) -> None:
    if M > 16:
        raise ValueError(f"BM16 TileLang GEMV path only supports M <= 16, got M={M}")


def _bench_torch(fn, M: int, N: int, K: int, dtype_bytes: int, args):
    if bench_kineto is None:
        raise RuntimeError("mate.testing.utils.bench_kineto is required for --backend torch/both")
    gemm_time_s, gemv_time_s = bench_kineto(
        fn,
        ("gemm", "gemv"),
        suppress_kineto_output=True,
        num_tests=args.rep,
        trace_path=f"trace_{M}_{N}_{K}.json",
    )
    time_s = gemm_time_s if gemm_time_s != 0 else gemv_time_s
    flops = 2 * M * N * K
    mem_io_byte = (M * N + N * K + M * K) * dtype_bytes
    return {
        "backend": "torch",
        "impl": "torch_matmul",
        "M": M,
        "N": N,
        "K": K,
        "config": "x @ w",
        "time_us": time_s * S2US if time_s > 0 else 0.0,
        "tflops": flops / (time_s * TFLOPS_SCALE) if time_s > 0 else 0.0,
        "gbps": (mem_io_byte / GB_SCALE) / time_s if time_s > 0 else 0.0,
        "bytes": mem_io_byte,
        "correct": True,
    }


def _bench_tilelang(x, w, ref, M: int, N: int, K: int, dtype_name: str, args, weight_layout: str):
    # Bench inputs are x[M, N_reduce], w[N_reduce, K_out]. The contiguous path
    # keeps the older internal B[K_out, N_reduce] layout; original uses w as-is.
    if weight_layout == "strided":
        weight_layout = "original"
    if weight_layout == "contiguous":
        B = w.T.contiguous()
        b_layout = "contiguous_transb"
    elif weight_layout == "original":
        B = w
        b_layout = "original_w_nk"
    else:
        raise ValueError(f"Unsupported weight layout: {weight_layout}")
    dtype = _tilelang_dtype_name(dtype_name)
    flush_buffers = _make_cache_flush_buffers(args.flush_cache_mb)
    bytes_rw = _gemv_bytes(M, K, N, dtype)
    results = []

    def add_result(kind: str, config: str, fn, slice_out, tensor_for_layout=None):
        layout_tensor = B if tensor_for_layout is None else tensor_for_layout
        out = fn()
        synchronize()
        correct = True if not args.check else _check_close(slice_out(out), ref)
        ms = _measure_tilelang_kernel_kineto(fn, rep=args.rep, flush_buffers=flush_buffers)
        results.append(
            {
                "backend": "tilelang",
                "impl": f"{kind}/{weight_layout}",
                "M": M,
                "N": N,
                "K": K,
                "config": f"{config},B_layout={b_layout},B_stride={tuple(layout_tensor.stride())}",
                "time_us": ms * 1000.0,
                "tflops": _tflops(M, K, N, ms),
                "gbps": _gbps(bytes_rw, ms),
                "bytes": bytes_rw,
                "correct": correct,
            }
        )

    if M == 1:
        A1 = x.reshape(N).contiguous()
        simt_mode = "simt" if weight_layout == "original" and args.m1_mode == "simt" else args.m1_mode
        for name, block_n, reduce_threads in _select_m1_simt_configs(K, N, simt_mode):
            simt_B = w.T if weight_layout == "original" else B
            simt_b_layout = "strided_transb" if weight_layout == "original" else b_layout
            kernel = gemv_m1_splitk(K, N, block_n, reduce_threads, dtype=dtype, b_layout=simt_b_layout)
            add_result(
                "m1_splitk",
                f"name={name},BLOCK_N={block_n},reduce_threads={reduce_threads}",
                lambda kernel=kernel, simt_B=simt_B: kernel(A1, simt_B),
                lambda out: out.reshape(1, K),
                tensor_for_layout=simt_B,
            )

        bm16_mode = "sqmma_pc" if weight_layout == "original" and args.m1_mode == "auto" else args.m1_mode
        bm16_configs = _select_m1_bm16_configs(K, N, bm16_mode)
        if weight_layout == "original":
            bm16_configs = _w_kn_configs(bm16_configs)
        if bm16_configs:
            A_pad = torch.zeros((16, N), device=DEVICE, dtype=x.dtype)
            A_pad[0, :] = x[0, :]
            synchronize()
            for config in bm16_configs:
                (
                    _,
                    block_m,
                    block_n,
                    block_k,
                    threads,
                    stages,
                    policy,
                    a_inner,
                    a_outer,
                    b_inner,
                    b_outer,
                ) = config
                kernel_fn = small_m_bm16_sqmma_tme_pc_w_kn if weight_layout == "original" else small_m_bm16_sqmma_tme_pc_transb
                if weight_layout == "original":
                    kernel = kernel_fn(
                        M,
                        K,
                        N,
                        block_m,
                        block_n,
                        block_k,
                        threads,
                        stages,
                        policy,
                        a_inner,
                        a_outer,
                        b_inner,
                        b_outer,
                        dtype=dtype,
                    )
                else:
                    kernel = kernel_fn(
                        M,
                        K,
                        N,
                        block_m,
                        block_n,
                        block_k,
                        threads,
                        stages,
                        policy,
                        a_inner,
                        a_outer,
                        b_inner,
                        b_outer,
                        dtype=dtype,
                        b_layout=b_layout,
                    )
                add_result(
                    "m1_bm16_sqmma_tme_pc_w_nk" if weight_layout == "original" else "m1_bm16_sqmma_tme_pc",
                    _fmt_config(config),
                    lambda kernel=kernel: kernel(A_pad, B),
                    lambda out: out[:M, :K],
                )
    else:
        _check_bm16_supported(M)
        configs = _select_m5_bm16_configs(K, N, args.small_m_mode, M=M)
        if weight_layout == "original":
            configs = _w_kn_configs(configs)
        A_pad = torch.zeros((16, N), device=DEVICE, dtype=x.dtype)
        A_pad[:M, :] = x
        synchronize()
        for config in configs:
            (
                _,
                block_m,
                block_n,
                block_k,
                threads,
                stages,
                policy,
                a_inner,
                a_outer,
                b_inner,
                b_outer,
            ) = config
            kernel_fn = small_m_bm16_sqmma_tme_pc_w_kn if weight_layout == "original" else small_m_bm16_sqmma_tme_pc_transb
            if weight_layout == "original":
                kernel = kernel_fn(
                    M,
                    K,
                    N,
                    block_m,
                    block_n,
                    block_k,
                    threads,
                    stages,
                    policy,
                    a_inner,
                    a_outer,
                    b_inner,
                    b_outer,
                    dtype=dtype,
                )
            else:
                kernel = kernel_fn(
                    M,
                    K,
                    N,
                    block_m,
                    block_n,
                    block_k,
                    threads,
                    stages,
                    policy,
                    a_inner,
                    a_outer,
                    b_inner,
                    b_outer,
                    dtype=dtype,
                    b_layout=b_layout,
                )
            add_result(
                "bm16_sqmma_tme_pc_w_nk" if weight_layout == "original" else "bm16_sqmma_tme_pc",
                _fmt_config(config),
                lambda kernel=kernel: kernel(A_pad, B),
                lambda out: out[:M, :K],
            )

    if not results:
        raise RuntimeError(f"No TileLang candidate selected for M={M}, N={N}, K={K}")
    correct = [r for r in results if r["correct"]]
    return min(correct or results, key=lambda r: r["time_us"]), results


def _print_case(M: int, N: int, K: int, result: dict):
    mem_io_gb = result["bytes"] / GB_SCALE
    print(f"Batch: {M:2d} | Shape (M={M}, N={N}, K={K}) | Backend: {result['backend']}")
    print(
        f" 耗时: {result['time_us']:8.2f} us | FLOPs: {2 * M * N * K:.2e} | "
        f"TFLOPS: {result['tflops']:.2f} | 带宽: {result['gbps']:.2f} GB/s"
    )
    print(f" 显存读写: {result['bytes']:.2e} Byte | {mem_io_gb:.2f} GB | correct={result['correct']}")
    print(f" TileLang实现: {result['impl']} | 分块: {result['config']}")
    print("-" * 120)


def _write_csv(path: str, results: list[dict]) -> None:
    if not path:
        return
    fields = ["backend", "impl", "M", "N", "K", "time_us", "tflops", "gbps", "bytes", "correct", "config"]
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for result in results:
            writer.writerow({field: result.get(field, "") for field in fields})


def _write_markdown(path: str, results: list[dict]) -> None:
    if not path:
        return
    with open(path, "w", encoding="utf-8") as f:
        f.write("# GEMV Benchmark Results\n\n")
        f.write("Device timings only; cache flush and host overhead are excluded.\n\n")
        f.write("| Backend | Impl | Shape `(M,N,K)` | Time us | TFLOPS | GB/s | Correct | Config |\n")
        f.write("| --- | --- | --- | ---: | ---: | ---: | --- | --- |\n")
        for result in results:
            shape = f"({result.get('M')},{result.get('N')},{result.get('K')})"
            f.write(
                f"| {result['backend']} | `{result['impl']}` | `{shape}` | "
                f"{result['time_us']:.2f} | {result['tflops']:.2f} | {result['gbps']:.2f} | "
                f"{result['correct']} | `{result['config']}` |\n"
            )


def main():
    parser = argparse.ArgumentParser(description="Torch/TileLang GEMV benchmark on MUSA")
    parser.add_argument("--backend", choices=["torch", "tilelang", "both"], default="tilelang")
    parser.add_argument("--dtype", choices=["bfloat16", "bf16", "float16", "fp16"], default="bfloat16")
    parser.add_argument("--batches", default="1,5", help="Comma-separated M values")
    parser.add_argument("--case", default="", help="Optional single shape as M,N,K using bench.py convention")
    parser.add_argument(
        "--preset",
        choices=sorted(PRESET_CASES),
        default="screenshot",
        help="Case preset used when --case is not set. screenshot matches the Qwen3-VL image; generality adds M/N/K sweep cases.",
    )
    parser.add_argument("--rep", type=int, default=20)
    parser.add_argument("--check", action="store_true", default=True)
    parser.add_argument("--flush-cache-mb", type=int, default=512)
    parser.add_argument("--m1-mode", choices=["auto", "simt", "sqmma_pc", "all"], default="auto")
    parser.add_argument("--small-m-mode", choices=["auto", "pc", "all"], default="auto")
    parser.add_argument(
        "--weight-layout",
        choices=["contiguous", "original", "strided", "both"],
        default="both",
        help="contiguous uses w.T.contiguous(); original passes bench input w[N,K] directly; strided is a legacy alias for original.",
    )
    parser.add_argument("--csv", default="", help="Optional CSV output path for all printed benchmark rows.")
    parser.add_argument("--markdown", default="", help="Optional markdown output path for all printed benchmark rows.")
    args = parser.parse_args()

    torch.manual_seed(0)
    dtype = _torch_dtype(args.dtype)
    dtype_bytes = torch.finfo(dtype).bits // 8
    if args.case:
        parts = [int(x.strip()) for x in args.case.split(",")]
        if len(parts) != 3:
            raise ValueError("--case must be M,N,K")
        cases = [(parts[0], parts[2], parts[1])]
    else:
        preset_cases = PRESET_CASES[args.preset]
        if args.preset == "screenshot":
            batches = [int(x) for x in args.batches.split(",") if x.strip()]
            preset_cases = [(M, N, K) for M, N, K in preset_cases if M in batches]
        cases = [(M, K, N) for M, N, K in preset_cases]

    all_results = []
    for M, K, N in cases:
        x = torch.randn((M, N), device=DEVICE, dtype=dtype)
        w = torch.randn((N, K), device=DEVICE, dtype=dtype)
        ref = x @ w
        synchronize()

        if args.backend in ("torch", "both"):
            torch_result = _bench_torch(lambda x=x, w=w: x @ w, M, N, K, dtype_bytes, args)
            _print_case(M, N, K, torch_result)
            all_results.append(torch_result)

        if args.backend in ("tilelang", "both"):
            layouts = ["contiguous", "original"] if args.weight_layout == "both" else [args.weight_layout]
            for weight_layout in layouts:
                tilelang_best, tilelang_results = _bench_tilelang(x, w, ref, M, N, K, args.dtype, args, weight_layout)
                if args.m1_mode == "all" or args.small_m_mode in ("pc", "all"):
                    for result in tilelang_results:
                        _print_case(M, N, K, result)
                        all_results.append(result)
                else:
                    _print_case(M, N, K, tilelang_best)
                    all_results.append(tilelang_best)

    _write_csv(args.csv, all_results)
    _write_markdown(args.markdown, all_results)


if __name__ == "__main__":
    main()
