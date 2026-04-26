import argparse
import sys
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
import torch_musa
import tilelang
from tilelang import language as T
from tilelang.profiler.bench import do_bench


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    },
)
def get_empty_kernel(num_blocks: int, num_threads: int):
    @T.prim_func
    def empty_kernel(out: T.Tensor[(1,), T.int32]):
        with T.Kernel(num_blocks, threads=num_threads):
            pass

    return empty_kernel


@tilelang.jit(
    pass_configs={
        tilelang.PassConfigKey.TL_DISABLE_WARP_SPECIALIZED: True,
        tilelang.PassConfigKey.TL_DISABLE_THREAD_STORAGE_SYNC: True,
        tilelang.PassConfigKey.TL_ENABLE_MUSA_BURST: True,
        tilelang.PassConfigKey.TL_ENABLE_REDUCE_BURST: True,
        tilelang.PassConfigKey.TL_DISABLE_SAFE_MEMORY_ACCESS: True,
        tilelang.PassConfigKey.TL_DISABLE_INDEX_TYPE_PROMOTION: True,
    },
)
def get_min_write_kernel(num_blocks: int, num_threads: int):
    @T.prim_func
    def min_write_kernel(out: T.Tensor[(1,), T.int32]):
        with T.Kernel(num_blocks, threads=num_threads):
            if T.get_thread_binding() == 0:
                out[0] = 1

    return min_write_kernel


def bench(name, fn):
    fn()
    torch.musa.synchronize()
    t_us = do_bench(fn, backend="cupti", warmup=20, rep=200) * 1e3
    print(f"{name},{t_us:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_REPO_ROOT))
    args = parser.parse_args()
    root = Path(args.root)
    sys.path.insert(0, str(root))

    print(f"tilelang_version,{tilelang.__version__}")
    print("case,time_us")
    noop_tensor = torch.empty((1,), dtype=torch.int32, device="musa")
    bench("torch_empty_like", lambda: torch.empty_like(noop_tensor))
    bench("torch_add_zero", lambda: noop_tensor.add_(0))
    out = torch.empty((1,), dtype=torch.int32, device="musa")
    for blocks, threads in [(1, 1), (1, 32), (1, 128), (16, 128), (1024, 128)]:
        empty_kernel = get_empty_kernel(blocks, threads)
        bench(f"empty blocks={blocks} threads={threads}", lambda k=empty_kernel: k(out))
        min_write_kernel = get_min_write_kernel(blocks, threads)
        bench(f"min_write blocks={blocks} threads={threads}", lambda k=min_write_kernel: k(out))


if __name__ == "__main__":
    main()
