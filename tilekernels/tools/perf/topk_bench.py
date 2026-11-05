#!/usr/bin/env python3
import argparse
import importlib.util
import sys
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
import torch_musa
from tilelang.profiler.bench import do_bench


def load_module(name: str, path: str):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def bench(name, fn):
    fn()
    t_us = do_bench(fn, backend="cupti", warmup=5, rep=30) * 1e3
    print(f"{name},{t_us:.3f}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_REPO_ROOT))
    args = parser.parse_args()

    root = Path(args.root)
    sys.path.insert(0, str(root))

    import tile_kernels

    device = "musa"
    print("case,time_us")
    for nt, ne, nk in [(4001, 72, 6), (8001, 256, 8)]:
        scores = torch.randn((nt, ne), dtype=torch.float32, device=device)
        bench(f"topk_gate nt={nt} experts={ne} topk={nk}", lambda: tile_kernels.moe.topk_gate(scores, nk))

    for nt, ne, ng, nsum, ntopg in [(4001, 72, 12, 2, 4), (8001, 256, 8, 2, 4), (8001, 256, 16, 1, 4)]:
        scores = torch.randn((nt, ng, ne // ng), dtype=torch.float32, device=device)
        bench(
            f"topk_sum nt={nt} experts={ne} groups={ng} sum={nsum} topg={ntopg}",
            lambda: tile_kernels.moe.topk_sum_and_topk_group_idx(scores, nsum, ntopg),
        )


if __name__ == "__main__":
    main()
