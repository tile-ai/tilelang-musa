import argparse
import contextlib
import sys
from pathlib import Path

DEFAULT_REPO_ROOT = Path(__file__).resolve().parents[2]

import torch
import torch_musa
import tilelang
from tilelang import language as T
from tilelang.profiler.bench import do_bench
from tilelang.utils.device import GPUEvent, synchronize


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
def get_empty_kernel():
    @T.prim_func
    def empty_kernel(out: T.Tensor[(1,), T.int32]):
        with T.Kernel(1, threads=32):
            pass

    return empty_kernel


@contextlib.contextmanager
def torch_profile():
    schedule = torch.profiler.schedule(wait=1, warmup=0, active=1, repeat=1)
    with torch.profiler.profile(activities=[torch.profiler.ProfilerActivity.MUSA], schedule=schedule) as profiler:
        yield profiler


def event_loop_ms(fn, n_repeat, flush=False, cache=None):
    start_events = [GPUEvent(enable_timing=True) for _ in range(n_repeat)]
    end_events = [GPUEvent(enable_timing=True) for _ in range(n_repeat)]
    for i in range(n_repeat):
        if flush:
            cache.zero_()
        start_events[i].record()
        fn()
        end_events[i].record()
    synchronize()
    times = torch.tensor([s.elapsed_time(e) for s, e in zip(start_events, end_events)], dtype=torch.float)
    return torch.mean(times).item()


def cupti_total_us(fn, n_repeat, flush=False, cache=None):
    with torch_profile() as profiler:
        for _ in range(2):
            for _ in range(n_repeat):
                if flush:
                    cache.zero_()
                fn()
            profiler.step()
    total = 0.0
    by_key = []
    for event in profiler.key_averages():
        total += event.self_device_time_total
        by_key.append((event.key, event.self_device_time_total))
    by_key.sort(key=lambda x: x[1], reverse=True)
    return total / n_repeat, by_key[:8]


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default=str(DEFAULT_REPO_ROOT))
    parser.add_argument("--repeat", type=int, default=100)
    args = parser.parse_args()
    sys.path.insert(0, str(Path(args.root)))

    print(f"tilelang_version,{tilelang.__version__}")
    out = torch.empty((1,), dtype=torch.int32, device="musa")
    cache = torch.empty(int(256e6 // 4), dtype=torch.int, device="musa")
    kernel = get_empty_kernel()
    fn = lambda: kernel(out)
    fn()
    synchronize()

    print("case,time_us")
    for backend in ["event", "cupti"]:
        t_us = do_bench(fn, backend=backend, warmup=5, rep=30) * 1e3
        print(f"do_bench_{backend}_auto,{t_us:.3f}")
        t_us = do_bench(fn, backend=backend, warmup=5, rep=30, _n_warmup=5, _n_repeat=args.repeat) * 1e3
        print(f"do_bench_{backend}_manual_repeat,{t_us:.3f}")

    print(f"manual_event_no_flush,{event_loop_ms(fn, args.repeat, False, cache) * 1e3:.3f}")
    print(f"manual_event_with_flush,{event_loop_ms(fn, args.repeat, True, cache) * 1e3:.3f}")

    total_no_flush, top_no_flush = cupti_total_us(fn, args.repeat, False, cache)
    print(f"manual_cupti_total_no_flush,{total_no_flush:.3f}")
    for key, value in top_no_flush:
        print(f"manual_cupti_no_flush_key,{key},{value / args.repeat:.3f}")

    total_flush, top_flush = cupti_total_us(fn, args.repeat, True, cache)
    print(f"manual_cupti_total_with_flush,{total_flush:.3f}")
    for key, value in top_flush:
        print(f"manual_cupti_with_flush_key,{key},{value / args.repeat:.3f}")


if __name__ == "__main__":
    main()
