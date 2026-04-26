# Performance Debug Helpers

Small MUSA/TileLang helper programs used to validate benchmark overhead and top-k kernel performance.

Run from the repo root inside the MUSA Docker container:

```bash
export MUSA_VISIBLE_DEVICES=4
python tools/perf/profile_overhead.py --root /home/workspace/tilelang_dev/tilekernels --repeat 100
python tools/perf/empty_kernel_bench.py --root /home/workspace/tilelang_dev/tilekernels
python tools/perf/topk_bench.py --root /home/workspace/tilelang_dev/tilekernels
python tools/perf/dump_topk_sum_source.py
mcc tools/perf/empty_kernel_latency.mu -o /tmp/empty_kernel_latency \
  -L/usr/local/musa-4.3.5/lib -lmusart -Wl,-rpath,/usr/local/musa-4.3.5/lib
/tmp/empty_kernel_latency 1000 100
```

`profile_overhead.py` is useful for checking whether CUPTI timing excludes cache flush kernels such as `KernelFill<int,...>`.
