#include <musa_runtime.h>

#include <chrono>
#include <cstdio>
#include <cstdlib>

#define CHECK_MUSA(expr)                                                         \
  do {                                                                          \
    musaError_t status = (expr);                                                 \
    if (status != musaSuccess) {                                                 \
      std::fprintf(stderr, "MUSA error %s:%d: %s\n", __FILE__, __LINE__,        \
                   musaGetErrorString(status));                                 \
      std::exit(1);                                                              \
    }                                                                           \
  } while (0)

__global__ void empty_kernel() {}

__global__ void min_write_kernel(int* out) {
  if (blockIdx.x == 0 && threadIdx.x == 0) {
    out[0] = 1;
  }
}

template <typename Fn>
float bench_us(Fn launch, int warmup, int repeat) {
  musaEvent_t start;
  musaEvent_t stop;
  CHECK_MUSA(musaEventCreate(&start));
  CHECK_MUSA(musaEventCreate(&stop));

  for (int i = 0; i < warmup; ++i) {
    launch();
  }
  CHECK_MUSA(musaDeviceSynchronize());

  CHECK_MUSA(musaEventRecord(start));
  for (int i = 0; i < repeat; ++i) {
    launch();
  }
  CHECK_MUSA(musaEventRecord(stop));
  CHECK_MUSA(musaEventSynchronize(stop));

  float ms = 0.0f;
  CHECK_MUSA(musaEventElapsedTime(&ms, start, stop));
  CHECK_MUSA(musaEventDestroy(start));
  CHECK_MUSA(musaEventDestroy(stop));
  return ms * 1000.0f / repeat;
}

template <typename Fn>
float bench_host_sync_us(Fn launch, int warmup, int repeat) {
  for (int i = 0; i < warmup; ++i) {
    launch();
    CHECK_MUSA(musaDeviceSynchronize());
  }

  auto start = std::chrono::steady_clock::now();
  for (int i = 0; i < repeat; ++i) {
    launch();
    CHECK_MUSA(musaDeviceSynchronize());
  }
  auto stop = std::chrono::steady_clock::now();
  std::chrono::duration<double, std::micro> elapsed = stop - start;
  return static_cast<float>(elapsed.count() / repeat);
}

int main(int argc, char** argv) {
  int warmup = 1000;
  int repeat = 10000;
  if (argc > 1) repeat = std::atoi(argv[1]);
  if (argc > 2) warmup = std::atoi(argv[2]);

  int* out = nullptr;
  CHECK_MUSA(musaMalloc(&out, sizeof(int)));
  CHECK_MUSA(musaMemset(out, 0, sizeof(int)));

  std::printf("case,time_us\n");
  for (int blocks : {1, 16, 1024}) {
    for (int threads : {1, 32, 128}) {
      float empty_us = bench_us([&] { empty_kernel<<<blocks, threads>>>(); }, warmup, repeat);
      CHECK_MUSA(musaGetLastError());
      std::printf("event_empty blocks=%d threads=%d,%.3f\n", blocks, threads, empty_us);

      float write_us = bench_us([&] { min_write_kernel<<<blocks, threads>>>(out); }, warmup, repeat);
      CHECK_MUSA(musaGetLastError());
      std::printf("event_min_write blocks=%d threads=%d,%.3f\n", blocks, threads, write_us);

      float host_empty_us = bench_host_sync_us([&] { empty_kernel<<<blocks, threads>>>(); }, warmup, repeat);
      CHECK_MUSA(musaGetLastError());
      std::printf("host_sync_empty blocks=%d threads=%d,%.3f\n", blocks, threads, host_empty_us);

      float host_write_us = bench_host_sync_us([&] { min_write_kernel<<<blocks, threads>>>(out); }, warmup, repeat);
      CHECK_MUSA(musaGetLastError());
      std::printf("host_sync_min_write blocks=%d threads=%d,%.3f\n", blocks, threads, host_write_us);
    }
  }

  CHECK_MUSA(musaFree(out));
  return 0;
}
