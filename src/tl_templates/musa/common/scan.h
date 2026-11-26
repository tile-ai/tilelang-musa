#pragma once

#include "intrin.h"

#include <limits>

namespace tl {

struct ScanSumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }

  template <typename T> TL_DEVICE static T identity() { return T(0); }
};

template <typename T> struct ScanMaxIdentity {
  TL_DEVICE static T value() { return std::numeric_limits<T>::lowest(); }
};

#ifdef TL_MUSA_ENABLE_FP16
template <> struct ScanMaxIdentity<half> {
  TL_DEVICE static half value() { return half(-65504.0f); }
};
#endif

#ifdef TL_MUSA_ENABLE_BF16
template <> struct ScanMaxIdentity<mt_bfloat16> {
  TL_DEVICE static mt_bfloat16 value() {
    return mt_bfloat16(-3.3895313892515355e38f);
  }
};
#endif

struct ScanMaxOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x > y ? x : y;
  }

#ifdef TL_MUSA_ENABLE_FP16
  TL_DEVICE half operator()(half const &x, half const &y) {
    return __hmax(x, y);
  }
#endif

#ifdef TL_MUSA_ENABLE_BF16
  TL_DEVICE mt_bfloat16 operator()(mt_bfloat16 const &x, mt_bfloat16 const &y) {
    return __hmax(x, y);
  }
#endif

  template <typename T> TL_DEVICE static T identity() {
    return ScanMaxIdentity<T>::value();
  }
};

template <typename T>
TL_DEVICE T shfl_down_sync(unsigned mask, T val, int delta) {
  return __shfl_down_sync(mask, val, delta);
}

template <typename T>
TL_DEVICE T shfl_up_sync(unsigned mask, T val, int delta) {
  return __shfl_up_sync(mask, val, delta);
}

template <typename T>
TL_DEVICE T shfl_sync(unsigned mask, T val, int src_lane) {
  return __shfl_sync(mask, val, src_lane);
}

#ifdef TL_MUSA_ENABLE_FP16
TL_DEVICE half shfl_down_sync(unsigned mask, half val, int delta) {
  float raw = static_cast<float>(val);
  return half(__shfl_down_sync(mask, raw, delta));
}

TL_DEVICE half shfl_up_sync(unsigned mask, half val, int delta) {
  float raw = static_cast<float>(val);
  return half(__shfl_up_sync(mask, raw, delta));
}

TL_DEVICE half shfl_sync(unsigned mask, half val, int src_lane) {
  float raw = static_cast<float>(val);
  return half(__shfl_sync(mask, raw, src_lane));
}
#endif

#ifdef TL_MUSA_ENABLE_BF16
TL_DEVICE mt_bfloat16 shfl_down_sync(unsigned mask, mt_bfloat16 val,
                                     int delta) {
  float raw = static_cast<float>(val);
  return mt_bfloat16(__shfl_down_sync(mask, raw, delta));
}

TL_DEVICE mt_bfloat16 shfl_up_sync(unsigned mask, mt_bfloat16 val, int delta) {
  float raw = static_cast<float>(val);
  return mt_bfloat16(__shfl_up_sync(mask, raw, delta));
}

TL_DEVICE mt_bfloat16 shfl_sync(unsigned mask, mt_bfloat16 val, int src_lane) {
  float raw = static_cast<float>(val);
  return mt_bfloat16(__shfl_sync(mask, raw, src_lane));
}
#endif

template <class Reducer, bool reverse, typename T, int SEG = 32>
static TL_DEVICE T InclusiveScanLine(const T *__restrict__ src,
                                     T *__restrict__ dst, int extent,
                                     int src_stride, int dst_stride) {
  T carry = Reducer::template identity<T>();
  if (extent <= 0)
    return carry;

  constexpr unsigned MASK = 0xffffffff;
  const int lane = threadIdx.x % SEG;
  const int num_segments = (extent + SEG - 1) / SEG;

  if constexpr (reverse) {
    for (int seg = num_segments - 1; seg >= 0; --seg) {
      const int idx = seg * SEG + lane;
      T val = (idx < extent) ? src[idx * src_stride]
                             : Reducer::template identity<T>();

#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_down_sync(MASK, val, off);
        if (lane < SEG - off)
          val = Reducer()(val, n);
      }

      val = Reducer()(val, carry);

      if (idx < extent)
        dst[idx * dst_stride] = val;

      carry = tl::shfl_sync(MASK, val, 0);
    }
  } else {
    for (int seg = 0; seg < num_segments; ++seg) {
      const int idx = seg * SEG + lane;
      T val = (idx < extent) ? src[idx * src_stride]
                             : Reducer::template identity<T>();

#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_up_sync(MASK, val, off);
        if (lane >= off)
          val = Reducer()(val, n);
      }

      val = Reducer()(val, carry);

      if (idx < extent)
        dst[idx * dst_stride] = val;

      carry = tl::shfl_sync(MASK, val, SEG - 1);
    }
  }
  return carry;
}

template <class Reducer, int threads, bool reverse, typename T, int SEG = 32>
static TL_DEVICE void InclusiveScan1DMultiWarp(const T *__restrict__ src,
                                               T *__restrict__ dst,
                                               int extent) {
  if (extent <= 0)
    return;

  constexpr int kNumWarps = threads / SEG;
  static_assert(kNumWarps > 1 && kNumWarps <= SEG,
                "InclusiveScan1DMultiWarp requires 2 <= threads / SEG <= SEG");
  constexpr unsigned MASK = 0xffffffff;

  struct alignas(T) RawSlot {
    unsigned char bytes[sizeof(T)];
  };
  __shared__ RawSlot warp_totals_raw[kNumWarps];
  T *wt_buf = reinterpret_cast<T *>(warp_totals_raw);

  const int tid = static_cast<int>(threadIdx.x);
  const int warp_id = tid / SEG;
  const int lane = tid % SEG;

  // Fast register-resident 1-pass scan when the whole array fits in one CTA.
  if (extent <= threads) {
    T val = (tid < extent) ? src[tid] : Reducer::template identity<T>();
    if constexpr (reverse) {
#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_down_sync(MASK, val, off);
        if (lane < SEG - off)
          val = Reducer()(val, n);
      }
      if (lane == 0)
        wt_buf[warp_id] = val;
      __syncthreads();

      if (warp_id == 0) {
        T wt = (lane < kNumWarps) ? wt_buf[lane]
                                  : Reducer::template identity<T>();
#pragma unroll
        for (int off = 1; off < kNumWarps; off <<= 1) {
          T n = tl::shfl_down_sync(MASK, wt, off);
          if (lane + off < kNumWarps)
            wt = Reducer()(wt, n);
        }
        if (lane < kNumWarps)
          wt_buf[lane] = wt;
      }
      __syncthreads();

      if (warp_id + 1 < kNumWarps)
        val = Reducer()(val, wt_buf[warp_id + 1]);
      if (tid < extent)
        dst[tid] = val;
    } else {
#pragma unroll
      for (int off = 1; off < SEG; off <<= 1) {
        T n = tl::shfl_up_sync(MASK, val, off);
        if (lane >= off)
          val = Reducer()(val, n);
      }
      if (lane == SEG - 1)
        wt_buf[warp_id] = val;
      __syncthreads();

      if (warp_id == 0) {
        T wt = (lane < kNumWarps) ? wt_buf[lane]
                                  : Reducer::template identity<T>();
#pragma unroll
        for (int off = 1; off < kNumWarps; off <<= 1) {
          T n = tl::shfl_up_sync(MASK, wt, off);
          if (lane >= off)
            wt = Reducer()(wt, n);
        }
        if (lane < kNumWarps)
          wt_buf[lane] = wt;
      }
      __syncthreads();

      if (warp_id > 0)
        val = Reducer()(val, wt_buf[warp_id - 1]);
      if (tid < extent)
        dst[tid] = val;
    }
    return;
  }

  // General N > threads: warp-contiguous 3-phase scan.
  // Only 2 __syncthreads() total regardless of N, with stride-1 conflict-free
  // shared memory accesses in every warp.
  const int raw_per_warp = (extent + kNumWarps - 1) / kNumWarps;
  const int per_warp = ((raw_per_warp + SEG - 1) / SEG) * SEG;
  const int start = warp_id * per_warp;
  const int len =
      (start < extent) ? ((start + per_warp < extent) ? per_warp : (extent - start)) : 0;

  T warp_total = (len > 0) ? InclusiveScanLine<Reducer, reverse, T, SEG>(
                                 src + start, dst + start, len, 1, 1)
                           : Reducer::template identity<T>();
  if (lane == 0) {
    wt_buf[warp_id] = warp_total;
  }
  __syncthreads();

  if (warp_id == 0) {
    T wt = (lane < kNumWarps) ? wt_buf[lane] : Reducer::template identity<T>();
    if constexpr (reverse) {
#pragma unroll
      for (int off = 1; off < kNumWarps; off <<= 1) {
        T n = tl::shfl_down_sync(MASK, wt, off);
        if (lane + off < kNumWarps)
          wt = Reducer()(wt, n);
      }
    } else {
#pragma unroll
      for (int off = 1; off < kNumWarps; off <<= 1) {
        T n = tl::shfl_up_sync(MASK, wt, off);
        if (lane >= off)
          wt = Reducer()(wt, n);
      }
    }
    if (lane < kNumWarps) {
      wt_buf[lane] = wt;
    }
  }
  __syncthreads();

  if constexpr (reverse) {
    if (warp_id + 1 < kNumWarps && len > 0) {
      const T prefix = wt_buf[warp_id + 1];
      const int end = start + len;
      for (int i = start + lane; i < end; i += SEG) {
        dst[i] = Reducer()(dst[i], prefix);
      }
    }
  } else {
    if (warp_id > 0 && len > 0) {
      const T prefix = wt_buf[warp_id - 1];
      const int end = start + len;
      for (int i = start + lane; i < end; i += SEG) {
        dst[i] = Reducer()(dst[i], prefix);
      }
    }
  }
}

template <class Reducer, int threads, bool reverse = false>
struct InclusiveScan1D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    if constexpr (threads > SEG) {
      if (static_cast<int>(blockDim.x) == threads && N > SEG) {
        InclusiveScan1DMultiWarp<Reducer, threads, reverse, T, SEG>(src, dst,
                                                                    N);
        return;
      }
    }
    if (threadIdx.x >= SEG)
      return;
    InclusiveScanLine<Reducer, reverse, T, SEG>(src, dst, N, 1, 1);
  }
};

template <class Reducer, int threads, int Axis = 0, bool reverse = false>
struct InclusiveScan2D {
  static_assert(threads == 1024 or threads == 512 or threads == 256 or
                threads == 128 or threads == 64 or threads == 32);
  static_assert(Axis == 0 or Axis == 1);
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W, int src_stride, int dst_stride) {
    if (H <= 0 || W <= 0)
      return;

    if constexpr (Axis == 1 && threads > SEG) {
      if (H == 1 && static_cast<int>(blockDim.x) == threads && W > SEG) {
        InclusiveScan1DMultiWarp<Reducer, threads, reverse, T, SEG>(src, dst,
                                                                    W);
        return;
      }
    }

    constexpr int TILE = threads / SEG;
    const int item = threadIdx.x / SEG;

    if constexpr (Axis == 1) {
      const int num_blocks = (H + TILE - 1) / TILE;
      for (int b = 0; b < num_blocks; ++b) {
        const int row = b * TILE + item;
        if (row >= H)
          return;
        InclusiveScanLine<Reducer, reverse, T, SEG>(
            src + row * src_stride, dst + row * dst_stride, W, 1, 1);
      }
    } else {
      const int num_blocks = (W + TILE - 1) / TILE;
      for (int b = 0; b < num_blocks; ++b) {
        const int col = b * TILE + item;
        if (col >= W)
          return;
        InclusiveScanLine<Reducer, reverse, T, SEG>(src + col, dst + col, H,
                                                    src_stride, dst_stride);
      }
    }
  }
};

template <int threads, bool reverse = false> struct CumSum1D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    InclusiveScan1D<ScanSumOp, threads, reverse>::template run<T, SEG>(src, dst,
                                                                       N);
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumSum2D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W, int src_stride, int dst_stride) {
    InclusiveScan2D<ScanSumOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W, src_stride, dst_stride);
  }
};

template <int threads, bool reverse = false> struct CumMax1D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int N) {
    InclusiveScan1D<ScanMaxOp, threads, reverse>::template run<T, SEG>(src, dst,
                                                                       N);
  }
};

template <int threads, int Axis = 0, bool reverse = false> struct CumMax2D {
  template <typename T, int SEG = 32>
  static TL_DEVICE void run(const T *__restrict__ src, T *__restrict__ dst,
                            int H, int W, int src_stride, int dst_stride) {
    InclusiveScan2D<ScanMaxOp, threads, Axis, reverse>::template run<T, SEG>(
        src, dst, H, W, src_stride, dst_stride);
  }
};

} // namespace tl
