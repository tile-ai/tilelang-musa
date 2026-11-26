#pragma once

#include "intrin.h"

#include <cstdint>
#include <type_traits>

namespace tl {

template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op);

template <typename T> struct AccType {
  using type = T;
};

#ifdef TL_MUSA_ENABLE_FP16
template <> struct AccType<half> {
  using type = float;
};
#endif

#ifdef TL_MUSA_ENABLE_BF16
template <> struct AccType<mt_bfloat16> {
  using type = float;
};
#endif

struct SumOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x + y;
  }
};

struct ProdOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x * y;
  }
};

struct MaxOp {
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
};

struct MinOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x < y ? x : y;
  }

#ifdef TL_MUSA_ENABLE_FP16
  TL_DEVICE half operator()(half const &x, half const &y) {
    return __hmin(x, y);
  }
#endif

#ifdef TL_MUSA_ENABLE_BF16
  TL_DEVICE mt_bfloat16 operator()(mt_bfloat16 const &x, mt_bfloat16 const &y) {
    return __hmin(x, y);
  }
#endif
};

struct BitAndOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x & y;
  }
};

struct BitOrOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x | y;
  }
};

struct BitXorOp {
  template <typename T> TL_DEVICE T operator()(T const &x, T const &y) {
    return x ^ y;
  }
};

struct SyncThreadsBarrier {
  template <int phase = 0> static TL_DEVICE void sync() { __syncthreads(); }
};

template <typename T>
TL_DEVICE T shfl_xor_sync(unsigned mask, T val, int lane_mask) {
  return __shfl_xor_sync(mask, val, lane_mask);
}

TL_DEVICE float2 shfl_xor_sync(unsigned mask, float2 val, int lane_mask) {
  float2 out;
  out.x = __shfl_xor_sync(mask, val.x, lane_mask);
  out.y = __shfl_xor_sync(mask, val.y, lane_mask);
  return out;
}

#ifdef TL_MUSA_ENABLE_FP16
TL_DEVICE half shfl_xor_sync(unsigned mask, half val, int lane_mask) {
  float raw = static_cast<float>(val);
  return half(__shfl_xor_sync(mask, raw, lane_mask));
}
#endif

#ifdef TL_MUSA_ENABLE_BF16
TL_DEVICE mt_bfloat16 shfl_xor_sync(unsigned mask, mt_bfloat16 val,
                                    int lane_mask) {
  float raw = static_cast<float>(val);
  return mt_bfloat16(__shfl_xor_sync(mask, raw, lane_mask));
}
#endif

template <class Reducer, int threads, int scale, int thread_offset = 0,
          class Barrier = SyncThreadsBarrier, int batch_size = 1,
          int workspace_stride = 0>
struct AllReduce {
  static_assert(threads > 0, "tl::AllReduce threads must be positive");
  static_assert(scale > 0, "tl::AllReduce scale must be positive");
  static_assert(threads % scale == 0,
                "tl::AllReduce threads must be divisible by scale");
  static_assert(((threads / scale) & (threads / scale - 1)) == 0,
                "tl::AllReduce reduce width must be a power of two");

  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return x;
    } else {
      return butterfly_reduce_scalar(x, red_buf);
    }
  }

  template <typename T>
  static TL_DEVICE void run_batch(T *x, T *red_buf = nullptr) {
    if constexpr (threads == scale) {
      return;
    } else {
      butterfly_reduce_batch(x, red_buf);
    }
  }

private:
  using Next = AllReduce<Reducer, threads / 2, scale, thread_offset, Barrier,
                         batch_size, workspace_stride>;

  template <typename T>
  static TL_DEVICE T butterfly_reduce_scalar(T x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      Barrier::template sync<1>();
      red_buf[threadIdx.x - thread_offset] = x;
      Barrier::template sync<2>();
      x = Reducer()(x, red_buf[(threadIdx.x - thread_offset) ^ offset]);
    } else {
      x = Reducer()(x, tl::shfl_xor_sync(uint32_t(-1), x, offset));
    }
    if constexpr (offset == scale) {
      return x;
    } else {
      return Next::run(x, red_buf);
    }
  }

  template <typename T>
  static TL_DEVICE void butterfly_reduce_batch(T *x, T *red_buf) {
    constexpr int offset = threads / 2;
    if constexpr (offset >= 32) {
      Barrier::template sync<1>();
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        red_buf[(threadIdx.x - thread_offset) + i * workspace_stride] = x[i];
      }
      Barrier::template sync<2>();
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] =
            Reducer()(x[i], red_buf[((threadIdx.x - thread_offset) ^ offset) +
                                    i * workspace_stride]);
      }
    } else {
#pragma unroll
      for (int i = 0; i < batch_size; i++) {
        x[i] = Reducer()(x[i], tl::shfl_xor_sync(uint32_t(-1), x[i], offset));
      }
    }
    if constexpr (offset == scale) {
      return;
    } else {
      Next::run_batch(x, red_buf);
    }
  }
};

template <typename T, typename ReduceOp>
TL_DEVICE T warp_reduce(T value, ReduceOp op) {
  constexpr uint32_t mask = 0xffffffff;
  int warp_size = detail::default_warp_size();
  for (int offset = warp_size / 2; offset > 0; offset >>= 1) {
    value = op(value, tl::shfl_xor_sync(mask, value, offset));
  }
  return value;
}

template <typename T> TL_DEVICE T warp_reduce_sum(T value) {
  return warp_reduce<T>(value, SumOp());
}

template <typename T> TL_DEVICE T warp_reduce_prod(T value) {
  return warp_reduce<T>(value, ProdOp());
}

// Warp-first multi-warp AllReduce supporting:
//   - Full-block and sub-block multi-warp groups (all_threads % threads == 0),
//   - Arbitrary power-of-two scale <= warp_size (1, 2, 4, 8, 16, 32),
//   - Divergence-free, race-free back-to-back invocations using only 2 barriers.
template <class Reducer, int threads, int scale = 1, int thread_offset = 0,
          class Barrier = SyncThreadsBarrier, int batch_size = 1,
          int workspace_stride = 0>
struct WarpFirstAllReduce {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ <= 220)
  static constexpr int warp_size = 128;
#else
  static constexpr int warp_size = 32;
#endif
  static constexpr int num_warps = threads / warp_size;

  static_assert(threads > warp_size,
                "WarpFirstAllReduce requires more than one warp");
  static_assert(threads % warp_size == 0,
                "WarpFirstAllReduce threads must be warp aligned");
  static_assert(scale > 0 && scale <= warp_size,
                "WarpFirstAllReduce requires 0 < scale <= warp_size");
  static_assert((scale & (scale - 1)) == 0,
                "WarpFirstAllReduce scale must be a power of two");
  static_assert(num_warps <= 32,
                "WarpFirstAllReduce cross-warp reduction exceeds mask width");
  static_assert((num_warps & (num_warps - 1)) == 0,
                "WarpFirstAllReduce warp count must be a power of two");
  static_assert(batch_size == 1 || workspace_stride > 0,
                "WarpFirstAllReduce batched workspace must have a stride");

  template <typename T> static TL_DEVICE T run(T x, T *red_buf = nullptr) {
    const int local_thread = static_cast<int>(threadIdx.x) - thread_offset;
    const int group_id = local_thread / threads;
    const int tid_in_group = local_thread % threads;
    const int warp_in_group = tid_in_group / warp_size;
    const int lane = tid_in_group % warp_size;
    T *group_buf = red_buf + group_id * threads;

#pragma unroll
    for (int offset = warp_size / 2; offset >= scale; offset >>= 1) {
      x = Reducer()(x, tl::shfl_xor_sync(0xffffffffu, x, offset));
    }

    if (lane < scale) {
      group_buf[warp_in_group * scale + lane] = x;
    }
    Barrier::template sync<1>();

    const int sub_lane = lane & (scale - 1);
    T total = group_buf[sub_lane];
#pragma unroll
    for (int w = 1; w < num_warps; ++w) {
      total = Reducer()(total, group_buf[w * scale + sub_lane]);
    }
    Barrier::template sync<2>();
    return total;
  }

  template <typename T>
  static TL_DEVICE void run_batch(T *x, T *red_buf = nullptr) {
    const int local_thread = static_cast<int>(threadIdx.x) - thread_offset;
    const int group_id = local_thread / threads;
    const int tid_in_group = local_thread % threads;
    const int warp_in_group = tid_in_group / warp_size;
    const int lane = tid_in_group % warp_size;
    const int group_base = group_id * threads;

#pragma unroll
    for (int i = 0; i < batch_size; ++i) {
      T val = x[i];
#pragma unroll
      for (int offset = warp_size / 2; offset >= scale; offset >>= 1) {
        val = Reducer()(val, tl::shfl_xor_sync(0xffffffffu, val, offset));
      }
      if (lane < scale) {
        red_buf[i * workspace_stride + group_base + warp_in_group * scale +
                lane] = val;
      }
    }
    Barrier::template sync<1>();

    const int sub_lane = lane & (scale - 1);
#pragma unroll
    for (int i = 0; i < batch_size; ++i) {
      T *group_buf = red_buf + i * workspace_stride + group_base;
      T total = group_buf[sub_lane];
#pragma unroll
      for (int w = 1; w < num_warps; ++w) {
        total = Reducer()(total, group_buf[w * scale + sub_lane]);
      }
      x[i] = total;
    }
    Barrier::template sync<2>();
  }
};

template <typename T> TL_DEVICE T warp_reduce_max(T value) {
  return warp_reduce<T>(value, MaxOp());
}

template <typename T> TL_DEVICE T warp_reduce_min(T value) {
  return warp_reduce<T>(value, MinOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitand(T value) {
  return warp_reduce<T>(value, BitAndOp());
}

template <typename T> TL_DEVICE T warp_reduce_bitor(T value) {
  return warp_reduce<T>(value, BitOrOp());
}

} // namespace tl
