#pragma once

#include "intrin.h"

#include <cstdint>
#include <limits>

namespace tl {

namespace sort_detail {

template <typename T> struct SortKeyTraits {
  TL_DEVICE static bool less(const T &a, const T &b) { return a < b; }
  TL_DEVICE static bool equal(const T &a, const T &b) { return a == b; }
  TL_DEVICE static T pos_sentinel() { return std::numeric_limits<T>::max(); }
  TL_DEVICE static T neg_sentinel() { return std::numeric_limits<T>::lowest(); }
};

template <> struct SortKeyTraits<float> {
  TL_DEVICE static bool less(float a, float b) { return a < b; }
  TL_DEVICE static bool equal(float a, float b) { return a == b; }
  TL_DEVICE static float pos_sentinel() { return __int_as_float(0x7f800000); }
  TL_DEVICE static float neg_sentinel() { return __int_as_float(0xff800000); }
};

template <> struct SortKeyTraits<double> {
  TL_DEVICE static bool less(double a, double b) { return a < b; }
  TL_DEVICE static bool equal(double a, double b) { return a == b; }
  TL_DEVICE static double pos_sentinel() {
    return __longlong_as_double(0x7ff0000000000000LL);
  }
  TL_DEVICE static double neg_sentinel() {
    return __longlong_as_double(0xfff0000000000000ULL);
  }
};

#ifdef TL_MUSA_ENABLE_FP16
template <> struct SortKeyTraits<half> {
  TL_DEVICE static bool less(const half &a, const half &b) {
    return static_cast<float>(a) < static_cast<float>(b);
  }
  TL_DEVICE static bool equal(const half &a, const half &b) {
    return static_cast<float>(a) == static_cast<float>(b);
  }
  TL_DEVICE static half pos_sentinel() { return __ushort_as_half(0x7c00); }
  TL_DEVICE static half neg_sentinel() { return __ushort_as_half(0xfc00); }
};
#endif

#ifdef TL_MUSA_ENABLE_BF16
template <> struct SortKeyTraits<mt_bfloat16> {
  TL_DEVICE static bool less(const mt_bfloat16 &a, const mt_bfloat16 &b) {
    return static_cast<float>(a) < static_cast<float>(b);
  }
  TL_DEVICE static bool equal(const mt_bfloat16 &a, const mt_bfloat16 &b) {
    return static_cast<float>(a) == static_cast<float>(b);
  }
  TL_DEVICE static mt_bfloat16 pos_sentinel() {
    return __ushort_as_bfloat16(0x7f80);
  }
  TL_DEVICE static mt_bfloat16 neg_sentinel() {
    return __ushort_as_bfloat16(0xff80);
  }
};
#endif

template <bool descending, typename T>
TL_DEVICE bool comes_before_val(const T &a, const T &b) {
  return descending ? SortKeyTraits<T>::less(b, a)
                    : SortKeyTraits<T>::less(a, b);
}

template <bool descending, typename T, typename IdxT>
TL_DEVICE bool comes_before_pair(const T &va, IdxT ia, const T &vb, IdxT ib) {
  if (SortKeyTraits<T>::equal(va, vb)) {
    // Stable tie-breaking: smaller original index comes first in the final
    // total order (matches torch.sort(..., stable=True) / torch.topk).
    return ia < ib;
  }
  return descending ? SortKeyTraits<T>::less(vb, va)
                    : SortKeyTraits<T>::less(va, vb);
}

template <bool descending, typename T>
TL_DEVICE bool should_swap_val(const T &va, const T &vb, bool want_forward) {
  return want_forward ? comes_before_val<descending>(vb, va)
                      : comes_before_val<descending>(va, vb);
}

template <bool descending, typename T, typename IdxT>
TL_DEVICE bool should_swap_pair(const T &va, IdxT ia, const T &vb, IdxT ib,
                                bool want_forward) {
  return want_forward ? comes_before_pair<descending>(vb, ib, va, ia)
                      : comes_before_pair<descending>(va, ia, vb, ib);
}

TL_DEVICE int linear_tid() {
  return static_cast<int>(threadIdx.x +
                          blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z));
}

TL_DEVICE int block_threads() {
  return static_cast<int>(blockDim.x * blockDim.y * blockDim.z);
}

} // namespace sort_detail

// In-place block-parallel bitonic sort of a 1D shared buffer of length P
// (power of 2). When valid_n < P, entries [valid_n, P) are filled with
// sentinels so valid entries occupy [0, valid_n) in sorted order.
template <bool descending> struct BlockBitonicSort1D {
  template <typename T>
  static TL_DEVICE void run(T *__restrict__ vals, int valid_n, int P) {
    const int tid = sort_detail::linear_tid();
    const int nthreads = sort_detail::block_threads();
    const T sentinel = descending ? sort_detail::SortKeyTraits<T>::neg_sentinel()
                                  : sort_detail::SortKeyTraits<T>::pos_sentinel();
    for (int i = valid_n + tid; i < P; i += nthreads) {
      vals[i] = sentinel;
    }
    __syncthreads();

    for (int k = 2; k <= P; k <<= 1) {
      for (int j = k >> 1; j > 0; j >>= 1) {
        for (int i = tid; i < P; i += nthreads) {
          const int ixj = i ^ j;
          if (ixj > i) {
            const bool want_forward = ((i & k) == 0);
            T va = vals[i];
            T vb = vals[ixj];
            if (sort_detail::should_swap_val<descending>(va, vb,
                                                         want_forward)) {
              vals[i] = vb;
              vals[ixj] = va;
            }
          }
        }
        __syncthreads();
      }
    }
  }
};

// In-place block-parallel bitonic key-value sort (1D, length P power of 2)
// with stable tie-breaking by smaller original index.
template <bool descending> struct BlockBitonicSortPairs1D {
  template <typename T, typename IdxT>
  static TL_DEVICE void run(T *__restrict__ vals, IdxT *__restrict__ idxs,
                            int valid_n, int P) {
    const int tid = sort_detail::linear_tid();
    const int nthreads = sort_detail::block_threads();
    const T sentinel = descending ? sort_detail::SortKeyTraits<T>::neg_sentinel()
                                  : sort_detail::SortKeyTraits<T>::pos_sentinel();
    for (int i = tid; i < P; i += nthreads) {
      if (i >= valid_n) {
        vals[i] = sentinel;
      }
      idxs[i] = static_cast<IdxT>(i);
    }
    __syncthreads();

    for (int k = 2; k <= P; k <<= 1) {
      for (int j = k >> 1; j > 0; j >>= 1) {
        for (int i = tid; i < P; i += nthreads) {
          const int ixj = i ^ j;
          if (ixj > i) {
            const bool want_forward = ((i & k) == 0);
            T va = vals[i];
            T vb = vals[ixj];
            IdxT ia = idxs[i];
            IdxT ib = idxs[ixj];
            if (sort_detail::should_swap_pair<descending>(va, ia, vb, ib,
                                                          want_forward)) {
              vals[i] = vb;
              vals[ixj] = va;
              idxs[i] = ib;
              idxs[ixj] = ia;
            }
          }
        }
        __syncthreads();
      }
    }
  }
};

// In-place block-parallel 2D bitonic sort along Axis (0 or 1).
// Buffer shape is [H, P] when Axis == 1 (padded width P, valid width valid_n)
// or [P, W] when Axis == 0 (padded height P, valid height valid_n).
template <int Axis, bool descending> struct BlockBitonicSort2D {
  static_assert(Axis == 0 || Axis == 1, "BlockBitonicSort2D Axis must be 0 or 1");

  template <typename T>
  static TL_DEVICE void run(T *__restrict__ vals, int outer_dim, int valid_n,
                            int P) {
    const int tid = sort_detail::linear_tid();
    const int nthreads = sort_detail::block_threads();
    const T sentinel = descending ? sort_detail::SortKeyTraits<T>::neg_sentinel()
                                  : sort_detail::SortKeyTraits<T>::pos_sentinel();
    const int total = outer_dim * P;

    for (int idx = tid; idx < total; idx += nthreads) {
      const int lane = (Axis == 1) ? (idx % P) : (idx / outer_dim);
      if (lane >= valid_n) {
        vals[idx] = sentinel;
      }
    }
    __syncthreads();

    for (int k = 2; k <= P; k <<= 1) {
      for (int j = k >> 1; j > 0; j >>= 1) {
        for (int idx = tid; idx < total; idx += nthreads) {
          int i, base_other, pos_a, pos_b;
          if constexpr (Axis == 1) {
            base_other = idx / P;
            i = idx - base_other * P;
            const int ixj = i ^ j;
            if (ixj <= i) continue;
            pos_a = base_other * P + i;
            pos_b = base_other * P + ixj;
          } else {
            i = idx / outer_dim;
            base_other = idx - i * outer_dim;
            const int ixj = i ^ j;
            if (ixj <= i) continue;
            pos_a = i * outer_dim + base_other;
            pos_b = ixj * outer_dim + base_other;
          }
          const bool want_forward = ((i & k) == 0);
          T va = vals[pos_a];
          T vb = vals[pos_b];
          if (sort_detail::should_swap_val<descending>(va, vb, want_forward)) {
            vals[pos_a] = vb;
            vals[pos_b] = va;
          }
        }
        __syncthreads();
      }
    }
  }
};

// In-place block-parallel 2D bitonic key-value sort along Axis (0 or 1).
template <int Axis, bool descending> struct BlockBitonicSortPairs2D {
  static_assert(Axis == 0 || Axis == 1,
                "BlockBitonicSortPairs2D Axis must be 0 or 1");

  template <typename T, typename IdxT>
  static TL_DEVICE void run(T *__restrict__ vals, IdxT *__restrict__ idxs,
                            int outer_dim, int valid_n, int P) {
    const int tid = sort_detail::linear_tid();
    const int nthreads = sort_detail::block_threads();
    const T sentinel = descending ? sort_detail::SortKeyTraits<T>::neg_sentinel()
                                  : sort_detail::SortKeyTraits<T>::pos_sentinel();
    const int total = outer_dim * P;

    for (int idx = tid; idx < total; idx += nthreads) {
      const int lane = (Axis == 1) ? (idx % P) : (idx / outer_dim);
      if (lane >= valid_n) {
        vals[idx] = sentinel;
      }
      idxs[idx] = static_cast<IdxT>(lane);
    }
    __syncthreads();

    for (int k = 2; k <= P; k <<= 1) {
      for (int j = k >> 1; j > 0; j >>= 1) {
        for (int idx = tid; idx < total; idx += nthreads) {
          int i, base_other, pos_a, pos_b;
          if constexpr (Axis == 1) {
            base_other = idx / P;
            i = idx - base_other * P;
            const int ixj = i ^ j;
            if (ixj <= i) continue;
            pos_a = base_other * P + i;
            pos_b = base_other * P + ixj;
          } else {
            i = idx / outer_dim;
            base_other = idx - i * outer_dim;
            const int ixj = i ^ j;
            if (ixj <= i) continue;
            pos_a = i * outer_dim + base_other;
            pos_b = ixj * outer_dim + base_other;
          }
          const bool want_forward = ((i & k) == 0);
          T va = vals[pos_a];
          T vb = vals[pos_b];
          IdxT ia = idxs[pos_a];
          IdxT ib = idxs[pos_b];
          if (sort_detail::should_swap_pair<descending>(va, ia, vb, ib,
                                                        want_forward)) {
            vals[pos_a] = vb;
            vals[pos_b] = va;
            idxs[pos_a] = ib;
            idxs[pos_b] = ia;
          }
        }
        __syncthreads();
      }
    }
  }
};

} // namespace tl
