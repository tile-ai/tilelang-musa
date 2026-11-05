#pragma once

#include <cstdint>

#include "common.h"

#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__) >= 310
#include "copy_mp31.h"
#endif

#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__) >= 220
#include <musa_robust.h>
#endif

namespace tl {

TL_DEVICE void cp_async_commit() {
  // #if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  //   __musa_memcpy_g2s_commit_group();
  // #endif
}

template <int N> TL_DEVICE void cp_async_wait() {
  // #if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  //   __musa_memcpy_g2s_wait_group(N);
  // #else
  __musa_memcpy_g2s_wait();
  // #endif
}

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr,
                           void const *global_ptr) {
  __musa_memcpy_g2s((void _AS3 *)smem_addr, (void const _AS1 *)global_ptr,
                    N /* total_bytes */, 0 /* prefetch_size */);
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *global_ptr, bool cond) {
  if (cond) {
    cp_async_gs<N>(smem_addr, global_ptr);
  } else {
    auto *smem_ptr = (uint8_t _AS3 *)smem_addr;
#pragma unroll
    for (int i = 0; i < N; ++i) {
      smem_ptr[i] = 0;
    }
  }
}

#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__) >= 220

// MP22 uses the legacy 3-word robust address, while MP31+ switches to the
// v4 robust descriptor form with prefetch-capable load intrinsics.
#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__) >= 310
using robust_desc_t = __musa::robust_v4_addr_t;
#define TL_GET_ROBUST_DESC(ROBUST_PTR) ((ROBUST_PTR).get_v4_robust_addr())
#define TL_ROBUST_LOAD_CALL(LENGTH_TYPE, ADDR, DESC)                           \
  __musa_ld_v4_robust_##LENGTH_TYPE((ADDR), (DESC), 0)
#define TL_ROBUST_MEMCPY_G2S(DST, SRC, BYTES, DESC)                            \
  __musa_memcpy_g2s_robust_v4((DST), (SRC), (BYTES), (DESC),                   \
                              0 /* prefetch_size */)
#else
using robust_desc_t = __musa::robust_addr_t;
#define TL_GET_ROBUST_DESC(ROBUST_PTR) ((ROBUST_PTR).get_robust_addr())
#define TL_ROBUST_LOAD_CALL(LENGTH_TYPE, ADDR, DESC)                           \
  __musa_ld_robust_##LENGTH_TYPE((ADDR), (DESC))
#define TL_ROBUST_MEMCPY_G2S(DST, SRC, BYTES, DESC)                            \
  __musa_memcpy_g2s_robust((DST), (SRC), (BYTES), (DESC), 0 /* prefetch_size   \
                                                             */)
#endif

template <typename To, typename From>
TL_DEVICE To robust_bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  union {
    From from;
    To to;
  } storage{from};
  return storage.to;
}

TL_DEVICE robust_desc_t make_robust_desc(void const *robust_base_ptr,
                                         uint64_t robust_size) {
  auto *typed_base =
      reinterpret_cast<int8_t *>(const_cast<void *>(robust_base_ptr));
  __musa::robust_ptr<int8_t> robust_ptr(typed_base, robust_size);
  return TL_GET_ROBUST_DESC(robust_ptr);
}

template <typename T>
TL_DEVICE T robust_load(void const *global_ptr, void const *robust_base_ptr,
                        uint64_t robust_size) {
  auto robust_desc = make_robust_desc(robust_base_ptr, robust_size);
  void const *addr = global_ptr;
  if constexpr (sizeof(T) == 1) {
    auto raw = TL_ROBUST_LOAD_CALL(i8, addr, robust_desc);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 2) {
    auto raw = TL_ROBUST_LOAD_CALL(i16, addr, robust_desc);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 4) {
    auto raw = TL_ROBUST_LOAD_CALL(i32, addr, robust_desc);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 8) {
    auto raw = TL_ROBUST_LOAD_CALL(i64, addr, robust_desc);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 12) {
    auto raw = TL_ROBUST_LOAD_CALL(v3i32, addr, robust_desc);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 16) {
    auto raw = TL_ROBUST_LOAD_CALL(v4i32, addr, robust_desc);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 32) {
    auto raw = TL_ROBUST_LOAD_CALL(v8i32, addr, robust_desc);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 64) {
    auto raw = TL_ROBUST_LOAD_CALL(v16i32, addr, robust_desc);
    return robust_bit_cast<T>(raw);
  } else if constexpr (sizeof(T) == 128) {
    auto raw = TL_ROBUST_LOAD_CALL(v32i32, addr, robust_desc);
    return robust_bit_cast<T>(raw);
  } else {
    static_assert(sizeof(T) <= 128, "Unsupported robust load width");
    static_assert(sizeof(T) > 128, "Unsupported robust load width");
  }
}

template <int N>
TL_DEVICE void
cp_async_gs_robust(void const *const smem_addr, void const *global_ptr,
                   void const *robust_base_ptr, uint64_t robust_size) {
  auto robust_desc = make_robust_desc(robust_base_ptr, robust_size);
  TL_ROBUST_MEMCPY_G2S((void _AS3 *)smem_addr, (void _AS1 *)global_ptr,
                       N /* total_bytes */, robust_desc);
}

template <int N>
TL_DEVICE void cp_async_gs_robust_conditional(void const *const smem_addr,
                                              void const *global_ptr,
                                              void const *robust_base_ptr,
                                              uint64_t robust_size, bool cond) {
  uint64_t selected_size = cond ? robust_size : 0;
  cp_async_gs_robust<N>(smem_addr, global_ptr, robust_base_ptr, selected_size);
}

#undef TL_ROBUST_MEMCPY_G2S
#undef TL_ROBUST_LOAD_CALL
#undef TL_GET_ROBUST_DESC

#endif

// Global memory load/store helpers used by T.ldg*/T.stg* codegen.
// Keep the implementation conservative: MUSA supports vectorized pointer
// load/store for these POD vector types, and alignment is the caller's
// responsibility.
TL_DEVICE uint32_t load_global_32(const void *ptr) {
  return *reinterpret_cast<const uint32_t *>(ptr);
}

TL_DEVICE uint2 load_global_64(const void *ptr) {
  return *reinterpret_cast<const uint2 *>(ptr);
}

TL_DEVICE uint4 load_global_128(const void *ptr) {
  return *reinterpret_cast<const uint4 *>(ptr);
}

TL_DEVICE ulonglong3 load_global_192(const void *ptr) {
  return *reinterpret_cast<const ulonglong3 *>(ptr);
}

TL_DEVICE longlong4 load_global_256(const longlong4 *ptr) { return *ptr; }

TL_DEVICE ulonglong4 load_global_256(const ulonglong4 *ptr) { return *ptr; }

template <typename T> TL_DEVICE ulonglong4 load_global_256(const T *ptr) {
  return *reinterpret_cast<const ulonglong4 *>(ptr);
}

TL_DEVICE uint32_t load_global_32_conditional(const void *ptr, bool pred) {
  return pred ? load_global_32(ptr) : uint32_t{};
}

TL_DEVICE uint2 load_global_64_conditional(const void *ptr, bool pred) {
  return pred ? load_global_64(ptr) : uint2{};
}

TL_DEVICE uint4 load_global_128_conditional(const void *ptr, bool pred) {
  return pred ? load_global_128(ptr) : uint4{};
}

TL_DEVICE longlong4 load_global_256_conditional(const longlong4 *ptr,
                                                bool pred) {
  return pred ? load_global_256(ptr) : longlong4{};
}

TL_DEVICE ulonglong4 load_global_256_conditional(const ulonglong4 *ptr,
                                                 bool pred) {
  return pred ? load_global_256(ptr) : ulonglong4{};
}

template <typename T>
TL_DEVICE ulonglong4 load_global_256_conditional(const T *ptr, bool pred) {
  return pred ? load_global_256(ptr) : ulonglong4{};
}

TL_DEVICE void store_global_32(void *ptr, uint32_t value) {
  *reinterpret_cast<uint32_t *>(ptr) = value;
}

TL_DEVICE void store_global_64(void *ptr, uint2 value) {
  *reinterpret_cast<uint2 *>(ptr) = value;
}

TL_DEVICE void store_global_128(void *ptr, uint4 value) {
  *reinterpret_cast<uint4 *>(ptr) = value;
}

TL_DEVICE void store_global_192(void *ptr, const ulonglong3 &value) {
  *reinterpret_cast<ulonglong3 *>(ptr) = value;
}

TL_DEVICE void store_global_256(void *ptr, const longlong4 &value) {
  *reinterpret_cast<longlong4 *>(ptr) = value;
}

TL_DEVICE void store_global_256(void *ptr, const ulonglong4 &value) {
  *reinterpret_cast<ulonglong4 *>(ptr) = value;
}

template <typename T>
TL_DEVICE void store_global_256(void *ptr, const T &value) {
  const ulonglong4 &value_u64 = *reinterpret_cast<const ulonglong4 *>(&value);
  *reinterpret_cast<ulonglong4 *>(ptr) = value_u64;
}

TL_DEVICE void store_global_32_conditional(void *ptr, uint32_t value,
                                           bool pred) {
  if (pred) {
    store_global_32(ptr, value);
  }
}

TL_DEVICE void store_global_64_conditional(void *ptr, uint2 value, bool pred) {
  if (pred) {
    store_global_64(ptr, value);
  }
}

TL_DEVICE void store_global_128_conditional(void *ptr, uint4 value, bool pred) {
  if (pred) {
    store_global_128(ptr, value);
  }
}

TL_DEVICE void store_global_256_conditional(void *ptr, const longlong4 &value,
                                            bool pred) {
  if (pred) {
    store_global_256(ptr, value);
  }
}

TL_DEVICE void store_global_256_conditional(void *ptr, const ulonglong4 &value,
                                            bool pred) {
  if (pred) {
    store_global_256(ptr, value);
  }
}

template <typename T>
TL_DEVICE void store_global_256_conditional(void *ptr, const T &value,
                                            bool pred) {
  if (pred) {
    store_global_256(ptr, value);
  }
}

} // namespace tl
