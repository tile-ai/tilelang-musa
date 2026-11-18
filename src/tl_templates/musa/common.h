#pragma once

#include "atomic.h"
#include <cstdio> // snprintf
#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_runtime.h>
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
#include <mute/arch/simd_mp31.hpp>
#endif
#include <mutlass/fast_math.h>
#include <mutlass/numeric_types.h>
#include <sstream> // std::stringstream

using uint = unsigned int;
using uchar = unsigned char;
using ushort = unsigned short;

using mutlass::bfloat16_t;
using mutlass::half_t;
using mutlass::tfloat32_t;
using int4_t = int4;

// Keep half-vector lanes as _Float16. Using the raw unsigned-short field
// forces MCC to insert per-lane pack/unpack instructions around SIMD ops.
using tl_h_elem_t = decltype(__half_raw{}.data);
using tl_bf_elem_t = decltype(__mt_bfloat16_raw{}.data);
using tl_h2 = tl_h_elem_t __attribute__((ext_vector_type(2)));
using tl_h4 = tl_h_elem_t __attribute__((ext_vector_type(4)));
using tl_h8 = tl_h_elem_t __attribute__((ext_vector_type(8)));
using tl_bf2 = tl_bf_elem_t __attribute__((ext_vector_type(2)));
using tl_bf4 = tl_bf_elem_t __attribute__((ext_vector_type(4)));
using tl_bf8 = tl_bf_elem_t __attribute__((ext_vector_type(8)));
using tl_f2 = float __attribute__((ext_vector_type(2)));
using tl_f4 = float __attribute__((ext_vector_type(4)));

using v2i32_t = int32_t __attribute__((vector_size(8)));
using v3i32_t = int32_t __attribute__((vector_size(12)));
using v4i32_t = int32_t __attribute__((vector_size(16)));
using v5i32_t = int32_t __attribute__((vector_size(20)));
using v8i32_t = int32_t __attribute__((vector_size(32)));
using v16i32_t = int32_t __attribute__((vector_size(64)));
using v32i32_t = int32_t __attribute__((vector_size(128)));

#define __log2 log2
#define __log2f log2f

#define hexp mutlass::fast_exp
#define hlog mutlass::fast_log
#define hsqrt mutlass::fast_sqrt
#define hsin mutlass::fast_sin
#define hcos mutlass::fast_cos
#define htanh mutlass::fast_tanh
#define hpow powf

#define TL_HOST_DEVICE __forceinline__ __host__ __device__
#define TL_DEVICE __forceinline__ __device__
#define TL_DEVICE_NOINLINE __noinline__ __device__
#define TL_PATCH

#define _AS1 __attribute__((address_space(1)))
#define _AS3 __attribute__((address_space(3)))

namespace tl {

// Use fields prepared outside the kernel when the divisor is launch-invariant.
TL_DEVICE int fast_div(int dividend, int divisor, uint32_t multiplier,
                       uint32_t shift_right) {
  return divisor != 1
             ? static_cast<int>(
                   __umulhi(static_cast<uint32_t>(dividend), multiplier) >>
                   shift_right)
             : dividend;
}

TL_DEVICE int fast_mod(int dividend, int divisor, uint32_t multiplier,
                       uint32_t shift_right) {
  int quotient = fast_div(dividend, divisor, multiplier, shift_right);
  return dividend - quotient * divisor;
}

// Preserve mutlass::FastDivmod semantics when the divisor is only known on the
// device.  Construction prepares the multiplier and shift for this value.
TL_DEVICE int fast_div(int dividend, int divisor) {
  return mutlass::FastDivmod(divisor).div(dividend);
}

TL_DEVICE int fast_mod(int dividend, int divisor) {
  int quotient;
  int remainder;
  mutlass::FastDivmod fast_divmod(divisor);
  fast_divmod(quotient, remainder, dividend);
  return remainder;
}

} // namespace tl

TL_DEVICE tl_h_elem_t make_tl_h_elem(half_t x) {
  return static_cast<__half_raw>(x.to_half()).data;
}

template <typename To, typename From>
TL_DEVICE To tl_bit_cast(const From &from) {
  static_assert(sizeof(To) == sizeof(From));
  union {
    From from;
    To to;
  } storage{from};
  return storage.to;
}

// MCC can incorrectly expand a vector of repeated FP32-to-FP16 conversions
// when the packed destination aliases the scalar source. Convert once and
// replicate the FP16 bits as 32-bit pairs.
TL_DEVICE uint32_t make_tl_h_pair_bits(half_t x) {
  uint32_t bits = static_cast<uint32_t>(x.raw());
  return bits | (bits << 16);
}

TL_DEVICE tl_h2 make_tl_h2_splat(half_t x) {
  uint32_t pair = make_tl_h_pair_bits(x);
  return tl_bit_cast<tl_h2>(pair);
}

TL_DEVICE tl_h4 make_tl_h4_splat(half_t x) {
  int32_t pair = static_cast<int32_t>(make_tl_h_pair_bits(x));
  v2i32_t packed = {pair, pair};
  return tl_bit_cast<tl_h4>(packed);
}

TL_DEVICE tl_h8 make_tl_h8_splat(half_t x) {
  int32_t pair = static_cast<int32_t>(make_tl_h_pair_bits(x));
  v4i32_t packed = {pair, pair, pair, pair};
  return tl_bit_cast<tl_h8>(packed);
}

TL_DEVICE tl_bf_elem_t make_tl_bf_elem(bfloat16_t x) {
  return *((tl_bf_elem_t *)&x);
}

TL_DEVICE half_t tl_h_elem_to_half(tl_h_elem_t x) {
  return half_t(__half(__half_raw{x}));
}

TL_DEVICE bfloat16_t tl_bf_elem_to_bfloat16(tl_bf_elem_t x) {
  __mt_bfloat16_raw raw;
  raw.data = x;
  return bfloat16_t(static_cast<float>(__mt_bfloat16(raw)));
}

TL_DEVICE tl_h2 make_tl_h2(tl_h_elem_t x0, tl_h_elem_t x1) {
  return tl_h2{x0, x1};
}

TL_DEVICE tl_h2 make_tl_h2(half_t x0, half_t x1) {
  return make_tl_h2(make_tl_h_elem(x0), make_tl_h_elem(x1));
}

TL_DEVICE tl_h4 make_tl_h4(tl_h_elem_t x0, tl_h_elem_t x1, tl_h_elem_t x2,
                           tl_h_elem_t x3) {
  return tl_h4{x0, x1, x2, x3};
}

TL_DEVICE tl_h4 make_tl_h4(half_t x0, half_t x1, half_t x2, half_t x3) {
  return make_tl_h4(make_tl_h_elem(x0), make_tl_h_elem(x1), make_tl_h_elem(x2),
                    make_tl_h_elem(x3));
}

TL_DEVICE tl_h8 make_tl_h8(tl_h_elem_t x0, tl_h_elem_t x1, tl_h_elem_t x2,
                           tl_h_elem_t x3, tl_h_elem_t x4, tl_h_elem_t x5,
                           tl_h_elem_t x6, tl_h_elem_t x7) {
  return tl_h8{x0, x1, x2, x3, x4, x5, x6, x7};
}

TL_DEVICE tl_h8 make_tl_h8(half_t x0, half_t x1, half_t x2, half_t x3,
                           half_t x4, half_t x5, half_t x6, half_t x7) {
  return make_tl_h8(make_tl_h_elem(x0), make_tl_h_elem(x1), make_tl_h_elem(x2),
                    make_tl_h_elem(x3), make_tl_h_elem(x4), make_tl_h_elem(x5),
                    make_tl_h_elem(x6), make_tl_h_elem(x7));
}

TL_DEVICE tl_bf2 make_tl_bf2(tl_bf_elem_t x0, tl_bf_elem_t x1) {
  return tl_bf2{x0, x1};
}

TL_DEVICE tl_bf2 make_tl_bf2(bfloat16_t x0, bfloat16_t x1) {
  return make_tl_bf2(make_tl_bf_elem(x0), make_tl_bf_elem(x1));
}

TL_DEVICE tl_bf4 make_tl_bf4(tl_bf_elem_t x0, tl_bf_elem_t x1, tl_bf_elem_t x2,
                             tl_bf_elem_t x3) {
  return tl_bf4{x0, x1, x2, x3};
}

TL_DEVICE tl_bf4 make_tl_bf4(bfloat16_t x0, bfloat16_t x1, bfloat16_t x2,
                             bfloat16_t x3) {
  return make_tl_bf4(make_tl_bf_elem(x0), make_tl_bf_elem(x1),
                     make_tl_bf_elem(x2), make_tl_bf_elem(x3));
}

TL_DEVICE tl_bf8 make_tl_bf8(tl_bf_elem_t x0, tl_bf_elem_t x1, tl_bf_elem_t x2,
                             tl_bf_elem_t x3, tl_bf_elem_t x4, tl_bf_elem_t x5,
                             tl_bf_elem_t x6, tl_bf_elem_t x7) {
  return tl_bf8{x0, x1, x2, x3, x4, x5, x6, x7};
}

TL_DEVICE tl_bf8 make_tl_bf8(bfloat16_t x0, bfloat16_t x1, bfloat16_t x2,
                             bfloat16_t x3, bfloat16_t x4, bfloat16_t x5,
                             bfloat16_t x6, bfloat16_t x7) {
  return make_tl_bf8(make_tl_bf_elem(x0), make_tl_bf_elem(x1),
                     make_tl_bf_elem(x2), make_tl_bf_elem(x3),
                     make_tl_bf_elem(x4), make_tl_bf_elem(x5),
                     make_tl_bf_elem(x6), make_tl_bf_elem(x7));
}

TL_DEVICE tl_f2 make_tl_f2(float x0, float x1) { return tl_f2{x0, x1}; }

TL_DEVICE tl_f4 make_tl_f4(float x0, float x1, float x2, float x3) {
  return tl_f4{x0, x1, x2, x3};
}

#define TILELANG_CHECK(stmt)                                                   \
  do {                                                                         \
    musaError_t __err = (stmt);                                                \
    if (__err != musaSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, "%s:%d: %s - %s", __FILE__,          \
               __LINE__, musaGetErrorName(__err), musaGetErrorString(__err));  \
      return -1;                                                               \
    }                                                                          \
  } while (0)

#define TILELANG_CHECK_LAST_ERROR(kernel_name)                                 \
  do {                                                                         \
    musaError_t __err = musaGetLastError();                                    \
    if (__err != musaSuccess) {                                                \
      snprintf(error_buf, ERROR_BUF_SIZE, kernel_name ": %s - %s",             \
               musaGetErrorName(__err), musaGetErrorString(__err));            \
      return -1;                                                               \
    }                                                                          \
  } while (0)

static const int NumThreadsPerWarpBeforeMP31 = 128;
static const int NumThreadsPerWarp = 32;
static const int NumThreadsPerWarpSquad = 128;
static const int NumWarpsPerWarpSquad =
    NumThreadsPerWarpSquad / NumThreadsPerWarp;
static const int NumThreadsPerHalfWarp = NumThreadsPerWarp / 2;

enum class SmemSwizzleGranularity : uint8_t {
  NONE = 0,
  B16 = 1,
  B32 = 2,
  B64 = 3,
};

enum class SmemSwizzleStride : uint8_t {
  B32 = 0,
  B64 = 1,
  B128 = 2,
  B256 = 3,
};

enum class SmemSwizzleLine : uint8_t {
  B128 = 0,
  B256 = 1,
};

enum class CacheHint : uint8_t {
  CACHE_NONE = 0,
  CACHE_ONCE = 1,
  CACHE_NORMAL = 2,
  CACHE_PERSIST = 3,
};

// MUSA 5.3.0 changed the block TME load/store intrinsics to take an immediate
// offset before the cache-control arguments.
#if defined(__MUSACC_VER_MAJOR__) &&                                           \
    (__MUSACC_VER_MAJOR__ > 5 ||                                               \
     (__MUSACC_VER_MAJOR__ == 5 && __MUSACC_VER_MINOR__ >= 3))
#define TL_MUSA_HAS_TME_BLOCK_IMM_OFFSET 1
#else
#define TL_MUSA_HAS_TME_BLOCK_IMM_OFFSET 0
#endif


// Wrapper for the MTCC LSU load cache-hint intrinsic. The compiler intrinsic
// is available on MP31 only; keeping policy values as template arguments
// guarantees that MTCC receives immediate operands.
namespace tl {

template <int InnerPersistence = 4, int OuterPersistence = 2, int Chrnt = 0,
          int L2CachePolicy = 0, bool IsVolatile = false, typename T>
TL_DEVICE T lsu_ld_cache_hint(const T *ptr) {
  static_assert(InnerPersistence >= 0 && InnerPersistence <= 5,
                "invalid LSU inner persistence hint");
  static_assert(OuterPersistence >= 0 && OuterPersistence <= 3,
                "invalid LSU outer persistence hint");
  static_assert(Chrnt >= 0 && Chrnt <= 1, "invalid LSU chrnt hint");
  static_assert(L2CachePolicy >= 0 && L2CachePolicy <= 1,
                "invalid LSU L2 cache policy");
#if !defined(__MUSA_ARCH__) || __MUSA_ARCH__ == 310
#if defined(__has_builtin)
#if __has_builtin(__musa_lsu_ld_volatile_cache_hint_i)
  return __lsu_ld_cache_hint(ptr, InnerPersistence, OuterPersistence, Chrnt,
                             L2CachePolicy, IsVolatile);
#else
  static_assert(!IsVolatile, "volatile LSU loads require compiler builtin support");
  return __lsu_ld_cache_hint(ptr, InnerPersistence, OuterPersistence, Chrnt,
                             L2CachePolicy);
#endif
#else
  static_assert(!IsVolatile, "volatile LSU loads require compiler builtin support");
  return __lsu_ld_cache_hint(ptr, InnerPersistence, OuterPersistence, Chrnt,
                             L2CachePolicy);
#endif
#else
  static_assert(InnerPersistence < 0,
                "tl::lsu_ld_cache_hint is supported on MP31 only");
  return *ptr;
#endif
}

} // namespace tl

enum class PrefetchSize : uint8_t {
  NONE = 0,
  B64 = 64,
  B128 = 128,
};

enum class AddressSpace {
  Generic = 0,
  Global = 1,
  Shared = 3,
};

template <AddressSpace AS>
    TL_HOST_DEVICE constexpr void
    __attribute__((address_space(static_cast<int>(AS)))) *
    make_ptr_with_address_space(uint64_t ptr) {
  return reinterpret_cast<void __attribute__((
      address_space(static_cast<int>(AS)))) *>(ptr);
}

/// MUTE helper to cast SMEM pointer to unsigned
TL_HOST_DEVICE
uint32_t cast_smem_ptr_to_uint(void const *const ptr) {
  /// MUTE helper to get SMEM pointer
  return static_cast<uint32_t>(__cvta_generic_to_shared(ptr));
}

// using mutlass abs function for half_t
TL_PATCH TL_DEVICE half_t __habs(const half_t x) { return abs(x); }

// using mutlass abs function for bfloat_t
TL_PATCH TL_DEVICE bfloat16_t __habs(const bfloat16_t x) { return abs(x); }

// hrsqrt function for half_t
TL_PATCH TL_DEVICE half_t hrsqrt(const half_t x) {
  return half_t(hrsqrt(x.to_half()));
}

// Pack two half values.
TL_DEVICE unsigned __pack_half2(const half x, const half y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two half_t values.
TL_DEVICE unsigned __pack_half2(const half_t x, const half_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_half2(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack two bfloat16_t values.
TL_DEVICE unsigned __pack_mt_bfloat162(const bfloat16_t x, const bfloat16_t y) {
  unsigned v0 = *((unsigned short *)&x);
  unsigned v1 = *((unsigned short *)&y);
  return (v1 << 16) | v0;
}

// Pack four char values. Build the 32-bit pattern from unsigned bytes: a
// negative signed char would otherwise sign-extend and flood the other lanes
// through the OR.
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  const unsigned int b0 = static_cast<unsigned char>(x0);
  const unsigned int b1 = static_cast<unsigned char>(x1);
  const unsigned int b2 = static_cast<unsigned char>(x2);
  const unsigned int b3 = static_cast<unsigned char>(x3);
  return static_cast<int>((b3 << 24) | (b2 << 16) | (b1 << 8) | b0);
}

// Pack eight char values.
TL_DEVICE int2 make_int2(signed char x0, signed char x1, signed char x2,
                         signed char x3, signed char y0, signed char y1,
                         signed char y2, signed char y3) {
  int2 result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  return result;
}

// Pack sixteen char values.
TL_DEVICE int4_t make_int4(signed char x0, signed char x1, signed char x2,
                           signed char x3, signed char y0, signed char y1,
                           signed char y2, signed char y3, signed char z0,
                           signed char z1, signed char z2, signed char z3,
                           signed char w0, signed char w1, signed char w2,
                           signed char w3) {
  int4_t result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  result.z = make_int(z0, z1, z2, z3);
  result.w = make_int(w0, w1, w2, w3);
  return result;
}

// Pack eight short values.
TL_DEVICE int4_t make_int4(short x0, short x1, short y0, short y1, short z0,
                           short z1, short w0, short w1) {
  int4_t result;
  *((short2 *)&result.x) = make_short2(x0, x1);
  *((short2 *)&result.y) = make_short2(y0, y1);
  *((short2 *)&result.z) = make_short2(z0, z1);
  *((short2 *)&result.w) = make_short2(w0, w1);
  return result;
}

// Pack four unsigned char values.
TL_DEVICE unsigned int make_uint(unsigned char x0, unsigned char x1,
                                 unsigned char x2, unsigned char x3) {
  return (x3 << 24) | (x2 << 16) | (x1 << 8) | x0;
}

// Pack eight unsigned char values.
TL_DEVICE uint2 make_uint2(unsigned char x0, unsigned char x1, unsigned char x2,
                           unsigned char x3, unsigned char y0, unsigned char y1,
                           unsigned char y2, unsigned char y3) {
  uint2 result;
  result.x = make_uint(x0, x1, x2, x3);
  result.y = make_uint(y0, y1, y2, y3);
  return result;
}

// Pack sixteen unsigned char values.
TL_DEVICE uint4 make_uint4(unsigned char x0, unsigned char x1, unsigned char x2,
                           unsigned char x3, unsigned char y0, unsigned char y1,
                           unsigned char y2, unsigned char y3, unsigned char z0,
                           unsigned char z1, unsigned char z2, unsigned char z3,
                           unsigned char w0, unsigned char w1, unsigned char w2,
                           unsigned char w3) {
  uint4 result;
  result.x = make_uint(x0, x1, x2, x3);
  result.y = make_uint(y0, y1, y2, y3);
  result.z = make_uint(z0, z1, z2, z3);
  result.w = make_uint(w0, w1, w2, w3);
  return result;
}

// Pack eight unsigned short values.
TL_DEVICE uint4 make_uint4(unsigned short x0, unsigned short x1,
                           unsigned short y0, unsigned short y1,
                           unsigned short z0, unsigned short z1,
                           unsigned short w0, unsigned short w1) {
  uint4 result;
  *((ushort2 *)&result.x) = make_ushort2(x0, x1);
  *((ushort2 *)&result.y) = make_ushort2(y0, y1);
  *((ushort2 *)&result.z) = make_ushort2(z0, z1);
  *((ushort2 *)&result.w) = make_ushort2(w0, w1);
  return result;
}

// Pack eight int values.
TL_DEVICE longlong4 make_longlong4(int x0, int x1, int y0, int y1, int z0,
                                   int z1, int w0, int w1) {
  longlong4 result;
  *((int2 *)&result.x) = make_int2(x0, x1);
  *((int2 *)&result.y) = make_int2(y0, y1);
  *((int2 *)&result.z) = make_int2(z0, z1);
  *((int2 *)&result.w) = make_int2(w0, w1);
  return result;
}

// DP4A
template <typename InDatatype, typename OutDatatype>
TL_DEVICE void DP4A(InDatatype *a, InDatatype *b, OutDatatype *c) {
  const int c_int = *((int *)c);
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  const int a_int = *((int *)a);
  const int b_int = *((int *)b);
  *c = __dp4a(a_int, b_int, c_int);
#else
  const char4 a_vec = *(char4 *)a;
  const char4 b_vec = *(char4 *)b;
  *c = __dp4a(a_vec, b_vec, c_int);
#endif
}

namespace tl {

enum class DataType : int {
  kInt4 = 0,
  kUInt4 = 1,
  kInt8 = 2,
  kUInt8 = 3,
  kInt16 = 4,
  kUInt16 = 5,
  kInt32 = 6,
  kUInt32 = 7,
  kInt64 = 8,
  kUInt64 = 9,
  kFloat8_e4m3 = 10,
  kFloat8_e5m2 = 11,
  kFloat16 = 12,
  kBFloat16 = 13,
  kFloat16x2 = 14,
  kFloat32 = 15,
  kTensorFloat32 = 16,
  kFloat64 = 17,
  kBit1 = 18,
  kBit8 = 19,
  kBit16 = 20,
  kBit32 = 21,
  kBit64 = 22
};

// Any
template <typename T> TL_DEVICE bool Any(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (a[i]) {
      return true;
    }
  }
  return false;
}

// All
template <typename T> TL_DEVICE bool All(T *a, int size) {
  for (int i = 0; i < size; i++) {
    if (!a[i]) {
      return false;
    }
  }
  return true;
}

// Pow of int
template <int y = 1, typename T> TL_DEVICE T pow_of_int(T x) {
  T result = x;
  for (int i = 1; i < y; i++) {
    result *= x;
  }
  return result;
}

struct float_e4m3_t : public mutlass::float_e4m3_t {
  using mutlass::float_e4m3_t::float_e4m3_t;

  TL_HOST_DEVICE
  float_e4m3_t() = default;

  TL_HOST_DEVICE
  float_e4m3_t(mutlass::float_e4m3_t x) : mutlass::float_e4m3_t(x) {}

  TL_HOST_DEVICE
  explicit float_e4m3_t(__mt_bfloat16 x)
      : float_e4m3_t(static_cast<float>(x)) {}

  TL_HOST_DEVICE
  float_e4m3_t &operator=(mutlass::float_e4m3_t x) {
    mutlass::float_e4m3_t::operator=(x);
    return *this;
  }
};

struct float_e5m2_t : public mutlass::float_e5m2_t {
  using mutlass::float_e5m2_t::float_e5m2_t;

  TL_HOST_DEVICE
  float_e5m2_t() = default;

  TL_HOST_DEVICE
  float_e5m2_t(mutlass::float_e5m2_t x) : mutlass::float_e5m2_t(x) {}

  TL_HOST_DEVICE
  explicit float_e5m2_t(__mt_bfloat16 x)
      : float_e5m2_t(static_cast<float>(x)) {}

  TL_HOST_DEVICE
  float_e5m2_t &operator=(mutlass::float_e5m2_t x) {
    mutlass::float_e5m2_t::operator=(x);
    return *this;
  }
};

template <typename T> struct to_mute_type {
  using type = T;
};

template <> struct to_mute_type<tl::float_e4m3_t> {
  using type = mutlass::float_e4m3_t;
};
template <> struct to_mute_type<tl::float_e5m2_t> {
  using type = mutlass::float_e5m2_t;
};

// Generic passthroughs
template <typename T>
TL_DEVICE T shfl_xor_sync(unsigned mask, T val, int laneMask) {
  return __shfl_xor_sync(mask, val, laneMask);
}

TL_DEVICE float2 shfl_xor_sync(unsigned mask, float2 val, int laneMask) {
  float2 out;
  out.x = __shfl_xor_sync(mask, val.x, laneMask);
  out.y = __shfl_xor_sync(mask, val.y, laneMask);
  return out;
}

TL_DEVICE tl_f2 shfl_xor_sync(unsigned mask, tl_f2 val, int laneMask) {
  return tl_f2{__shfl_xor_sync(mask, val[0], laneMask),
               __shfl_xor_sync(mask, val[1], laneMask)};
}

TL_DEVICE float4 shfl_xor_sync(unsigned mask, float4 val, int laneMask) {
  float4 out;
  out.x = __shfl_xor_sync(mask, val.x, laneMask);
  out.y = __shfl_xor_sync(mask, val.y, laneMask);
  out.z = __shfl_xor_sync(mask, val.z, laneMask);
  out.w = __shfl_xor_sync(mask, val.w, laneMask);
  return out;
}

TL_DEVICE tl_f4 shfl_xor_sync(unsigned mask, tl_f4 val, int laneMask) {
  return tl_f4{__shfl_xor_sync(mask, val[0], laneMask),
               __shfl_xor_sync(mask, val[1], laneMask),
               __shfl_xor_sync(mask, val[2], laneMask),
               __shfl_xor_sync(mask, val[3], laneMask)};
}

template <typename T> TL_DEVICE uint32_t packed_shuffle_to_uint32(T val) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  return reinterpret_cast<uint32_t &>(val);
}

template <typename T> TL_DEVICE T packed_shuffle_from_uint32(uint32_t val) {
  static_assert(sizeof(T) == sizeof(uint32_t));
  return reinterpret_cast<T &>(val);
}

TL_DEVICE tl_h2 shfl_xor_sync(unsigned mask, tl_h2 val, int laneMask) {
  uint32_t raw = packed_shuffle_to_uint32(val);
  raw = __shfl_xor_sync(mask, raw, laneMask);
  return packed_shuffle_from_uint32<tl_h2>(raw);
}

TL_DEVICE tl_bf2 shfl_xor_sync(unsigned mask, tl_bf2 val, int laneMask) {
  uint32_t raw = packed_shuffle_to_uint32(val);
  raw = __shfl_xor_sync(mask, raw, laneMask);
  return packed_shuffle_from_uint32<tl_bf2>(raw);
}

template <typename T>
TL_DEVICE T shfl_down_sync(unsigned mask, T val, int delta) {
  return __shfl_down_sync(mask, val, delta);
}

TL_DEVICE tl_f2 shfl_down_sync(unsigned mask, tl_f2 val, int delta) {
  return tl_f2{__shfl_down_sync(mask, val[0], delta),
               __shfl_down_sync(mask, val[1], delta)};
}

TL_DEVICE tl_f4 shfl_down_sync(unsigned mask, tl_f4 val, int delta) {
  return tl_f4{__shfl_down_sync(mask, val[0], delta),
               __shfl_down_sync(mask, val[1], delta),
               __shfl_down_sync(mask, val[2], delta),
               __shfl_down_sync(mask, val[3], delta)};
}

TL_DEVICE tl_h2 shfl_down_sync(unsigned mask, tl_h2 val, int delta) {
  uint32_t raw = packed_shuffle_to_uint32(val);
  raw = __shfl_down_sync(mask, raw, delta);
  return packed_shuffle_from_uint32<tl_h2>(raw);
}

TL_DEVICE tl_bf2 shfl_down_sync(unsigned mask, tl_bf2 val, int delta) {
  uint32_t raw = packed_shuffle_to_uint32(val);
  raw = __shfl_down_sync(mask, raw, delta);
  return packed_shuffle_from_uint32<tl_bf2>(raw);
}

template <typename T>
TL_DEVICE T shfl_up_sync(unsigned mask, T val, int delta) {
  return __shfl_up_sync(mask, val, delta);
}

TL_DEVICE tl_f2 shfl_up_sync(unsigned mask, tl_f2 val, int delta) {
  return tl_f2{__shfl_up_sync(mask, val[0], delta),
               __shfl_up_sync(mask, val[1], delta)};
}

TL_DEVICE tl_f4 shfl_up_sync(unsigned mask, tl_f4 val, int delta) {
  return tl_f4{
      __shfl_up_sync(mask, val[0], delta), __shfl_up_sync(mask, val[1], delta),
      __shfl_up_sync(mask, val[2], delta), __shfl_up_sync(mask, val[3], delta)};
}

TL_DEVICE tl_h2 shfl_up_sync(unsigned mask, tl_h2 val, int delta) {
  uint32_t raw = packed_shuffle_to_uint32(val);
  raw = __shfl_up_sync(mask, raw, delta);
  return packed_shuffle_from_uint32<tl_h2>(raw);
}

TL_DEVICE tl_bf2 shfl_up_sync(unsigned mask, tl_bf2 val, int delta) {
  uint32_t raw = packed_shuffle_to_uint32(val);
  raw = __shfl_up_sync(mask, raw, delta);
  return packed_shuffle_from_uint32<tl_bf2>(raw);
}

template <typename T> TL_DEVICE T shfl_sync(unsigned mask, T val, int srcLane) {
  return __shfl_sync(mask, val, srcLane);
}

TL_DEVICE tl_f2 shfl_sync(unsigned mask, tl_f2 val, int srcLane) {
  return tl_f2{__shfl_sync(mask, val[0], srcLane),
               __shfl_sync(mask, val[1], srcLane)};
}

TL_DEVICE tl_f4 shfl_sync(unsigned mask, tl_f4 val, int srcLane) {
  return tl_f4{
      __shfl_sync(mask, val[0], srcLane), __shfl_sync(mask, val[1], srcLane),
      __shfl_sync(mask, val[2], srcLane), __shfl_sync(mask, val[3], srcLane)};
}

TL_DEVICE tl_h2 shfl_sync(unsigned mask, tl_h2 val, int srcLane) {
  uint32_t raw = packed_shuffle_to_uint32(val);
  raw = __shfl_sync(mask, raw, srcLane);
  return packed_shuffle_from_uint32<tl_h2>(raw);
}

TL_DEVICE tl_bf2 shfl_sync(unsigned mask, tl_bf2 val, int srcLane) {
  uint32_t raw = packed_shuffle_to_uint32(val);
  raw = __shfl_sync(mask, raw, srcLane);
  return packed_shuffle_from_uint32<tl_bf2>(raw);
}

// Specializations for mutlass::half_t
template <>
TL_DEVICE half_t shfl_xor_sync(unsigned mask, half_t val, int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor_sync(mask, f, laneMask);
  return half_t(r);
}

template <>
TL_DEVICE half_t shfl_down_sync(unsigned mask, half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down_sync(mask, f, delta);
  return half_t(r);
}

template <>
TL_DEVICE half_t shfl_up_sync(unsigned mask, half_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up_sync(mask, f, delta);
  return half_t(r);
}

template <> TL_DEVICE half_t shfl_sync(unsigned mask, half_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl_sync(mask, f, srcLane);
  return half_t(r);
}

// Specializations for mutlass::bfloat16_t
template <>
TL_DEVICE bfloat16_t shfl_xor_sync(unsigned mask, bfloat16_t val,
                                   int laneMask) {
  float f = static_cast<float>(val);
  float r = __shfl_xor_sync(mask, f, laneMask);
  return bfloat16_t(r);
}

template <>
TL_DEVICE bfloat16_t shfl_down_sync(unsigned mask, bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_down_sync(mask, f, delta);
  return bfloat16_t(r);
}

template <>
TL_DEVICE bfloat16_t shfl_up_sync(unsigned mask, bfloat16_t val, int delta) {
  float f = static_cast<float>(val);
  float r = __shfl_up_sync(mask, f, delta);
  return bfloat16_t(r);
}

template <>
TL_DEVICE bfloat16_t shfl_sync(unsigned mask, bfloat16_t val, int srcLane) {
  float f = static_cast<float>(val);
  float r = __shfl_sync(mask, f, srcLane);
  return bfloat16_t(r);
}

TL_DEVICE float2 vec_max_f2(float2 a, float2 b) {
  float2 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  mute::max(out, a, b);
#else
  out.x = max(a.x, b.x);
  out.y = max(a.y, b.y);
#endif
  return out;
}

TL_DEVICE tl_f2 vec_max_f2(tl_f2 a, tl_f2 b) {
  tl_f2 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  float2 a_float2 = {a[0], a[1]};
  float2 b_float2 = {b[0], b[1]};
  float2 out_float2;
  mute::max(out_float2, a_float2, b_float2);
  out = tl_f2{out_float2.x, out_float2.y};
#else
  out = tl_f2{max(a[0], b[0]), max(a[1], b[1])};
#endif
  return out;
}

TL_DEVICE float4 vec_max_f4(float4 a, float4 b) {
  float4 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  mute::max(out, a, b);
#else
  out.x = max(a.x, b.x);
  out.y = max(a.y, b.y);
  out.z = max(a.z, b.z);
  out.w = max(a.w, b.w);
#endif
  return out;
}

TL_DEVICE tl_f4 vec_max_f4(tl_f4 a, tl_f4 b) {
  tl_f4 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  float4 a_float4 = {a[0], a[1], a[2], a[3]};
  float4 b_float4 = {b[0], b[1], b[2], b[3]};
  float4 out_float4;
  mute::max(out_float4, a_float4, b_float4);
  out = tl_f4{out_float4.x, out_float4.y, out_float4.z, out_float4.w};
#else
  out =
      tl_f4{max(a[0], b[0]), max(a[1], b[1]), max(a[2], b[2]), max(a[3], b[3])};
#endif
  return out;
}

TL_DEVICE float2 vec_sum_f2(float2 a, float2 b) {
  float2 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  mute::add(out, a, b);
#else
  out.x = a.x + b.x;
  out.y = a.y + b.y;
#endif
  return out;
}

TL_DEVICE tl_f2 vec_sum_f2(tl_f2 a, tl_f2 b) {
  tl_f2 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  float2 a_float2 = {a[0], a[1]};
  float2 b_float2 = {b[0], b[1]};
  float2 out_float2;
  mute::add(out_float2, a_float2, b_float2);
  out = tl_f2{out_float2.x, out_float2.y};
#else
  out = tl_f2{a[0] + b[0], a[1] + b[1]};
#endif
  return out;
}

TL_DEVICE float4 vec_sum_f4(float4 a, float4 b) {
  float4 out;
  out.x = a.x + b.x;
  out.y = a.y + b.y;
  out.z = a.z + b.z;
  out.w = a.w + b.w;
  // todo: check bug
  // mute::add(out, a, b);
  return out;
}

TL_DEVICE tl_f4 vec_sum_f4(tl_f4 a, tl_f4 b) {
  return tl_f4{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
}

TL_DEVICE float2 vec_exp2_f2(float2 a) {
  float2 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  mute::fast_exp2(out, a);
#else
  out.x = exp2f(a.x);
  out.y = exp2f(a.y);
#endif
  return out;
}

TL_DEVICE tl_f2 vec_exp2_f2(tl_f2 a) {
  tl_f2 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  float2 a_float2 = {a[0], a[1]};
  float2 out_float2;
  mute::fast_exp2(out_float2, a_float2);
  out = tl_f2{out_float2.x, out_float2.y};
#else
  out = tl_f2{exp2f(a[0]), exp2f(a[1])};
#endif
  return out;
}

TL_DEVICE float4 vec_exp2_f4(float4 a) {
  float4 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  mute::fast_exp2(out, a);
#else
  out.x = exp2f(a.x);
  out.y = exp2f(a.y);
  out.z = exp2f(a.z);
  out.w = exp2f(a.w);
#endif
  return out;
}

TL_DEVICE tl_f4 vec_exp2_f4(tl_f4 a) {
  tl_f4 out;
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ >= 310)
  float4 a_float4 = {a[0], a[1], a[2], a[3]};
  float4 out_float4;
  mute::fast_exp2(out_float4, a_float4);
  out = tl_f4{out_float4.x, out_float4.y, out_float4.z, out_float4.w};
#else
  out = tl_f4{exp2f(a[0]), exp2f(a[1]), exp2f(a[2]), exp2f(a[3])};
#endif
  return out;
}

} // namespace tl
