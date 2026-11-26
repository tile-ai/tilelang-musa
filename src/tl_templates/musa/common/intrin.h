#pragma once

#include <musa.h>

#ifndef TL_DEVICE
#define TL_DEVICE __forceinline__ __device__
#endif

#ifndef TL_DEVICE_NOINLINE
#define TL_DEVICE_NOINLINE __noinline__ __device__
#endif

#ifndef TL_HOST_DEVICE
#define TL_HOST_DEVICE __forceinline__ __host__ __device__
#endif

#ifndef TL_PATCH
#define TL_PATCH
#endif

// Packed integer vector constructors emitted by CodeGenMUSA::PrintVecConstructor
// for 8-bit, 16-bit, and wide 32-bit vector types (e.g. int8x8 -> int2).
// Build 32-bit lanes from unsigned bytes so negative signed char values do not
// sign-extend and corrupt upper lanes.
TL_DEVICE int make_int(signed char x0, signed char x1, signed char x2,
                       signed char x3) {
  const unsigned int b0 = static_cast<unsigned char>(x0);
  const unsigned int b1 = static_cast<unsigned char>(x1);
  const unsigned int b2 = static_cast<unsigned char>(x2);
  const unsigned int b3 = static_cast<unsigned char>(x3);
  return static_cast<int>((b3 << 24) | (b2 << 16) | (b1 << 8) | b0);
}

TL_DEVICE int2 make_int2(signed char x0, signed char x1, signed char x2,
                         signed char x3, signed char y0, signed char y1,
                         signed char y2, signed char y3) {
  int2 result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  return result;
}

TL_DEVICE int4 make_int4(signed char x0, signed char x1, signed char x2,
                         signed char x3, signed char y0, signed char y1,
                         signed char y2, signed char y3, signed char z0,
                         signed char z1, signed char z2, signed char z3,
                         signed char w0, signed char w1, signed char w2,
                         signed char w3) {
  int4 result;
  result.x = make_int(x0, x1, x2, x3);
  result.y = make_int(y0, y1, y2, y3);
  result.z = make_int(z0, z1, z2, z3);
  result.w = make_int(w0, w1, w2, w3);
  return result;
}

TL_DEVICE int3 make_int3(short x0, short x1, short y0, short y1, short z0,
                         short z1) {
  int3 result;
  *reinterpret_cast<short2 *>(&result.x) = make_short2(x0, x1);
  *reinterpret_cast<short2 *>(&result.y) = make_short2(y0, y1);
  *reinterpret_cast<short2 *>(&result.z) = make_short2(z0, z1);
  return result;
}

TL_DEVICE int4 make_int4(short x0, short x1, short y0, short y1, short z0,
                         short z1, short w0, short w1) {
  int4 result;
  *reinterpret_cast<short2 *>(&result.x) = make_short2(x0, x1);
  *reinterpret_cast<short2 *>(&result.y) = make_short2(y0, y1);
  *reinterpret_cast<short2 *>(&result.z) = make_short2(z0, z1);
  *reinterpret_cast<short2 *>(&result.w) = make_short2(w0, w1);
  return result;
}

TL_DEVICE unsigned int make_uint(unsigned char x0, unsigned char x1,
                                 unsigned char x2, unsigned char x3) {
  const unsigned int b0 = static_cast<unsigned int>(x0);
  const unsigned int b1 = static_cast<unsigned int>(x1);
  const unsigned int b2 = static_cast<unsigned int>(x2);
  const unsigned int b3 = static_cast<unsigned int>(x3);
  return (b3 << 24) | (b2 << 16) | (b1 << 8) | b0;
}

TL_DEVICE uint2 make_uint2(unsigned char x0, unsigned char x1, unsigned char x2,
                           unsigned char x3, unsigned char y0, unsigned char y1,
                           unsigned char y2, unsigned char y3) {
  uint2 result;
  result.x = make_uint(x0, x1, x2, x3);
  result.y = make_uint(y0, y1, y2, y3);
  return result;
}

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

TL_DEVICE uint3 make_uint3(unsigned short x0, unsigned short x1,
                           unsigned short y0, unsigned short y1,
                           unsigned short z0, unsigned short z1) {
  uint3 result;
  *reinterpret_cast<ushort2 *>(&result.x) = make_ushort2(x0, x1);
  *reinterpret_cast<ushort2 *>(&result.y) = make_ushort2(y0, y1);
  *reinterpret_cast<ushort2 *>(&result.z) = make_ushort2(z0, z1);
  return result;
}

TL_DEVICE uint4 make_uint4(unsigned short x0, unsigned short x1,
                           unsigned short y0, unsigned short y1,
                           unsigned short z0, unsigned short z1,
                           unsigned short w0, unsigned short w1) {
  uint4 result;
  *reinterpret_cast<ushort2 *>(&result.x) = make_ushort2(x0, x1);
  *reinterpret_cast<ushort2 *>(&result.y) = make_ushort2(y0, y1);
  *reinterpret_cast<ushort2 *>(&result.z) = make_ushort2(z0, z1);
  *reinterpret_cast<ushort2 *>(&result.w) = make_ushort2(w0, w1);
  return result;
}

TL_DEVICE longlong3 make_longlong3(int x0, int x1, int y0, int y1, int z0,
                                   int z1) {
  longlong3 result;
  *reinterpret_cast<int2 *>(&result.x) = make_int2(x0, x1);
  *reinterpret_cast<int2 *>(&result.y) = make_int2(y0, y1);
  *reinterpret_cast<int2 *>(&result.z) = make_int2(z0, z1);
  return result;
}

TL_DEVICE longlong4 make_longlong4(int x0, int x1, int y0, int y1, int z0,
                                   int z1, int w0, int w1) {
  longlong4 result;
  *reinterpret_cast<int2 *>(&result.x) = make_int2(x0, x1);
  *reinterpret_cast<int2 *>(&result.y) = make_int2(y0, y1);
  *reinterpret_cast<int2 *>(&result.z) = make_int2(z0, z1);
  *reinterpret_cast<int2 *>(&result.w) = make_int2(w0, w1);
  return result;
}

TL_DEVICE ulonglong3 make_ulonglong3(unsigned int x0, unsigned int x1,
                                     unsigned int y0, unsigned int y1,
                                     unsigned int z0, unsigned int z1) {
  ulonglong3 result;
  *reinterpret_cast<uint2 *>(&result.x) = make_uint2(x0, x1);
  *reinterpret_cast<uint2 *>(&result.y) = make_uint2(y0, y1);
  *reinterpret_cast<uint2 *>(&result.z) = make_uint2(z0, z1);
  return result;
}

TL_DEVICE ulonglong4 make_ulonglong4(unsigned int x0, unsigned int x1,
                                     unsigned int y0, unsigned int y1,
                                     unsigned int z0, unsigned int z1,
                                     unsigned int w0, unsigned int w1) {
  ulonglong4 result;
  *reinterpret_cast<uint2 *>(&result.x) = make_uint2(x0, x1);
  *reinterpret_cast<uint2 *>(&result.y) = make_uint2(y0, y1);
  *reinterpret_cast<uint2 *>(&result.z) = make_uint2(z0, z1);
  *reinterpret_cast<uint2 *>(&result.w) = make_uint2(w0, w1);
  return result;
}

namespace tl {
namespace detail {

TL_DEVICE constexpr int default_warp_size() {
#if defined(__MUSA_ARCH__) && (__MUSA_ARCH__ <= 220)
  return 128;
#else
  return 32;
#endif
}

TL_DEVICE constexpr int default_warps_per_group() { return 4; }

TL_DEVICE int linear_thread_idx_in_block() {
#if defined(__MUSA_ARCH__)
  return threadIdx.x + blockDim.x * (threadIdx.y + blockDim.y * threadIdx.z);
#else
  return 0;
#endif
}

} // namespace detail
} // namespace tl

namespace tl {

TL_DEVICE int get_lane_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() % warp_size;
}

TL_DEVICE int get_warp_idx_sync(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int get_warp_idx(int warp_size = detail::default_warp_size()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  return detail::linear_thread_idx_in_block() / warp_size;
}

TL_DEVICE int
get_warp_group_idx(int warp_size = detail::default_warp_size(),
                   int warps_per_group = detail::default_warps_per_group()) {
  warp_size = warp_size > 0 ? warp_size : detail::default_warp_size();
  warps_per_group =
      warps_per_group > 0 ? warps_per_group : detail::default_warps_per_group();
  int threads_per_group = warp_size * warps_per_group;
  threads_per_group = threads_per_group > 0 ? threads_per_group : warp_size;
  return detail::linear_thread_idx_in_block() / threads_per_group;
}

TL_DEVICE uint32_t elect_one_sync() { return get_lane_idx() == 0; }

TL_DEVICE int canonical_warp_idx_sync() { return get_warp_idx_sync(); }

template <int y = 1, typename T> TL_DEVICE T pow_of_int(T x) {
  T result = x;
  for (int i = 1; i < y; i++) {
    result *= x;
  }
  return result;
}

template <int thread_extent> TL_DEVICE bool tl_shuffle_elect() {
  if constexpr (thread_extent == 0) {
    return canonical_warp_idx_sync() == 0 && elect_one_sync();
  } else {
    constexpr int warp_size = detail::default_warp_size();
    constexpr int warp_extent = (thread_extent + warp_size - 1) / warp_size;
    static_assert(warp_extent > 0);
    return (canonical_warp_idx_sync() % warp_extent) == 0 && elect_one_sync();
  }
}

} // namespace tl
