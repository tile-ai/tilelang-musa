#pragma once

#include "common.h"
#include "musa_fp8.h"

namespace tl {

using tl_char_v2 = char __attribute__((ext_vector_type(2)));
using tl_char_v4 = char __attribute__((ext_vector_type(4)));
using tl_float_v2 = float __attribute__((ext_vector_type(2)));
using tl_float_v4 = float __attribute__((ext_vector_type(4)));

struct __MUSA_ALIGN__(8) half4_t {
  half2 lo;
  half2 hi;
};

struct __MUSA_ALIGN__(8) bfloat164_t {
  __mt_bfloat162 lo;
  __mt_bfloat162 hi;
};

static_assert(sizeof(half4_t) == sizeof(__half4),
              "tl::half4_t must match MUSA __half4 size");
static_assert(alignof(half4_t) == alignof(__half4),
              "tl::half4_t must match MUSA __half4 alignment");
static_assert(sizeof(bfloat164_t) == sizeof(__mt_bfloat164),
              "tl::bfloat164_t must match MUSA __mt_bfloat164 size");
static_assert(alignof(bfloat164_t) == alignof(__mt_bfloat164),
              "tl::bfloat164_t must match MUSA __mt_bfloat164 alignment");

TL_DEVICE half4_t make_half4_t(half2 lo, half2 hi) {
  half4_t out;
  out.lo = lo;
  out.hi = hi;
  return out;
}

TL_DEVICE half4_t make_half4_t(__half4 in) {
  return *reinterpret_cast<const half4_t *>(&in);
}

TL_DEVICE __half4 to_musa_half4(half4_t in) {
  return *reinterpret_cast<const __half4 *>(&in);
}

TL_DEVICE bfloat164_t make_bfloat164_t(__mt_bfloat162 lo, __mt_bfloat162 hi) {
  bfloat164_t out;
  out.lo = lo;
  out.hi = hi;
  return out;
}

TL_DEVICE bfloat164_t make_bfloat164_t(__mt_bfloat164 in) {
  return *reinterpret_cast<const bfloat164_t *>(&in);
}

TL_DEVICE __mt_bfloat164 to_musa_bfloat164(bfloat164_t in) {
  return *reinterpret_cast<const __mt_bfloat164 *>(&in);
}

TL_DEVICE float4 make_float4_from_float2(float2 lo, float2 hi) {
  float4 out = {lo.x, lo.y, hi.x, hi.y};
  return out;
}

// fp16 <-> fp32
TL_DEVICE float2 cvt_half_to_float_x2(half2 in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  const tl_float_v2 out = __musa_f162f32_rn_bst2(in);
  return *reinterpret_cast<const float2 *>(&out);
#else
  return __half22float2(in);
#endif
}

TL_DEVICE float4 cvt_half_to_float_x4(half4_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  const tl_float_v4 out = __musa_f162f32_rn_bst4(to_musa_half4(in));
  return *reinterpret_cast<const float4 *>(&out);
#else
  return make_float4_from_float2(__half22float2(in.lo), __half22float2(in.hi));
#endif
}

TL_DEVICE half2 cvt_float_to_half_x2(float2 in) {
  return __float22half2_rn(in);
}

TL_DEVICE half4_t cvt_float_to_half_x4(float4 in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  return make_half4_t(__float42half4_rn(in));
#else
  float2 lo = {in.x, in.y};
  float2 hi = {in.z, in.w};
  return make_half4_t(__float22half2_rn(lo), __float22half2_rn(hi));
#endif
}

// bf16 <-> fp32
TL_DEVICE float2 cvt_bfloat16_to_float_x2(__mt_bfloat162 in) {
  return __bfloat1622float2(in);
}

TL_DEVICE float4 cvt_bfloat16_to_float_x4(bfloat164_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  return __bfloat1642float4(to_musa_bfloat164(in));
#else
  return make_float4_from_float2(__bfloat1622float2(in.lo),
                                 __bfloat1622float2(in.hi));
#endif
}

TL_DEVICE __mt_bfloat162 cvt_float_to_bfloat16_x2(float2 in) {
  return __float22bfloat162_rn(in);
}

TL_DEVICE bfloat164_t cvt_float_to_bfloat16_x4(float4 in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  return make_bfloat164_t(__float42bfloat164_rn(in));
#else
  float2 lo = {in.x, in.y};
  float2 hi = {in.z, in.w};
  return make_bfloat164_t(__float22bfloat162_rn(lo), __float22bfloat162_rn(hi));
#endif
}

TL_DEVICE bfloat164_t mul_half_float_to_bfloat16_x4(half4_t lhs, float4 rhs) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  const _BFloat16_4 out = __musa_mul_bhf_bst4_vv(
      to_musa_half4(lhs), *reinterpret_cast<const _Float4 *>(&rhs));
  return *reinterpret_cast<const bfloat164_t *>(&out);
#else
  const float4 lhs_float = cvt_half_to_float_x4(lhs);
  float4 out = {lhs_float.x * rhs.x, lhs_float.y * rhs.y, lhs_float.z * rhs.z,
                lhs_float.w * rhs.w};
  return cvt_float_to_bfloat16_x4(out);
#endif
}

// fp8 -> fp16 / fp32
TL_DEVICE float4 cvt_fp8e4m3_to_float_x4(fp8_e4_4_t in);
TL_DEVICE float4 cvt_fp8e5m2_to_float_x4(fp8_e5_4_t in);

TL_DEVICE half2 cvt_fp8e4m3_to_half_x2(fp8_e4_2_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  return __musa_e4m32f16_rn_bst2(*reinterpret_cast<const tl_char_v2 *>(&in));
#else
  __mt_fp8x2_e4m3 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x2_storage_t *>(&in);
  return static_cast<half2>(packed);
#endif
}

TL_DEVICE half4_t cvt_fp8e4m3_to_half_x4(fp8_e4_4_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  return make_half4_t(
      __musa_e4m32f16_rn_bst4(*reinterpret_cast<const tl_char_v4 *>(&in)));
#else
  return cvt_float_to_half_x4(cvt_fp8e4m3_to_float_x4(in));
#endif
}

TL_DEVICE half2 cvt_fp8e5m2_to_half_x2(fp8_e5_2_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  return __musa_e5m22f16_rn_bst2(*reinterpret_cast<const tl_char_v2 *>(&in));
#else
  __mt_fp8x2_e5m2 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x2_storage_t *>(&in);
  return static_cast<half2>(packed);
#endif
}

TL_DEVICE half4_t cvt_fp8e5m2_to_half_x4(fp8_e5_4_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  return make_half4_t(
      __musa_e5m22f16_rn_bst4(*reinterpret_cast<const tl_char_v4 *>(&in)));
#else
  return cvt_float_to_half_x4(cvt_fp8e5m2_to_float_x4(in));
#endif
}

TL_DEVICE float2 cvt_fp8e4m3_to_float_x2(fp8_e4_2_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  const tl_float_v2 out = __musa_f162f32_rn_bst2(
      __musa_e4m32f16_rn_bst2(*reinterpret_cast<const tl_char_v2 *>(&in)));
  return *reinterpret_cast<const float2 *>(&out);
#else
  __mt_fp8x2_e4m3 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x2_storage_t *>(&in);
  return static_cast<float2>(packed);
#endif
}

TL_DEVICE float4 cvt_fp8e4m3_to_float_x4(fp8_e4_4_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  const tl_float_v4 out = __musa_f162f32_rn_bst4(
      __musa_e4m32f16_rn_bst4(*reinterpret_cast<const tl_char_v4 *>(&in)));
  return *reinterpret_cast<const float4 *>(&out);
#else
  __mt_fp8x4_e4m3 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x4_storage_t *>(&in);
  return static_cast<float4>(packed);
#endif
}

TL_DEVICE float2 cvt_fp8e5m2_to_float_x2(fp8_e5_2_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  const tl_float_v2 out = __musa_f162f32_rn_bst2(
      __musa_e5m22f16_rn_bst2(*reinterpret_cast<const tl_char_v2 *>(&in)));
  return *reinterpret_cast<const float2 *>(&out);
#else
  __mt_fp8x2_e5m2 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x2_storage_t *>(&in);
  return static_cast<float2>(packed);
#endif
}

TL_DEVICE float4 cvt_fp8e5m2_to_float_x4(fp8_e5_4_t in) {
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  const tl_float_v4 out = __musa_f162f32_rn_bst4(
      __musa_e5m22f16_rn_bst4(*reinterpret_cast<const tl_char_v4 *>(&in)));
  return *reinterpret_cast<const float4 *>(&out);
#else
  __mt_fp8x4_e5m2 packed;
  packed.__x = *reinterpret_cast<const __mt_fp8x4_storage_t *>(&in);
  return static_cast<float4>(packed);
#endif
}

// fp16 / fp32 -> fp8
TL_DEVICE fp8_e4_2_t cvt_half_to_fp8e4m3_x2(half2 in) {
  fp8_e4_2_t out;
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) =
      (__mt_fp8x2_storage_t)__musa_f162e4m3_rn_bst2(in);
#else
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) = __mt_fp8x2_e4m3(in).__x;
#endif
  return out;
}

TL_DEVICE fp8_e4_4_t cvt_half_to_fp8e4m3_x4(half4_t in) {
  fp8_e4_4_t out;
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      (__mt_fp8x4_storage_t)__musa_f162e4m3_rn_bst4(to_musa_half4(in));
#else
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      __mt_fp8x4_e4m3(in.lo, in.hi).__x;
#endif
  return out;
}

TL_DEVICE fp8_e5_2_t cvt_half_to_fp8e5m2_x2(half2 in) {
  fp8_e5_2_t out;
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) =
      (__mt_fp8x2_storage_t)__musa_f162e5m2_rn_bst2(in);
#else
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) = __mt_fp8x2_e5m2(in).__x;
#endif
  return out;
}

TL_DEVICE fp8_e5_4_t cvt_half_to_fp8e5m2_x4(half4_t in) {
  fp8_e5_4_t out;
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      (__mt_fp8x4_storage_t)__musa_f162e5m2_rn_bst4(to_musa_half4(in));
#else
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      __mt_fp8x4_e5m2(in.lo, in.hi).__x;
#endif
  return out;
}

TL_DEVICE fp8_e4_2_t cvt_float_to_fp8e4m3_x2(float2 in) {
  fp8_e4_2_t out;
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) =
      (__mt_fp8x2_storage_t)__musa_f2e4m3_rn_bst2(
          *reinterpret_cast<const tl_float_v2 *>(&in));
#else
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) = __mt_fp8x2_e4m3(in).__x;
#endif
  return out;
}

TL_DEVICE fp8_e4_4_t cvt_float_to_fp8e4m3_x4(float4 in) {
  fp8_e4_4_t out;
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      (__mt_fp8x4_storage_t)__musa_f2e4m3_rn_bst4(
          *reinterpret_cast<const tl_float_v4 *>(&in));
#else
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) = __mt_fp8x4_e4m3(in).__x;
#endif
  return out;
}

TL_DEVICE fp8_e5_2_t cvt_float_to_fp8e5m2_x2(float2 in) {
  fp8_e5_2_t out;
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) =
      (__mt_fp8x2_storage_t)__musa_f2e5m2_rn_bst2(
          *reinterpret_cast<const tl_float_v2 *>(&in));
#else
  *reinterpret_cast<__mt_fp8x2_storage_t *>(&out) = __mt_fp8x2_e5m2(in).__x;
#endif
  return out;
}

TL_DEVICE fp8_e5_4_t cvt_float_to_fp8e5m2_x4(float4 in) {
  fp8_e5_4_t out;
#if defined(__MUSACC_VER_MAJOR__) && (__MUSACC_VER_MAJOR__ > 4)
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) =
      (__mt_fp8x4_storage_t)__musa_f2e5m2_rn_bst4(
          *reinterpret_cast<const tl_float_v4 *>(&in));
#else
  *reinterpret_cast<__mt_fp8x4_storage_t *>(&out) = __mt_fp8x4_e5m2(in).__x;
#endif
  return out;
}

} // namespace tl
