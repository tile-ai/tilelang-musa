#pragma once

#include <musa_bf16.h>
#include <musa_fp16.h>
#include <musa_fp8.h>

namespace tl {

__device__ inline float2 cvt_half_to_float_x2(half2 in) {
  return __half22float2(in);
}

__device__ inline float4 cvt_half_to_float_x4(half4 in) {
  return make_float4(__half2float(in.x), __half2float(in.y), __half2float(in.z),
                     __half2float(in.w));
}

__device__ inline half2 cvt_float_to_half_x2(float2 in) {
  return __float22half2_rn(in);
}

__device__ inline half4 cvt_float_to_half_x4(float4 in) {
  return make_half4(__float2half(in.x), __float2half(in.y), __float2half(in.z),
                    __float2half(in.w));
}

__device__ inline float2 cvt_bfloat16_to_float_x2(mt_bfloat162 in) {
  return __bfloat1622float2(in);
}

__device__ inline float4 cvt_bfloat16_to_float_x4(mt_bfloat164 in) {
  return make_float4(__bfloat162float(in.x), __bfloat162float(in.y),
                     __bfloat162float(in.z), __bfloat162float(in.w));
}

__device__ inline mt_bfloat162 cvt_float_to_bfloat16_x2(float2 in) {
  return __float22bfloat162_rn(in);
}

__device__ inline mt_bfloat164 cvt_float_to_bfloat16_x4(float4 in) {
  return make_mt_bfloat164(__float2bfloat16(in.x), __float2bfloat16(in.y),
                           __float2bfloat16(in.z), __float2bfloat16(in.w));
}

// Match CodeGenMUSA's packed storage: two 16-bit lanes per uint and two
// FP32 lanes per ulonglong. Convert x6 in three x2 chunks and x8 in two x4.
__device__ inline ulonglong3 cvt_half_to_float_x6(uint3 in) {
  ulonglong3 out;
  const half2 *src = reinterpret_cast<const half2 *>(&in);
  float2 *dst = reinterpret_cast<float2 *>(&out);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    dst[i] = cvt_half_to_float_x2(src[i]);
  }
  return out;
}

__device__ inline ulonglong4 cvt_half_to_float_x8(uint4 in) {
  ulonglong4 out;
  const half4 *src = reinterpret_cast<const half4 *>(&in);
  float4 *dst = reinterpret_cast<float4 *>(&out);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    dst[i] = cvt_half_to_float_x4(src[i]);
  }
  return out;
}

__device__ inline uint3 cvt_float_to_half_x6(ulonglong3 in) {
  uint3 out;
  const float2 *src = reinterpret_cast<const float2 *>(&in);
  half2 *dst = reinterpret_cast<half2 *>(&out);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    dst[i] = cvt_float_to_half_x2(src[i]);
  }
  return out;
}

__device__ inline uint4 cvt_float_to_half_x8(ulonglong4 in) {
  uint4 out;
  const float4 *src = reinterpret_cast<const float4 *>(&in);
  half4 *dst = reinterpret_cast<half4 *>(&out);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    dst[i] = cvt_float_to_half_x4(src[i]);
  }
  return out;
}

__device__ inline ulonglong3 cvt_bfloat16_to_float_x6(uint3 in) {
  ulonglong3 out;
  const mt_bfloat162 *src = reinterpret_cast<const mt_bfloat162 *>(&in);
  float2 *dst = reinterpret_cast<float2 *>(&out);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    dst[i] = cvt_bfloat16_to_float_x2(src[i]);
  }
  return out;
}

__device__ inline ulonglong4 cvt_bfloat16_to_float_x8(uint4 in) {
  ulonglong4 out;
  const mt_bfloat164 *src = reinterpret_cast<const mt_bfloat164 *>(&in);
  float4 *dst = reinterpret_cast<float4 *>(&out);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    dst[i] = cvt_bfloat16_to_float_x4(src[i]);
  }
  return out;
}

__device__ inline uint3 cvt_float_to_bfloat16_x6(ulonglong3 in) {
  uint3 out;
  const float2 *src = reinterpret_cast<const float2 *>(&in);
  mt_bfloat162 *dst = reinterpret_cast<mt_bfloat162 *>(&out);
#pragma unroll
  for (int i = 0; i < 3; ++i) {
    dst[i] = cvt_float_to_bfloat16_x2(src[i]);
  }
  return out;
}

__device__ inline uint4 cvt_float_to_bfloat16_x8(ulonglong4 in) {
  uint4 out;
  const float4 *src = reinterpret_cast<const float4 *>(&in);
  mt_bfloat164 *dst = reinterpret_cast<mt_bfloat164 *>(&out);
#pragma unroll
  for (int i = 0; i < 2; ++i) {
    dst[i] = cvt_float_to_bfloat16_x4(src[i]);
  }
  return out;
}

#define TL_MUSA_DEFINE_FP8_CASTS(name, type2, type4)                           \
  __device__ inline float2 cvt_##name##_to_float_x2(type2 in) {                \
    return static_cast<float2>(in);                                            \
  }                                                                            \
  __device__ inline float4 cvt_##name##_to_float_x4(type4 in) {                \
    return static_cast<float4>(in);                                            \
  }                                                                            \
  __device__ inline type2 cvt_float_to_##name##_x2(float2 in) {                \
    return type2(in);                                                          \
  }                                                                            \
  __device__ inline type4 cvt_float_to_##name##_x4(float4 in) {                \
    return type4(in);                                                          \
  }

TL_MUSA_DEFINE_FP8_CASTS(fp8e4m3, __mt_fp8x2_e4m3, __mt_fp8x4_e4m3)
TL_MUSA_DEFINE_FP8_CASTS(fp8e5m2, __mt_fp8x2_e5m2, __mt_fp8x4_e5m2)
TL_MUSA_DEFINE_FP8_CASTS(fp8e8m0, __mt_fp8x2_e8m0, __mt_fp8x4_e8m0)

#undef TL_MUSA_DEFINE_FP8_CASTS

#define TL_MUSA_DEFINE_FP8_HALF_CASTS(name, type2, type4)                      \
  __device__ inline half2 cvt_##name##_to_half_x2(type2 in) {                  \
    return cvt_float_to_half_x2(cvt_##name##_to_float_x2(in));                 \
  }                                                                            \
  __device__ inline half4 cvt_##name##_to_half_x4(type4 in) {                  \
    return cvt_float_to_half_x4(cvt_##name##_to_float_x4(in));                 \
  }                                                                            \
  __device__ inline type2 cvt_half_to_##name##_x2(half2 in) {                  \
    return cvt_float_to_##name##_x2(cvt_half_to_float_x2(in));                 \
  }                                                                            \
  __device__ inline type4 cvt_half_to_##name##_x4(half4 in) {                  \
    return cvt_float_to_##name##_x4(cvt_half_to_float_x4(in));                 \
  }

TL_MUSA_DEFINE_FP8_HALF_CASTS(fp8e4m3, __mt_fp8x2_e4m3, __mt_fp8x4_e4m3)
TL_MUSA_DEFINE_FP8_HALF_CASTS(fp8e5m2, __mt_fp8x2_e5m2, __mt_fp8x4_e5m2)

#undef TL_MUSA_DEFINE_FP8_HALF_CASTS

} // namespace tl
