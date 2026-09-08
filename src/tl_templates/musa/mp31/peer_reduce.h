#pragma once

#include <cstdint>
#include <musa_bf16.h>
#include <musa_fp16.h>

namespace tl {

using peer_h8 = uint16_t __attribute__((ext_vector_type(8)));
using peer_bf8 = uint16_t __attribute__((ext_vector_type(8)));

TL_DEVICE float peer_half_to_float(uint16_t bits) {
  __half_raw raw;
  raw.x = bits;
  return __half2float(__half(raw));
}

TL_DEVICE uint16_t peer_float_to_half(float value) {
  return static_cast<__half_raw>(__float2half_rn(value)).x;
}

TL_DEVICE float peer_bfloat_to_float(uint16_t bits) {
  const __mt_bfloat16 value = *reinterpret_cast<const __mt_bfloat16 *>(&bits);
  return __bfloat162float(value);
}

TL_DEVICE uint16_t peer_float_to_bfloat(float value) {
  const __mt_bfloat16 converted = __float2bfloat16(value);
  return *reinterpret_cast<const uint16_t *>(&converted);
}

TL_DEVICE float4 peer_shfl_xor_sync(uint32_t mask, float4 value, int lane_mask,
                                    int width = 32) {
  return make_float4(__shfl_xor_sync(mask, value.x, lane_mask, width),
                     __shfl_xor_sync(mask, value.y, lane_mask, width),
                     __shfl_xor_sync(mask, value.z, lane_mask, width),
                     __shfl_xor_sync(mask, value.w, lane_mask, width));
}

// Each warp contains four eight-lane peer groups.  XOR 16 and XOR 8 reduce
// those four groups while preserving eight coalesced 16-byte loads per peer.
// A paired warp handles the other four peers; lanes 0..7 combine both halves
// through shared FP32 scratch.  Callers that reuse scratch in a loop must add
// a block sync between iterations; the final iteration needs no trailing sync.
TL_DEVICE float4 peer_add_float4(float4 a, float4 b) {
  return make_float4(a.x + b.x, a.y + b.y, a.z + b.z, a.w + b.w);
}

TL_DEVICE void peer_warp_reduce_float4_pair(float4 &lo, float4 &hi,
                                            float *scratch,
                                            int values_per_vector) {
  const int lane = int(threadIdx.x) & 31;
  const int warp = int(threadIdx.x) >> 5;
  const int scratch_base = (warp * 8 + lane) * values_per_vector;
  lo = peer_add_float4(lo, peer_shfl_xor_sync(0xffffffffu, lo, 16));
  lo = peer_add_float4(lo, peer_shfl_xor_sync(0xffffffffu, lo, 8));
  if (values_per_vector == 8) {
    hi = peer_add_float4(hi, peer_shfl_xor_sync(0xffffffffu, hi, 16));
    hi = peer_add_float4(hi, peer_shfl_xor_sync(0xffffffffu, hi, 8));
  }
  if (lane < 8) {
    scratch[scratch_base + 0] = lo.x;
    scratch[scratch_base + 1] = lo.y;
    scratch[scratch_base + 2] = lo.z;
    scratch[scratch_base + 3] = lo.w;
    if (values_per_vector == 8) {
      scratch[scratch_base + 4] = hi.x;
      scratch[scratch_base + 5] = hi.y;
      scratch[scratch_base + 6] = hi.z;
      scratch[scratch_base + 7] = hi.w;
    }
  }
  __syncthreads();
  if (((warp & 1) == 0) && lane < 8) {
    const int peer_half = scratch_base + 8 * values_per_vector;
    lo = peer_add_float4(
        make_float4(scratch[scratch_base + 0], scratch[scratch_base + 1],
                    scratch[scratch_base + 2], scratch[scratch_base + 3]),
        make_float4(scratch[peer_half + 0], scratch[peer_half + 1],
                    scratch[peer_half + 2], scratch[peer_half + 3]));
    if (values_per_vector == 8) {
      hi = peer_add_float4(
          make_float4(scratch[scratch_base + 4], scratch[scratch_base + 5],
                      scratch[scratch_base + 6], scratch[scratch_base + 7]),
          make_float4(scratch[peer_half + 4], scratch[peer_half + 5],
                      scratch[peer_half + 6], scratch[peer_half + 7]));
    }
  }
}

TL_DEVICE uint4 peer_warp_reduce_bfloat16x8(uint4 packed, float *scratch) {
  const peer_bf8 values = *reinterpret_cast<const peer_bf8 *>(&packed);
  float4 lo = make_float4(
      peer_bfloat_to_float(values[0]), peer_bfloat_to_float(values[1]),
      peer_bfloat_to_float(values[2]), peer_bfloat_to_float(values[3]));
  float4 hi = make_float4(
      peer_bfloat_to_float(values[4]), peer_bfloat_to_float(values[5]),
      peer_bfloat_to_float(values[6]), peer_bfloat_to_float(values[7]));
  peer_warp_reduce_float4_pair(lo, hi, scratch, 8);
  peer_bf8 result{};
  if (((int(threadIdx.x) >> 5) & 1) == 0 && (int(threadIdx.x) & 31) < 8) {
    result[0] = peer_float_to_bfloat(lo.x);
    result[1] = peer_float_to_bfloat(lo.y);
    result[2] = peer_float_to_bfloat(lo.z);
    result[3] = peer_float_to_bfloat(lo.w);
    result[4] = peer_float_to_bfloat(hi.x);
    result[5] = peer_float_to_bfloat(hi.y);
    result[6] = peer_float_to_bfloat(hi.z);
    result[7] = peer_float_to_bfloat(hi.w);
  }
  return *reinterpret_cast<const uint4 *>(&result);
}

TL_DEVICE uint4 peer_warp_reduce_fp16x8(uint4 packed, float *scratch) {
  const peer_h8 values = *reinterpret_cast<const peer_h8 *>(&packed);
  float4 lo =
      make_float4(peer_half_to_float(values[0]), peer_half_to_float(values[1]),
                  peer_half_to_float(values[2]), peer_half_to_float(values[3]));
  float4 hi =
      make_float4(peer_half_to_float(values[4]), peer_half_to_float(values[5]),
                  peer_half_to_float(values[6]), peer_half_to_float(values[7]));
  peer_warp_reduce_float4_pair(lo, hi, scratch, 8);
  peer_h8 result{};
  if (((int(threadIdx.x) >> 5) & 1) == 0 && (int(threadIdx.x) & 31) < 8) {
    result[0] = peer_float_to_half(lo.x);
    result[1] = peer_float_to_half(lo.y);
    result[2] = peer_float_to_half(lo.z);
    result[3] = peer_float_to_half(lo.w);
    result[4] = peer_float_to_half(hi.x);
    result[5] = peer_float_to_half(hi.y);
    result[6] = peer_float_to_half(hi.z);
    result[7] = peer_float_to_half(hi.w);
  }
  return *reinterpret_cast<const uint4 *>(&result);
}

TL_DEVICE uint4 peer_warp_reduce_float32x4(uint4 packed, float *scratch) {
  float4 lo = *reinterpret_cast<const float4 *>(&packed);
  float4 unused = make_float4(0.0f, 0.0f, 0.0f, 0.0f);
  peer_warp_reduce_float4_pair(lo, unused, scratch, 4);
  return *reinterpret_cast<const uint4 *>(&lo);
}

} // namespace tl
