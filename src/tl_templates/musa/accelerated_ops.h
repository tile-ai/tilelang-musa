#pragma once

#include "cvt.h"

namespace tl {

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

} // namespace tl
