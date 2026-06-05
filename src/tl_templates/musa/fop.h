#pragma once

#include "common.h"

#if defined(__has_builtin)
#define TL_HAS_MUSA_BUILTIN(x) __has_builtin(x)
#else
#define TL_HAS_MUSA_BUILTIN(x) 0
#endif

#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__ >= 310)
#define TL_MUSA_ARCH_GE_310 1
#else
#define TL_MUSA_ARCH_GE_310 0
#endif

#define TL_HAS_MUSA_BUILTIN_MP31(x)                                            \
  (TL_MUSA_ARCH_GE_310 && TL_HAS_MUSA_BUILTIN(x))

namespace tl {

TL_DEVICE float scalar_max(float a, float b) { return fmaxf(a, b); }

TL_DEVICE float scalar_min(float a, float b) { return fminf(a, b); }

TL_DEVICE half_t scalar_max(half_t a, half_t b) {
  return half_t(__hmax(a.to_half(), b.to_half()));
}

TL_DEVICE half_t scalar_min(half_t a, half_t b) {
  return half_t(__hmin(a.to_half(), b.to_half()));
}

TL_DEVICE bfloat16_t scalar_max(bfloat16_t a, bfloat16_t b) {
  return bfloat16_t(__hmax(__mt_bfloat16(a), __mt_bfloat16(b)));
}

TL_DEVICE bfloat16_t scalar_min(bfloat16_t a, bfloat16_t b) {
  return bfloat16_t(__hmin(__mt_bfloat16(a), __mt_bfloat16(b)));
}

TL_DEVICE tl_f2 add2(tl_f2 a, tl_f2 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_add_f_bst2_vv)
  return __musa_add_f_bst2_vv(a, b);
#else
  return tl_f2{a[0] + b[0], a[1] + b[1]};
#endif
}

TL_DEVICE tl_f2 sub2(tl_f2 a, tl_f2 b) {
  return tl_f2{a[0] - b[0], a[1] - b[1]};
}

TL_DEVICE tl_f2 mul2(tl_f2 a, tl_f2 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_mul_f_bst2_vv)
  return __musa_mul_f_bst2_vv(a, b);
#else
  return tl_f2{a[0] * b[0], a[1] * b[1]};
#endif
}

TL_DEVICE tl_f2 fma2(tl_f2 a, tl_f2 b, tl_f2 c) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_fma_f_bst2_vvv)
  return __musa_fma_f_bst2_vvv(a, b, c);
#else
  return tl_f2{fmaf(a[0], b[0], c[0]), fmaf(a[1], b[1], c[1])};
#endif
}

TL_DEVICE tl_f2 max2(tl_f2 a, tl_f2 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_max_f_bst2_vv)
  return __musa_max_f_bst2_vv(a, b);
#else
  return tl_f2{scalar_max(a[0], b[0]), scalar_max(a[1], b[1])};
#endif
}

TL_DEVICE tl_f2 min2(tl_f2 a, tl_f2 b) {
  return tl_f2{scalar_min(a[0], b[0]), scalar_min(a[1], b[1])};
}

TL_DEVICE tl_f2 abs2(tl_f2 a) { return tl_f2{fabsf(a[0]), fabsf(a[1])}; }

TL_DEVICE tl_f4 add4(tl_f4 a, tl_f4 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_add_f_bst4_vv)
  return __musa_add_f_bst4_vv(a, b);
#else
  return tl_f4{a[0] + b[0], a[1] + b[1], a[2] + b[2], a[3] + b[3]};
#endif
}

TL_DEVICE tl_f4 sub4(tl_f4 a, tl_f4 b) {
  return tl_f4{a[0] - b[0], a[1] - b[1], a[2] - b[2], a[3] - b[3]};
}

TL_DEVICE tl_f4 mul4(tl_f4 a, tl_f4 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_mul_f_bst4_vv)
  return __musa_mul_f_bst4_vv(a, b);
#else
  return tl_f4{a[0] * b[0], a[1] * b[1], a[2] * b[2], a[3] * b[3]};
#endif
}

TL_DEVICE tl_f4 fma4(tl_f4 a, tl_f4 b, tl_f4 c) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_fma_f_bst4_vvv)
  return __musa_fma_f_bst4_vvv(a, b, c);
#else
  return tl_f4{fmaf(a[0], b[0], c[0]), fmaf(a[1], b[1], c[1]),
               fmaf(a[2], b[2], c[2]), fmaf(a[3], b[3], c[3])};
#endif
}

TL_DEVICE tl_f4 max4(tl_f4 a, tl_f4 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_max_f_bst4_vv)
  return __musa_max_f_bst4_vv(a, b);
#else
  return tl_f4{scalar_max(a[0], b[0]), scalar_max(a[1], b[1]),
               scalar_max(a[2], b[2]), scalar_max(a[3], b[3])};
#endif
}

TL_DEVICE tl_f4 min4(tl_f4 a, tl_f4 b) {
  return tl_f4{scalar_min(a[0], b[0]), scalar_min(a[1], b[1]),
               scalar_min(a[2], b[2]), scalar_min(a[3], b[3])};
}

TL_DEVICE tl_f4 abs4(tl_f4 a) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_fabs_f_bst4)
  return __musa_fabs_f_bst4(a);
#else
  return tl_f4{fabsf(a[0]), fabsf(a[1]), fabsf(a[2]), fabsf(a[3])};
#endif
}

TL_DEVICE tl_h2 add2(tl_h2 a, tl_h2 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_add_h_bst2_vv)
  return __musa_add_h_bst2_vv(a, b);
#else
  return make_tl_h2(tl_h_elem_to_half(a[0]) + tl_h_elem_to_half(b[0]),
                    tl_h_elem_to_half(a[1]) + tl_h_elem_to_half(b[1]));
#endif
}

TL_DEVICE tl_h2 sub2(tl_h2 a, tl_h2 b) {
  return make_tl_h2(tl_h_elem_to_half(a[0]) - tl_h_elem_to_half(b[0]),
                    tl_h_elem_to_half(a[1]) - tl_h_elem_to_half(b[1]));
}

TL_DEVICE tl_h2 mul2(tl_h2 a, tl_h2 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_mul_h_bst2_vv)
  return __musa_mul_h_bst2_vv(a, b);
#else
  return make_tl_h2(tl_h_elem_to_half(a[0]) * tl_h_elem_to_half(b[0]),
                    tl_h_elem_to_half(a[1]) * tl_h_elem_to_half(b[1]));
#endif
}

TL_DEVICE tl_h2 fma2(tl_h2 a, tl_h2 b, tl_h2 c) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_fma_h_bst2_vvv)
  return __musa_fma_h_bst2_vvv(a, b, c);
#else
  return make_tl_h2(tl_h_elem_to_half(a[0]) * tl_h_elem_to_half(b[0]) +
                        tl_h_elem_to_half(c[0]),
                    tl_h_elem_to_half(a[1]) * tl_h_elem_to_half(b[1]) +
                        tl_h_elem_to_half(c[1]));
#endif
}

TL_DEVICE tl_h2 max2(tl_h2 a, tl_h2 b) {
  // TODO(MUSA): Re-enable __musa_max_h_bst2_vv after mtcc can select it.
  return make_tl_h2(
      scalar_max(tl_h_elem_to_half(a[0]), tl_h_elem_to_half(b[0])),
      scalar_max(tl_h_elem_to_half(a[1]), tl_h_elem_to_half(b[1])));
}

TL_DEVICE tl_h2 min2(tl_h2 a, tl_h2 b) {
  return make_tl_h2(
      scalar_min(tl_h_elem_to_half(a[0]), tl_h_elem_to_half(b[0])),
      scalar_min(tl_h_elem_to_half(a[1]), tl_h_elem_to_half(b[1])));
}

TL_DEVICE tl_h2 abs2(tl_h2 a) {
  return make_tl_h2(abs(tl_h_elem_to_half(a[0])), abs(tl_h_elem_to_half(a[1])));
}

TL_DEVICE tl_h4 add4(tl_h4 a, tl_h4 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_add_h_bst4_vv)
  return __musa_add_h_bst4_vv(a, b);
#else
  return make_tl_h4(tl_h_elem_to_half(a[0]) + tl_h_elem_to_half(b[0]),
                    tl_h_elem_to_half(a[1]) + tl_h_elem_to_half(b[1]),
                    tl_h_elem_to_half(a[2]) + tl_h_elem_to_half(b[2]),
                    tl_h_elem_to_half(a[3]) + tl_h_elem_to_half(b[3]));
#endif
}

TL_DEVICE tl_h4 sub4(tl_h4 a, tl_h4 b) {
  return make_tl_h4(tl_h_elem_to_half(a[0]) - tl_h_elem_to_half(b[0]),
                    tl_h_elem_to_half(a[1]) - tl_h_elem_to_half(b[1]),
                    tl_h_elem_to_half(a[2]) - tl_h_elem_to_half(b[2]),
                    tl_h_elem_to_half(a[3]) - tl_h_elem_to_half(b[3]));
}

TL_DEVICE tl_h4 mul4(tl_h4 a, tl_h4 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_mul_h_bst4_vv)
  return __musa_mul_h_bst4_vv(a, b);
#else
  return make_tl_h4(tl_h_elem_to_half(a[0]) * tl_h_elem_to_half(b[0]),
                    tl_h_elem_to_half(a[1]) * tl_h_elem_to_half(b[1]),
                    tl_h_elem_to_half(a[2]) * tl_h_elem_to_half(b[2]),
                    tl_h_elem_to_half(a[3]) * tl_h_elem_to_half(b[3]));
#endif
}

TL_DEVICE tl_h4 fma4(tl_h4 a, tl_h4 b, tl_h4 c) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_fma_h_bst4_vvv)
  return __musa_fma_h_bst4_vvv(a, b, c);
#else
  return make_tl_h4(tl_h_elem_to_half(a[0]) * tl_h_elem_to_half(b[0]) +
                        tl_h_elem_to_half(c[0]),
                    tl_h_elem_to_half(a[1]) * tl_h_elem_to_half(b[1]) +
                        tl_h_elem_to_half(c[1]),
                    tl_h_elem_to_half(a[2]) * tl_h_elem_to_half(b[2]) +
                        tl_h_elem_to_half(c[2]),
                    tl_h_elem_to_half(a[3]) * tl_h_elem_to_half(b[3]) +
                        tl_h_elem_to_half(c[3]));
#endif
}

TL_DEVICE tl_h4 max4(tl_h4 a, tl_h4 b) {
  // TODO(MUSA): Re-enable __musa_max_h_bst4_vv after mtcc can select it.
  return make_tl_h4(
      scalar_max(tl_h_elem_to_half(a[0]), tl_h_elem_to_half(b[0])),
      scalar_max(tl_h_elem_to_half(a[1]), tl_h_elem_to_half(b[1])),
      scalar_max(tl_h_elem_to_half(a[2]), tl_h_elem_to_half(b[2])),
      scalar_max(tl_h_elem_to_half(a[3]), tl_h_elem_to_half(b[3])));
}

TL_DEVICE tl_h4 min4(tl_h4 a, tl_h4 b) {
  return make_tl_h4(
      scalar_min(tl_h_elem_to_half(a[0]), tl_h_elem_to_half(b[0])),
      scalar_min(tl_h_elem_to_half(a[1]), tl_h_elem_to_half(b[1])),
      scalar_min(tl_h_elem_to_half(a[2]), tl_h_elem_to_half(b[2])),
      scalar_min(tl_h_elem_to_half(a[3]), tl_h_elem_to_half(b[3])));
}

TL_DEVICE tl_h4 abs4(tl_h4 a) {
  return make_tl_h4(abs(tl_h_elem_to_half(a[0])), abs(tl_h_elem_to_half(a[1])),
                    abs(tl_h_elem_to_half(a[2])), abs(tl_h_elem_to_half(a[3])));
}

TL_DEVICE tl_bf2 add2(tl_bf2 a, tl_bf2 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_add_b_bst2_vv)
  return __musa_add_b_bst2_vv(a, b);
#else
  return make_tl_bf2(bfloat16_t(float(tl_bf_elem_to_bfloat16(a[0])) +
                                float(tl_bf_elem_to_bfloat16(b[0]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[1])) +
                                float(tl_bf_elem_to_bfloat16(b[1]))));
#endif
}

TL_DEVICE tl_bf2 sub2(tl_bf2 a, tl_bf2 b) {
  return make_tl_bf2(bfloat16_t(float(tl_bf_elem_to_bfloat16(a[0])) -
                                float(tl_bf_elem_to_bfloat16(b[0]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[1])) -
                                float(tl_bf_elem_to_bfloat16(b[1]))));
}

TL_DEVICE tl_bf2 mul2(tl_bf2 a, tl_bf2 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_mul_b_bst2_vv)
  return __musa_mul_b_bst2_vv(a, b);
#else
  return make_tl_bf2(bfloat16_t(float(tl_bf_elem_to_bfloat16(a[0])) *
                                float(tl_bf_elem_to_bfloat16(b[0]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[1])) *
                                float(tl_bf_elem_to_bfloat16(b[1]))));
#endif
}

TL_DEVICE tl_bf2 fma2(tl_bf2 a, tl_bf2 b, tl_bf2 c) {
  __mt_bfloat162 out = __hfma2(*reinterpret_cast<const __mt_bfloat162 *>(&a),
                               *reinterpret_cast<const __mt_bfloat162 *>(&b),
                               *reinterpret_cast<const __mt_bfloat162 *>(&c));
  return *reinterpret_cast<const tl_bf2 *>(&out);
}

TL_DEVICE tl_bf2 max2(tl_bf2 a, tl_bf2 b) {
  // TODO(MUSA): Re-enable __musa_max_b_bst2_vv after mtcc can select it.
  return make_tl_bf2(
      scalar_max(tl_bf_elem_to_bfloat16(a[0]), tl_bf_elem_to_bfloat16(b[0])),
      scalar_max(tl_bf_elem_to_bfloat16(a[1]), tl_bf_elem_to_bfloat16(b[1])));
}

TL_DEVICE tl_bf2 min2(tl_bf2 a, tl_bf2 b) {
  return make_tl_bf2(
      scalar_min(tl_bf_elem_to_bfloat16(a[0]), tl_bf_elem_to_bfloat16(b[0])),
      scalar_min(tl_bf_elem_to_bfloat16(a[1]), tl_bf_elem_to_bfloat16(b[1])));
}

TL_DEVICE tl_bf2 abs2(tl_bf2 a) {
  return make_tl_bf2(bfloat16_t(fabsf(float(tl_bf_elem_to_bfloat16(a[0])))),
                     bfloat16_t(fabsf(float(tl_bf_elem_to_bfloat16(a[1])))));
}

TL_DEVICE tl_bf4 add4(tl_bf4 a, tl_bf4 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_add_b_bst4_vv)
  return __musa_add_b_bst4_vv(a, b);
#else
  return make_tl_bf4(bfloat16_t(float(tl_bf_elem_to_bfloat16(a[0])) +
                                float(tl_bf_elem_to_bfloat16(b[0]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[1])) +
                                float(tl_bf_elem_to_bfloat16(b[1]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[2])) +
                                float(tl_bf_elem_to_bfloat16(b[2]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[3])) +
                                float(tl_bf_elem_to_bfloat16(b[3]))));
#endif
}

TL_DEVICE tl_bf4 sub4(tl_bf4 a, tl_bf4 b) {
  return make_tl_bf4(bfloat16_t(float(tl_bf_elem_to_bfloat16(a[0])) -
                                float(tl_bf_elem_to_bfloat16(b[0]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[1])) -
                                float(tl_bf_elem_to_bfloat16(b[1]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[2])) -
                                float(tl_bf_elem_to_bfloat16(b[2]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[3])) -
                                float(tl_bf_elem_to_bfloat16(b[3]))));
}

TL_DEVICE tl_bf4 mul4(tl_bf4 a, tl_bf4 b) {
#if TL_HAS_MUSA_BUILTIN_MP31(__musa_mul_b_bst4_vv)
  return __musa_mul_b_bst4_vv(a, b);
#else
  return make_tl_bf4(bfloat16_t(float(tl_bf_elem_to_bfloat16(a[0])) *
                                float(tl_bf_elem_to_bfloat16(b[0]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[1])) *
                                float(tl_bf_elem_to_bfloat16(b[1]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[2])) *
                                float(tl_bf_elem_to_bfloat16(b[2]))),
                     bfloat16_t(float(tl_bf_elem_to_bfloat16(a[3])) *
                                float(tl_bf_elem_to_bfloat16(b[3]))));
#endif
}

TL_DEVICE tl_bf4 fma4(tl_bf4 a, tl_bf4 b, tl_bf4 c) {
  __mt_bfloat162 lo = __hfma2(*reinterpret_cast<const __mt_bfloat162 *>(&a),
                              *reinterpret_cast<const __mt_bfloat162 *>(&b),
                              *reinterpret_cast<const __mt_bfloat162 *>(&c));
  __mt_bfloat162 hi =
      __hfma2(*(reinterpret_cast<const __mt_bfloat162 *>(&a) + 1),
              *(reinterpret_cast<const __mt_bfloat162 *>(&b) + 1),
              *(reinterpret_cast<const __mt_bfloat162 *>(&c) + 1));
  tl_bf2 lo_out = *reinterpret_cast<const tl_bf2 *>(&lo);
  tl_bf2 hi_out = *reinterpret_cast<const tl_bf2 *>(&hi);
  return make_tl_bf4(lo_out[0], lo_out[1], hi_out[0], hi_out[1]);
}

TL_DEVICE tl_bf4 max4(tl_bf4 a, tl_bf4 b) {
  // TODO(MUSA): Re-enable __musa_max_b_bst4_vv after mtcc can select it.
  return make_tl_bf4(
      scalar_max(tl_bf_elem_to_bfloat16(a[0]), tl_bf_elem_to_bfloat16(b[0])),
      scalar_max(tl_bf_elem_to_bfloat16(a[1]), tl_bf_elem_to_bfloat16(b[1])),
      scalar_max(tl_bf_elem_to_bfloat16(a[2]), tl_bf_elem_to_bfloat16(b[2])),
      scalar_max(tl_bf_elem_to_bfloat16(a[3]), tl_bf_elem_to_bfloat16(b[3])));
}

TL_DEVICE tl_bf4 min4(tl_bf4 a, tl_bf4 b) {
  return make_tl_bf4(
      scalar_min(tl_bf_elem_to_bfloat16(a[0]), tl_bf_elem_to_bfloat16(b[0])),
      scalar_min(tl_bf_elem_to_bfloat16(a[1]), tl_bf_elem_to_bfloat16(b[1])),
      scalar_min(tl_bf_elem_to_bfloat16(a[2]), tl_bf_elem_to_bfloat16(b[2])),
      scalar_min(tl_bf_elem_to_bfloat16(a[3]), tl_bf_elem_to_bfloat16(b[3])));
}

TL_DEVICE tl_bf4 abs4(tl_bf4 a) {
  return make_tl_bf4(bfloat16_t(fabsf(float(tl_bf_elem_to_bfloat16(a[0])))),
                     bfloat16_t(fabsf(float(tl_bf_elem_to_bfloat16(a[1])))),
                     bfloat16_t(fabsf(float(tl_bf_elem_to_bfloat16(a[2])))),
                     bfloat16_t(fabsf(float(tl_bf_elem_to_bfloat16(a[3])))));
}

} // namespace tl

#undef TL_HAS_MUSA_BUILTIN
#undef TL_MUSA_ARCH_GE_310
#undef TL_HAS_MUSA_BUILTIN_MP31
