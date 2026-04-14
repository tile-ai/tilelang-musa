#pragma once

#include "common.h"
#include <musa_fp4.h>

using fp4_e2_t = __mt_fp4_e2m1;

class fp4_e2_2_t {
public:
  __mt_fp4x2_storage_t __x;

  TL_DEVICE fp4_e2_2_t() = default;
  TL_DEVICE fp4_e2_2_t(__mt_fp4x2_storage_t data) : __x(data) {}
  TL_DEVICE fp4_e2_2_t(__mt_fp4x2_e2m1 data) : __x(data.__x) {}

  TL_DEVICE fp4_e2_t x() const {
    fp4_e2_t out;
    out.__x = static_cast<__mt_fp4_storage_t>(__x & 0x0F);
    return out;
  }

  TL_DEVICE fp4_e2_t y() const {
    fp4_e2_t out;
    out.__x = static_cast<__mt_fp4_storage_t>((__x >> 4) & 0x0F);
    return out;
  }

  TL_DEVICE void set_x(fp4_e2_t val) {
    __x = static_cast<__mt_fp4x2_storage_t>((__x & 0xF0) | (val.__x & 0x0F));
  }

  TL_DEVICE void set_y(fp4_e2_t val) {
    __x = static_cast<__mt_fp4x2_storage_t>((__x & 0x0F) |
                                            ((val.__x & 0x0F) << 4));
  }
};

struct __MUSA_ALIGN__(2) fp4_e2_4_t {
  fp4_e2_2_t x;
  fp4_e2_2_t y;
};

struct __MUSA_ALIGN__(4) fp4_e2_8_t {
  fp4_e2_4_t x;
  fp4_e2_4_t y;
};

struct __MUSA_ALIGN__(8) fp4_e2_16_t {
  fp4_e2_8_t x;
  fp4_e2_8_t y;
};

struct __MUSA_ALIGN__(16) fp4_e2_32_t {
  fp4_e2_16_t x;
  fp4_e2_16_t y;

  TL_DEVICE fp4_e2_32_t &operator=(const ulonglong4 &rhs) {
    x.x = *(fp4_e2_8_t *)&rhs.x;
    x.y = *(fp4_e2_8_t *)&rhs.y;
    y.x = *(fp4_e2_8_t *)&rhs.z;
    y.y = *(fp4_e2_8_t *)&rhs.w;
    return *this;
  }
};

struct __MUSA_ALIGN__(32) fp4_e2_64_t {
  fp4_e2_32_t x;
  fp4_e2_32_t y;
};

TL_DEVICE fp4_e2_2_t make_fp4_e2_2_t(fp4_e2_t x, fp4_e2_t y) {
  __mt_fp4x2_storage_t packed =
      static_cast<__mt_fp4x2_storage_t>((x.__x & 0x0F) | ((y.__x & 0x0F) << 4));
  return fp4_e2_2_t(packed);
}

TL_DEVICE fp4_e2_4_t make_fp4_e2_4_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                     fp4_e2_t x3) {
  fp4_e2_4_t result;
  result.x = make_fp4_e2_2_t(x0, x1);
  result.y = make_fp4_e2_2_t(x2, x3);
  return result;
}

TL_DEVICE fp4_e2_8_t make_fp4_e2_8_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                     fp4_e2_t x3, fp4_e2_t x4, fp4_e2_t x5,
                                     fp4_e2_t x6, fp4_e2_t x7) {
  fp4_e2_8_t result;
  result.x = make_fp4_e2_4_t(x0, x1, x2, x3);
  result.y = make_fp4_e2_4_t(x4, x5, x6, x7);
  return result;
}

TL_DEVICE fp4_e2_16_t make_fp4_e2_16_t(fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2,
                                       fp4_e2_t x3, fp4_e2_t x4, fp4_e2_t x5,
                                       fp4_e2_t x6, fp4_e2_t x7, fp4_e2_t y0,
                                       fp4_e2_t y1, fp4_e2_t y2, fp4_e2_t y3,
                                       fp4_e2_t y4, fp4_e2_t y5, fp4_e2_t y6,
                                       fp4_e2_t y7) {
  fp4_e2_16_t result;
  result.x = make_fp4_e2_8_t(x0, x1, x2, x3, x4, x5, x6, x7);
  result.y = make_fp4_e2_8_t(y0, y1, y2, y3, y4, y5, y6, y7);
  return result;
}

TL_DEVICE fp4_e2_32_t make_fp4_e2_32_t(
    fp4_e2_t x0, fp4_e2_t x1, fp4_e2_t x2, fp4_e2_t x3, fp4_e2_t x4,
    fp4_e2_t x5, fp4_e2_t x6, fp4_e2_t x7, fp4_e2_t x8, fp4_e2_t x9,
    fp4_e2_t x10, fp4_e2_t x11, fp4_e2_t x12, fp4_e2_t x13, fp4_e2_t x14,
    fp4_e2_t x15, fp4_e2_t y0, fp4_e2_t y1, fp4_e2_t y2, fp4_e2_t y3,
    fp4_e2_t y4, fp4_e2_t y5, fp4_e2_t y6, fp4_e2_t y7, fp4_e2_t y8,
    fp4_e2_t y9, fp4_e2_t y10, fp4_e2_t y11, fp4_e2_t y12, fp4_e2_t y13,
    fp4_e2_t y14, fp4_e2_t y15) {
  fp4_e2_32_t result;
  result.x = make_fp4_e2_16_t(x0, x1, x2, x3, x4, x5, x6, x7, x8, x9, x10, x11,
                              x12, x13, x14, x15);
  result.y = make_fp4_e2_16_t(y0, y1, y2, y3, y4, y5, y6, y7, y8, y9, y10, y11,
                              y12, y13, y14, y15);
  return result;
}

TL_DEVICE fp4_e2_t tl_fp4_packed_load(fp4_e2_2_t *packed, int idx) {
  return (idx & 1) ? packed[idx >> 1].y() : packed[idx >> 1].x();
}

TL_DEVICE void tl_fp4_packed_store(fp4_e2_2_t *packed, int idx, fp4_e2_t val) {
  if (idx & 1) {
    packed[idx >> 1].set_y(val);
  } else {
    packed[idx >> 1].set_x(val);
  }
}
