#pragma once

#include <mute/arch/mma_mp22.hpp>
#include <mute/atom/mma_traits.hpp>
#include <mute/layout.hpp>

namespace mute {

namespace tl_mma_mp22_ext {

using MP22_16x16_16b_Row =
    Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
           Stride<Stride<_2, _16>, Stride<_1, _8, _128>>>;

using MP22_16x16_16b_Col =
    Layout<Shape<Shape<_4, _8>, Shape<_2, _2, _2>>,
           Stride<Stride<_32, _1>, Stride<_16, _128, _8>>>;

using MP22_16x16_32b = Layout<Shape<Shape<_8, _4>, Shape<_2, _4>>,
                              Stride<Stride<_1, _16>, Stride<_8, _64>>>;

using MP22_16x16_16b_A_Row = MP22_16x16_16b_Col;
using MP22_16x16_16b_A_Col = MP22_16x16_16b_Row;
using MP22_16x16_16b_B_Col = MP22_16x16_16b_Row;
using MP22_16x16_16b_B_Row = MP22_16x16_16b_Col;

} // namespace tl_mma_mp22_ext

#if defined(MUTE_ARCH_MMA_MP22_ENABLED)
#define TL_MP22_FMMA_M16N16K16(d, a, b, c, shape_flag)                         \
  __musa_fmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 1, shape_flag)
#define TL_MP22_BFMMA_M16N16K16(d, a, b, c, shape_flag)                        \
  __musa_bfmma_m16n16k16_mma(d, a, b, c, 0, 0, 0, 0, 0, shape_flag)
#else
#define TL_MP22_FMMA_M16N16K16(d, a, b, c, shape_flag)                         \
  MUTE_INVALID_CONTROL_PATH("Attempting to use MP22 m16n16k16 F16 MMA "        \
                            "without MUTE_ARCH_MMA_MP22_ENABLED")
#define TL_MP22_BFMMA_M16N16K16(d, a, b, c, shape_flag)                        \
  MUTE_INVALID_CONTROL_PATH("Attempting to use MP22 m16n16k16 BF16 MMA "       \
                            "without MUTE_ARCH_MMA_MP22_ENABLED")
#endif

#define TL_MP22_M16N16K16_F32F16F16F32_OP(NAME, SHAPE_FLAG)                    \
  struct NAME {                                                                \
    using DRegisters = int32_t[8];                                             \
    using ARegisters = int32_t[4];                                             \
    using BRegisters = int32_t[4];                                             \
    using CRegisters = int32_t[8];                                             \
                                                                               \
    MUTE_HOST_DEVICE static void fma(int32_t *d, int32_t const *a,             \
                                     int32_t const *b, int32_t const *c) {     \
      static_assert(SHAPE_FLAG >= 0 && SHAPE_FLAG <= 3,                        \
                    "invalid MP22 m16n16k16 shape flag");                      \
      TL_MP22_FMMA_M16N16K16(d, a, b, c, SHAPE_FLAG);                          \
    }                                                                          \
  }

TL_MP22_M16N16K16_F32F16F16F32_OP(TL_MP22_16x16x16_F32F16F16F32_TT, 0);
TL_MP22_M16N16K16_F32F16F16F32_OP(TL_MP22_16x16x16_F32F16F16F32_TN, 1);
TL_MP22_M16N16K16_F32F16F16F32_OP(TL_MP22_16x16x16_F32F16F16F32_NT, 2);
TL_MP22_M16N16K16_F32F16F16F32_OP(TL_MP22_16x16x16_F32F16F16F32_NN, 3);

#undef TL_MP22_M16N16K16_F32F16F16F32_OP

#define TL_MP22_M16N16K16_F32BF16BF16F32_OP(NAME, SHAPE_FLAG)                  \
  struct NAME {                                                                \
    using DRegisters = int32_t[8];                                             \
    using ARegisters = int32_t[4];                                             \
    using BRegisters = int32_t[4];                                             \
    using CRegisters = int32_t[8];                                             \
                                                                               \
    MUTE_HOST_DEVICE static void fma(int32_t *d, int32_t const *a,             \
                                     int32_t const *b, int32_t const *c) {     \
      static_assert(SHAPE_FLAG >= 0 && SHAPE_FLAG <= 3,                        \
                    "invalid MP22 m16n16k16 shape flag");                      \
      TL_MP22_BFMMA_M16N16K16(d, a, b, c, SHAPE_FLAG);                         \
    }                                                                          \
  }

TL_MP22_M16N16K16_F32BF16BF16F32_OP(TL_MP22_16x16x16_F32BF16BF16F32_TT, 0);
TL_MP22_M16N16K16_F32BF16BF16F32_OP(TL_MP22_16x16x16_F32BF16BF16F32_TN, 1);
TL_MP22_M16N16K16_F32BF16BF16F32_OP(TL_MP22_16x16x16_F32BF16BF16F32_NT, 2);
TL_MP22_M16N16K16_F32BF16BF16F32_OP(TL_MP22_16x16x16_F32BF16BF16F32_NN, 3);

#undef TL_MP22_M16N16K16_F32BF16BF16F32_OP

#define TL_MP22_M16N16K16_TRAITS(NAME, A_TYPE, B_TYPE, A_LAYOUT, B_LAYOUT)     \
  template <> struct MMA_Traits<NAME> {                                        \
    using ValTypeD = float;                                                    \
    using ValTypeA = A_TYPE;                                                   \
    using ValTypeB = B_TYPE;                                                   \
    using ValTypeC = float;                                                    \
                                                                               \
    using Shape_MNK = Shape<_16, _16, _16>;                                    \
    using ThrID = Layout<_32>;                                                 \
    using ALayout = tl_mma_mp22_ext::A_LAYOUT;                                 \
    using BLayout = tl_mma_mp22_ext::B_LAYOUT;                                 \
    using CLayout = tl_mma_mp22_ext::MP22_16x16_32b;                           \
  }

TL_MP22_M16N16K16_TRAITS(TL_MP22_16x16x16_F32F16F16F32_TT, half_t, half_t,
                         MP22_16x16_16b_A_Row, MP22_16x16_16b_B_Col);
TL_MP22_M16N16K16_TRAITS(TL_MP22_16x16x16_F32F16F16F32_TN, half_t, half_t,
                         MP22_16x16_16b_A_Row, MP22_16x16_16b_B_Row);
TL_MP22_M16N16K16_TRAITS(TL_MP22_16x16x16_F32F16F16F32_NT, half_t, half_t,
                         MP22_16x16_16b_A_Col, MP22_16x16_16b_B_Col);
TL_MP22_M16N16K16_TRAITS(TL_MP22_16x16x16_F32F16F16F32_NN, half_t, half_t,
                         MP22_16x16_16b_A_Col, MP22_16x16_16b_B_Row);

TL_MP22_M16N16K16_TRAITS(TL_MP22_16x16x16_F32BF16BF16F32_TT, bfloat16_t,
                         bfloat16_t, MP22_16x16_16b_A_Row,
                         MP22_16x16_16b_B_Col);
TL_MP22_M16N16K16_TRAITS(TL_MP22_16x16x16_F32BF16BF16F32_TN, bfloat16_t,
                         bfloat16_t, MP22_16x16_16b_A_Row,
                         MP22_16x16_16b_B_Row);
TL_MP22_M16N16K16_TRAITS(TL_MP22_16x16x16_F32BF16BF16F32_NT, bfloat16_t,
                         bfloat16_t, MP22_16x16_16b_A_Col,
                         MP22_16x16_16b_B_Col);
TL_MP22_M16N16K16_TRAITS(TL_MP22_16x16x16_F32BF16BF16F32_NN, bfloat16_t,
                         bfloat16_t, MP22_16x16_16b_A_Col,
                         MP22_16x16_16b_B_Row);

#undef TL_MP22_M16N16K16_TRAITS
#undef TL_MP22_FMMA_M16N16K16
#undef TL_MP22_BFMMA_M16N16K16

} // namespace mute
