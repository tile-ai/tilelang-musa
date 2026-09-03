#pragma once

#include <mma.h>

namespace tl {
namespace wmma {

struct f16 {
  using type = __half;
  static constexpr int bits = 16;
};
struct bf16 {
  using type = __mt_bfloat16;
  static constexpr int bits = 16;
};
struct tf32 {
  using type = float;
  static constexpr int bits = 32;
};
struct s8 {
  using type = signed char;
  static constexpr int bits = 8;
};
struct u8 {
  using type = unsigned char;
  static constexpr int bits = 8;
};
struct s4 {
  using type = signed char;
  static constexpr int bits = 4;
};
struct f32 {
  using type = float;
  static constexpr int bits = 32;
};
struct s32 {
  using type = int;
  static constexpr int bits = 32;
};
struct u32 {
  using type = unsigned;
  static constexpr int bits = 32;
};
struct e4m3 {
  using type = __mt_fp8_e4m3;
  static constexpr int bits = 8;
};
struct e5m2 {
  using type = __mt_fp8_e5m2;
  static constexpr int bits = 8;
};

template <typename ATag, typename BTag> struct instruction_type;

#define TL_MP31_WMMA_PAIR(A, B, TYPE, FMT)                                     \
  template <> struct instruction_type<A, B> {                                  \
    static constexpr int value = mtmusa::wmma::wmma_##TYPE;                    \
    static constexpr int fmt = FMT;                                            \
  }

TL_MP31_WMMA_PAIR(f16, f16, fmma, 0);
TL_MP31_WMMA_PAIR(bf16, bf16, bfmma, 0);
TL_MP31_WMMA_PAIR(tf32, tf32, tfmma, 0);
TL_MP31_WMMA_PAIR(s8, s8, smma, 1);
TL_MP31_WMMA_PAIR(u8, u8, umma, 0);
TL_MP31_WMMA_PAIR(e4m3, e4m3, e4m3, 0);
TL_MP31_WMMA_PAIR(e5m2, e5m2, e5m2, 0);
TL_MP31_WMMA_PAIR(e4m3, e5m2, e4m3_e5m2, 0);
TL_MP31_WMMA_PAIR(e5m2, e4m3, e5m2_e4m3, 0);
TL_MP31_WMMA_PAIR(f16, s8, fp16_int8, 0);
TL_MP31_WMMA_PAIR(bf16, s8, bf16_int8, 0);
TL_MP31_WMMA_PAIR(s8, f16, int8_fp16, 0);
TL_MP31_WMMA_PAIR(s8, bf16, int8_bf16, 0);
TL_MP31_WMMA_PAIR(f16, s4, fp16_int4, 0);
TL_MP31_WMMA_PAIR(bf16, s4, bf16_int4, 0);
TL_MP31_WMMA_PAIR(s4, f16, int4_fp16, 0);
TL_MP31_WMMA_PAIR(s4, bf16, int4_bf16, 0);

#undef TL_MP31_WMMA_PAIR

template <int M, int N, int K, int Fmt, int Shape, int Type>
__device__ __forceinline__ void issue(int *dst, const int *a, const int *b,
                                      const int *c, bool sat) {
  if constexpr (M == 16 && N == 8 && K == 4) {
    __musa_wmma_m16n8k4_mma(dst, a, b, c, 0, 0, 0, sat, Fmt, Shape, Type);
  } else if constexpr (M == 16 && N == 8 && K == 8) {
    __musa_wmma_m16n8k8_mma(dst, a, b, c, 0, 0, 0, sat, Fmt, Shape, Type);
  } else if constexpr (M == 16 && N == 8 && K == 16) {
    __musa_wmma_m16n8k16_mma(dst, a, b, c, 0, 0, 0, sat, Fmt, Shape, Type);
  } else if constexpr (M == 8 && N == 16 && K == 16) {
    __musa_wmma_m8n16k16_mma(dst, a, b, c, 0, 0, 0, sat, Fmt, Shape, Type);
  } else if constexpr (M == 16 && N == 16 && K == 16) {
    __musa_wmma_m16n16k16_mma(dst, a, b, c, 0, 0, 0, sat, Fmt, Shape, Type);
  } else if constexpr (M == 16 && N == 16 && K == 32) {
    __musa_wmma_m16n16k32_mma(dst, a, b, c, 0, 0, 0, sat, Fmt, Shape, Type);
  } else if constexpr (M == 16 && N == 16 && K == 64) {
    __musa_wmma_m16n16k64_mma(dst, a, b, c, 0, 0, 0, sat, Fmt, Shape, Type);
  }
}

} // namespace wmma

template <int M, int N, int K, int NumWarpM, int NumWarpN, int InstM, int InstN,
          int InstK, typename ATag, typename BTag, typename CTag, bool TransA,
          bool TransB>
__device__ __forceinline__ void wmma_rr(const void *a, const void *b, void *c,
                                        bool clear_accum) {
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ == 310
  static_assert(NumWarpM > 0 && NumWarpN > 0,
                "MP31 WMMA RR requires positive warp partitions");
  static_assert(M % NumWarpM == 0 && N % NumWarpN == 0,
                "MP31 WMMA RR requires block dimensions divisible by warps");
  static_assert((M / NumWarpM) * (N / NumWarpN) % 32 == 0,
                "MP31 WMMA C fragment must distribute over one warp");
  using Instruction = wmma::instruction_type<ATag, BTag>;
  constexpr int shape = (TransA ? 2 : 0) | (TransB ? 0 : 1);
  constexpr int warp_M = M / NumWarpM;
  constexpr int warp_N = N / NumWarpN;
  constexpr int m_tiles = warp_M / InstM;
  constexpr int n_tiles = warp_N / InstN;
  constexpr int k_tiles = K / InstK;
  constexpr int a_dwords = (InstM * InstK * ATag::bits) / 1024;
  constexpr int b_dwords = (InstN * InstK * BTag::bits) / 1024;
  constexpr int c_dwords = (InstM * InstN * CTag::bits) / 1024;
  static_assert(a_dwords > 0 && b_dwords > 0 && c_dwords > 0);

  auto *a_words = reinterpret_cast<const int *>(a);
  auto *b_words = reinterpret_cast<const int *>(b);
  auto *c_words = reinterpret_cast<int *>(c);
#pragma unroll
  for (int mt = 0; mt < m_tiles; ++mt) {
#pragma unroll
    for (int nt = 0; nt < n_tiles; ++nt) {
      int *tile_c = c_words + (mt + m_tiles * nt) * c_dwords;
      if (clear_accum) {
#pragma unroll
        for (int i = 0; i < c_dwords; ++i)
          tile_c[i] = 0;
      }
#pragma unroll
      for (int kt = 0; kt < k_tiles; ++kt) {
        const int *tile_a = a_words + (mt + m_tiles * kt) * a_dwords;
        const int *tile_b = b_words + (nt + n_tiles * kt) * b_dwords;
        wmma::issue<InstM, InstN, InstK, Instruction::fmt, shape,
                    Instruction::value>(tile_c, tile_a, tile_b, tile_c, false);
      }
    }
  }
#else
  static_assert(sizeof(ATag) == 0, "MP31 WMMA requires __MUSA_ARCH__ == 310");
#endif
}

} // namespace tl
