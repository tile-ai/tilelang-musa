#pragma once

#include <cstdint>
#include <stdio.h>
#include <type_traits>

#include <mute/algorithm/clear.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mute/underscore.hpp>

#include "common.h"
#include "intrin.h"

namespace mute::tl_mma {

template <typename A_type, typename B_type, typename C_type, int M, int N,
          int K, int num_warp_m, int num_warp_n, int inst_m, int inst_n,
          int inst_k, bool trans_A, bool trans_B>
struct DispatchInstruction;

using _X = Underscore;

} // namespace mute::tl_mma

#define TL_DISPATCH_MMA(A_type, B_type, C_type, inst_m, inst_n, inst_k,        \
                        trans_A, trans_B, MMA_instr)                           \
  namespace mute::tl_mma {                                                     \
  template <int M, int N, int K, int num_warp_m, int num_warp_n>               \
  struct DispatchInstruction<A_type, B_type, C_type, M, N, K, num_warp_m,      \
                             num_warp_n, inst_m, inst_n, inst_k, trans_A,      \
                             trans_B> {                                        \
    using MMA = MMA_Atom<MMA_instr>;                                           \
    using MMA_Group = Tile<_X, Int<std::min(num_warp_n *inst_n, N)>, _X>;      \
  };                                                                           \
  }

#ifdef __MUSA_ARCH_LIST__
#if __MUSA_ARCH_LIST__ == 220
#include "mp22_mma_ext.h"
#include <mute/arch/mma_mp22.hpp>
TL_DISPATCH_MMA(half_t, half_t, float, 32, 32, 16, false, false,
                MP22_32x32x16_F32F16F16F32_TT)
TL_DISPATCH_MMA(half_t, half_t, float, 32, 32, 16, false, true,
                MP22_32x32x16_F32F16F16F32_TN)
TL_DISPATCH_MMA(half_t, half_t, float, 32, 32, 16, true, false,
                MP22_32x32x16_F32F16F16F32_NT)
TL_DISPATCH_MMA(half_t, half_t, float, 32, 32, 16, true, true,
                MP22_32x32x16_F32F16F16F32_NN)

TL_DISPATCH_MMA(half_t, half_t, float, 16, 16, 16, false, false,
                TL_MP22_16x16x16_F32F16F16F32_TT)
TL_DISPATCH_MMA(half_t, half_t, float, 16, 16, 16, false, true,
                TL_MP22_16x16x16_F32F16F16F32_TN)
TL_DISPATCH_MMA(half_t, half_t, float, 16, 16, 16, true, false,
                TL_MP22_16x16x16_F32F16F16F32_NT)
TL_DISPATCH_MMA(half_t, half_t, float, 16, 16, 16, true, true,
                TL_MP22_16x16x16_F32F16F16F32_NN)

TL_DISPATCH_MMA(half_t, half_t, float, 8, 32, 16, false, false,
                TL_MP22_8x32x16_F32F16F16F32_TT)
TL_DISPATCH_MMA(half_t, half_t, float, 8, 32, 16, false, true,
                TL_MP22_8x32x16_F32F16F16F32_TN)
TL_DISPATCH_MMA(half_t, half_t, float, 8, 32, 16, true, false,
                TL_MP22_8x32x16_F32F16F16F32_NT)
TL_DISPATCH_MMA(half_t, half_t, float, 8, 32, 16, true, true,
                TL_MP22_8x32x16_F32F16F16F32_NN)

TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 32, 32, 16, false, false,
                MP22_32x32x16_F32BF16BF16F32_TT)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 32, 32, 16, false, true,
                MP22_32x32x16_F32BF16BF16F32_TN)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 32, 32, 16, true, false,
                MP22_32x32x16_F32BF16BF16F32_NT)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 32, 32, 16, true, true,
                MP22_32x32x16_F32BF16BF16F32_NN)

TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 16, 16, 16, false, false,
                TL_MP22_16x16x16_F32BF16BF16F32_TT)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 16, 16, 16, false, true,
                TL_MP22_16x16x16_F32BF16BF16F32_TN)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 16, 16, 16, true, false,
                TL_MP22_16x16x16_F32BF16BF16F32_NT)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 16, 16, 16, true, true,
                TL_MP22_16x16x16_F32BF16BF16F32_NN)

TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 8, 32, 16, false, false,
                TL_MP22_8x32x16_F32BF16BF16F32_TT)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 8, 32, 16, false, true,
                TL_MP22_8x32x16_F32BF16BF16F32_TN)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 8, 32, 16, true, false,
                TL_MP22_8x32x16_F32BF16BF16F32_NT)
TL_DISPATCH_MMA(bfloat16_t, bfloat16_t, float, 8, 32, 16, true, true,
                TL_MP22_8x32x16_F32BF16BF16F32_NN)

TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, 32, 32, 8, false, false,
                MP22_32x32x8_F32TF32TF32F32_TT)
TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, 32, 32, 8, false, true,
                MP22_32x32x8_F32TF32TF32F32_TN)
TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, 32, 32, 8, true, false,
                MP22_32x32x8_F32TF32TF32F32_NT)
TL_DISPATCH_MMA(tfloat32_t, tfloat32_t, float, 32, 32, 8, true, true,
                MP22_32x32x8_F32TF32TF32F32_NN)

TL_DISPATCH_MMA(float, float, float, 32, 32, 8, false, false,
                MP22_32x32x8_F32TF32TF32F32_TT)
TL_DISPATCH_MMA(float, float, float, 32, 32, 8, false, true,
                MP22_32x32x8_F32TF32TF32F32_TN)
TL_DISPATCH_MMA(float, float, float, 32, 32, 8, true, false,
                MP22_32x32x8_F32TF32TF32F32_NT)
TL_DISPATCH_MMA(float, float, float, 32, 32, 8, true, true,
                MP22_32x32x8_F32TF32TF32F32_NN)

TL_DISPATCH_MMA(int8_t, int8_t, int, 32, 32, 32, false, false,
                MP22_32x32x32_S32S8S8S32_TT)
TL_DISPATCH_MMA(int8_t, int8_t, int, 32, 32, 32, false, true,
                MP22_32x32x32_S32S8S8S32_TN)
TL_DISPATCH_MMA(int8_t, int8_t, int, 32, 32, 32, true, false,
                MP22_32x32x32_S32S8S8S32_NT)
TL_DISPATCH_MMA(int8_t, int8_t, int, 32, 32, 32, true, true,
                MP22_32x32x32_S32S8S8S32_NN)
#endif
#endif // __MUSA_ARCH_LIST__
#undef TL_DISPATCH_MMA

namespace mute::tl_mma {

template <int N, int num_warp_n, bool transpose> struct SelectCopy {
  using type = DefaultCopy;
};

template <int Bits, int N, int K, bool K_inner, int num_warp_n, int leading_dim,
          typename Enable = void>
struct OperandTraits {
  // Primary template, use padded layout and default copy
  static constexpr int stride = leading_dim;
  static constexpr int padded =
      stride % (256 / Bits) == 0 ? stride + 128 / Bits : stride;
  using Layout = typename std::conditional<
      K_inner, Layout<Shape<Int<N>, Int<leading_dim>>, Shape<Int<padded>, _1>>,
      Layout<Shape<Int<leading_dim>, Int<K>>, Shape<_1, Int<padded>>>>::type;
  using Copy = DefaultCopy;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 64 == 32>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 3, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 16>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<1, 3, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 64 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 64 == 32>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 3, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = typename SelectCopy<N, num_warp_n, false>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 16>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<1, 3, 3>{}, Layout<Shape<_16, _8>, Stride<_1, _16>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = typename SelectCopy<N, num_warp_n, false>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<16, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 64 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 3, 3>{}, Layout<Shape<_64, _8>, Stride<_1, _64>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = typename SelectCopy<N, num_warp_n, false>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 2, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 16>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 2, 3>{}, Layout<Shape<_8, _16>, Stride<_16, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 16 == 8>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<1, 2, 3>{}, Layout<Shape<_8, _8>, Stride<_8, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 2, 3>{}, Layout<Shape<_32, _8>, Stride<_1, _32>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = UniversalCopy<tfloat32_t>;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 32 == 16>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 2, 3>{}, Layout<Shape<_16, _8>, Stride<_1, _16>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = UniversalCopy<tfloat32_t>;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<32, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 16 == 8>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<1, 2, 3>{}, Layout<Shape<_8, _8>, Stride<_1, _8>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = UniversalCopy<tfloat32_t>;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<8, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 128 == 64>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 4, 3>{}, Layout<Shape<_8, _64>, Stride<_64, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<8, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 64 == 32>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<1, 4, 3>{}, Layout<Shape<_8, _32>, Stride<_32, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<8, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 128 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 4, 3>{}, Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<64, N, K, true, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 16 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 0, 4>{}, Layout<Shape<_4, _16>, Stride<_16, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = DefaultCopy;
};

template <int N, int K, int num_warp_n, int leading_dim>
struct OperandTraits<64, N, K, false, num_warp_n, leading_dim,
                     typename std::enable_if<leading_dim % 16 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<2, 2, 2>{}, Layout<Shape<_16, _4>, Stride<_1, _16>>{}));
  using Layout = decltype(tile_to_shape(
      LayoutAtom{}, Shape<Int<leading_dim>, Int<K>>{}, Step<_2, _1>{}));
  using Copy = DefaultCopy;
};

template <typename TV, int Threads, int Values, int X, int K>
constexpr auto make_fragment_inverse_atom_layout() {
  auto logical_to_tv = right_inverse(TV{}).with_shape(Shape<Int<X>, Int<K>>{});
  auto tv_to_physical =
      Layout<Shape<Int<Threads>, Int<Values>>, Stride<Int<Values>, _1>>{};
  return composition(tv_to_physical, logical_to_tv);
}

// Preserve one aligned 16-byte producer pack while arranging the 32-bit value
// pairs required by one MMA lane at a constant shared-memory burst stride.
template <typename TV, int Threads, int Values, int X, int K,
          int PackValues = 8>
constexpr auto make_fragment_packed_atom_layout() {
  static_assert(PackValues == 8,
                "QY2 fragment-packed layout currently uses 16-byte packs");
  static_assert(Values % 2 == 0,
                "fragment-packed layout requires 32-bit value pairs");
  static_assert(Threads % (PackValues / 2) == 0,
                "fragment-packed layout requires complete lane packs");
  constexpr int Slots = PackValues / 2;
  constexpr int Groups = Threads / Slots;
  constexpr int Pairs = Values / 2;

  auto logical_to_tv = right_inverse(TV{}).with_shape(Shape<Int<X>, Int<K>>{});
  auto tv_to_physical =
      Layout<Shape<Shape<Int<Slots>, Int<Groups>>, Shape<_2, Int<Pairs>>>,
             Stride<Stride<_2, Int<PackValues * Pairs>>,
                    Stride<_1, Int<PackValues>>>>{};
  return composition(tv_to_physical, logical_to_tv);
}

template <bool Enable, typename DefaultLayout, typename TV, int Threads,
          int Values, int X, int K, int AtomXStride, int AtomKStride>
struct SelectFragmentInverseSharedLayout {
  using type = DefaultLayout;
};

template <typename DefaultLayout, typename TV, int Threads, int Values, int X,
          int K, int AtomXStride, int AtomKStride>
struct SelectFragmentInverseSharedLayout<
    true, DefaultLayout, TV, Threads, Values, X, K, AtomXStride, AtomKStride> {
  static constexpr int AtomX = Values * Threads / 16;
  using Atom = decltype(make_fragment_inverse_atom_layout<TV, Threads, Values,
                                                          AtomX, 16>());
  static_assert(AtomXStride > 0 && AtomKStride > 0,
                "fragment-inverse parent atom strides must be positive");
  using type = decltype(make_layout(
      make_shape(make_shape(shape<0>(Atom{}), Int<X / AtomX>{}),
                 make_shape(shape<1>(Atom{}), Int<K / 16>{})),
      make_stride(make_stride(stride<0>(Atom{}), Int<AtomXStride>{}),
                  make_stride(stride<1>(Atom{}), Int<AtomKStride>{}))));
};

template <bool Enable, typename DefaultLayout, typename TV, int Threads,
          int Values, int X, int K, int AtomXStride, int AtomKStride>
struct SelectFragmentPackedSharedLayout {
  using type = DefaultLayout;
};

template <typename DefaultLayout, typename TV, int Threads, int Values, int X,
          int K, int AtomXStride, int AtomKStride>
struct SelectFragmentPackedSharedLayout<
    true, DefaultLayout, TV, Threads, Values, X, K, AtomXStride, AtomKStride> {
  static constexpr int AtomX = Values * Threads / 16;
  using Atom = decltype(make_fragment_packed_atom_layout<TV, Threads, Values,
                                                         AtomX, 16>());
  static_assert(AtomXStride > 0 && AtomKStride > 0,
                "fragment-packed parent atom strides must be positive");
  using type = decltype(make_layout(
      make_shape(make_shape(shape<0>(Atom{}), Int<X / AtomX>{}),
                 make_shape(shape<1>(Atom{}), Int<K / 16>{})),
      make_stride(make_stride(stride<0>(Atom{}), Int<AtomXStride>{}),
                  make_stride(stride<1>(Atom{}), Int<AtomKStride>{}))));
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, int inst_m_override, int inst_n_override,
          int inst_k_override, int subgroup_size, int subgroup_offset,
          typename A_type_raw, typename B_type_raw, typename C_type_raw,
          bool fragment_inverse_shared_a = false,
          bool fragment_inverse_shared_b = false,
          bool fragment_packed_shared_a = false,
          bool fragment_packed_shared_b = false,
          int fragment_shared_a_physical_offset = 0,
          int fragment_shared_b_physical_offset = 0,
          int fragment_shared_a_atom_x_stride = 0,
          int fragment_shared_a_atom_k_stride = 0,
          int fragment_shared_b_atom_x_stride = 0,
          int fragment_shared_b_atom_k_stride = 0>
class GemmTensorOp {
public:
  using A_type_mute = typename tl::to_mute_type<A_type_raw>::type;
  using B_type_mute = typename tl::to_mute_type<B_type_raw>::type;
  using A_type =
      typename std::conditional<std::is_same<A_type_mute, float>::value,
                                tfloat32_t, A_type_mute>::type;
  using B_type =
      typename std::conditional<std::is_same<B_type_mute, float>::value,
                                tfloat32_t, B_type_mute>::type;
  using C_type = C_type_raw;

  static constexpr int inst_m = inst_m_override != 0 ? inst_m_override : 32;
  static constexpr int inst_n = inst_n_override != 0 ? inst_n_override : 32;
  static constexpr int inst_k = inst_k_override != 0
                                    ? inst_k_override
                                    : (sizeof_bits<A_type>::value == 8    ? 32
                                       : sizeof_bits<A_type>::value == 16 ? 16
                                                                          : 8);

  using InstructionSS =
      DispatchInstruction<A_type_raw, B_type_raw, C_type_raw, M, N, K,
                          num_warp_m, num_warp_n, inst_m, inst_n, inst_k,
                          trans_A, trans_B>;
  using InstructionRS =
      DispatchInstruction<A_type_raw, B_type_raw, C_type_raw, M, N, K,
                          num_warp_m, num_warp_n, inst_m, inst_n, inst_k,
                          trans_A, trans_B>;
  using InstructionSR =
      DispatchInstruction<A_type_raw, B_type_raw, C_type_raw, M, N, K,
                          num_warp_m, num_warp_n, inst_m, inst_n, inst_k,
                          trans_A, trans_B>;
  using InstructionRR =
      DispatchInstruction<A_type_raw, B_type_raw, C_type_raw, M, N, K,
                          num_warp_m, num_warp_n, inst_m, inst_n, inst_k,
                          trans_A, trans_B>;

  using OperandATraits = OperandTraits<sizeof_bits<A_type>::value, M, K,
                                       !trans_A, num_warp_m, lda>;

  using OperandBTraits =
      OperandTraits<sizeof_bits<B_type>::value, N, K, trans_B, num_warp_n, ldb>;

  static constexpr bool is_f16_mma = std::is_same<A_type_raw, half_t>::value &&
                                     std::is_same<B_type_raw, half_t>::value &&
                                     std::is_same<C_type_raw, float>::value;
  static constexpr bool is_bf16_mma =
      std::is_same<A_type_raw, bfloat16_t>::value &&
      std::is_same<B_type_raw, bfloat16_t>::value &&
      std::is_same<C_type_raw, float>::value;
  static constexpr bool supports_optimized_shared_layout =
      (is_f16_mma || is_bf16_mma) && inst_k == 16 &&
      ((inst_m == 8 && inst_n == 32) || (inst_m == 16 && inst_n == 16) ||
       (inst_m == 32 && inst_n == 32));
  static_assert(!fragment_inverse_shared_a || supports_optimized_shared_layout,
                "fragment-inverse shared-A layout is enabled for an "
                "unsupported QY2 MMA instruction");
  static_assert(!fragment_inverse_shared_b || supports_optimized_shared_layout,
                "fragment-inverse shared-B layout is enabled for an "
                "unsupported QY2 MMA instruction");
  static_assert(!fragment_packed_shared_a || supports_optimized_shared_layout,
                "fragment-packed shared-A layout is enabled for an "
                "unsupported QY2 MMA instruction");
  static_assert(!fragment_packed_shared_b || supports_optimized_shared_layout,
                "fragment-packed shared-B layout is enabled for an "
                "unsupported QY2 MMA instruction");
  static_assert(!fragment_inverse_shared_a || !fragment_packed_shared_a,
                "fragment-inverse and fragment-packed shared-A layouts are "
                "mutually exclusive");
  static_assert(!fragment_inverse_shared_b || !fragment_packed_shared_b,
                "fragment-inverse and fragment-packed shared-B layouts are "
                "mutually exclusive");
  static constexpr bool use_optimized_shared_a =
      fragment_inverse_shared_a || fragment_packed_shared_a;
  static constexpr bool use_optimized_shared_b =
      fragment_inverse_shared_b || fragment_packed_shared_b;
  static_assert(fragment_shared_a_physical_offset == 0 ||
                    use_optimized_shared_a,
                "a physical shared-memory offset requires an optimized "
                "shared-A layout");
  static_assert(fragment_shared_b_physical_offset == 0 ||
                    use_optimized_shared_b,
                "a physical shared-memory offset requires an optimized "
                "shared-B layout");
  static_assert(!use_optimized_shared_a ||
                    (fragment_shared_a_atom_x_stride > 0 &&
                     fragment_shared_a_atom_k_stride > 0),
                "optimized shared-A layout requires positive parent atom "
                "strides");
  static_assert(!use_optimized_shared_b ||
                    (fragment_shared_b_atom_x_stride > 0 &&
                     fragment_shared_b_atom_k_stride > 0),
                "optimized shared-B layout requires positive parent atom "
                "strides");

  using DefaultSmemLayoutA = typename OperandATraits::Layout;
  using DefaultSmemLayoutB = typename OperandBTraits::Layout;
  static constexpr int fragment_inverse_threads = inst_m == 32 ? 128 : 32;
  static constexpr int fragment_inverse_a_values =
      inst_m * inst_k / fragment_inverse_threads;
  static constexpr int fragment_inverse_b_values =
      inst_n * inst_k / fragment_inverse_threads;
  using FragmentInverseSmemLayoutA = typename SelectFragmentInverseSharedLayout<
      fragment_inverse_shared_a, DefaultSmemLayoutA,
      typename InstructionSS::MMA::LayoutA_TV, fragment_inverse_threads,
      fragment_inverse_a_values, M, K, fragment_shared_a_atom_x_stride,
      fragment_shared_a_atom_k_stride>::type;
  using FragmentInverseSmemLayoutB = typename SelectFragmentInverseSharedLayout<
      fragment_inverse_shared_b, DefaultSmemLayoutB,
      typename InstructionSS::MMA::LayoutB_TV, fragment_inverse_threads,
      fragment_inverse_b_values, N, K, fragment_shared_b_atom_x_stride,
      fragment_shared_b_atom_k_stride>::type;
  using SmemLayoutA = typename SelectFragmentPackedSharedLayout<
      fragment_packed_shared_a, FragmentInverseSmemLayoutA,
      typename InstructionSS::MMA::LayoutA_TV, fragment_inverse_threads,
      fragment_inverse_a_values, M, K, fragment_shared_a_atom_x_stride,
      fragment_shared_a_atom_k_stride>::type;
  using SmemLayoutB = typename SelectFragmentPackedSharedLayout<
      fragment_packed_shared_b, FragmentInverseSmemLayoutB,
      typename InstructionSS::MMA::LayoutB_TV, fragment_inverse_threads,
      fragment_inverse_b_values, N, K, fragment_shared_b_atom_x_stride,
      fragment_shared_b_atom_k_stride>::type;
  using SmemCopyA = Copy_Atom<typename OperandATraits::Copy, A_type>;
  using SmemCopyB = Copy_Atom<typename OperandBTraits::Copy, B_type>;

  template <typename Instruction>
  using TileMmaT = TiledMMA<typename Instruction::MMA,
                            Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>,
                            typename Instruction::MMA_Group>;

  using TileMma = TileMmaT<InstructionSS>;
  using TileMmaRS = TileMmaT<InstructionRS>;
  using TileMmaSR = TileMmaT<InstructionSR>;
  using TileMmaRR = TileMmaT<InstructionRR>;

  template <class... Args>
  static MUTE_DEVICE auto remove_swizzle(Layout<Args...> const &layout) {
    return layout;
  }
  // In fp16, when layout is KxN and n_warp is 1 and N % 64 == 0
  // the original layout fail to compile, currently using this as a workaround
  template <class... Args>
  static MUTE_DEVICE auto
  remove_swizzle(ComposedLayout<Args...> const &layout) {
    if constexpr (sizeof(A_type) == 2)
      return layout.layout_b();
    else
      return layout;
  }

  // Keep the complete physical register fragment while exposing compile-time
  // K-slice load and access operations to the pipelined consumers.
  template <typename FullFragment, typename ThreadCopy, typename TiledCopy,
            typename SourceTensor, typename Consumer>
  static MUTE_DEVICE void
  with_full_k_buffer(FullFragment &full_fragment, ThreadCopy &thread_copy,
                     TiledCopy &tiled_copy, SourceTensor &source,
                     Consumer consumer) {
    Tensor fragment = full_fragment;
    Tensor copy_view = thread_copy.retile_D(fragment);
    auto mma_view =
        make_tensor(fragment.data(), remove_swizzle(fragment.layout()));
    auto load = [&](auto k) {
      copy(tiled_copy, source(_, _, k), copy_view(_, _, k));
    };
    auto get_fragment = [&](auto k) { return mma_view(_, _, k); };
    consumer(load, get_fragment);
  }

  static MUTE_DEVICE bool in_subgroup() {
    if constexpr (subgroup_size > 0) {
      return threadIdx.x >= subgroup_offset &&
             threadIdx.x < subgroup_offset + subgroup_size;
    }
    return true;
  }

  static MUTE_DEVICE int local_thread_idx() {
    if constexpr (subgroup_size > 0) {
      return threadIdx.x - subgroup_offset;
    }
    return threadIdx.x;
  }

  template <int offset, int NN, int KK, bool trans, int lddim, typename Engine0,
            typename Layout0>
  static MUTE_DEVICE auto get_region_tensor(Tensor<Engine0, Layout0> &sa) {
    if constexpr (offset == 0) {
      using NewLayout =
          Layout<Shape<Int<NN>, Int<KK>>,
                 Stride<_1, typename std::conditional<trans, Int<NN>,
                                                      Int<lddim>>::type>>;
      auto combined_layout = composition(sa.layout(), NewLayout{});
      return make_tensor(sa.data(), combined_layout);
    } else {
      if constexpr (trans) {
        static_assert(offset % KK == 0, "Offset must be a multiple of K");
        constexpr int offset_n = offset / KK;
        return flat_divide(sa, Shape<Int<NN>, Int<KK>>{})(_, _, _0{},
                                                          Int<offset_n>{});
      } else {
        static_assert(offset % NN == 0, "Offset must be a multiple of N");
        constexpr int offset_n = offset / NN;
        return flat_divide(sa, Shape<Int<NN>, Int<KK>>{})(_, _, Int<offset_n>{},
                                                          _0{});
      }
    }
  }

  static MUTE_DEVICE auto make_shared_a_tensor(A_type_raw *pA) {
    if constexpr (use_optimized_shared_a) {
      auto *base =
          reinterpret_cast<A_type *>(pA) + fragment_shared_a_physical_offset;
      Tensor all = make_tensor(make_smem_ptr(base), SmemLayoutA{});
      constexpr int local_lda = trans_A ? M : K;
      return get_region_tensor<0, M, K, !trans_A, local_lda>(all);
    } else {
      Tensor all = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                               SmemLayoutA{});
      return get_region_tensor<offset_a, M, K, !trans_A, lda>(all);
    }
  }

  static MUTE_DEVICE auto make_shared_b_tensor(B_type_raw *pB) {
    if constexpr (use_optimized_shared_b) {
      auto *base =
          reinterpret_cast<B_type *>(pB) + fragment_shared_b_physical_offset;
      Tensor all = make_tensor(make_smem_ptr(base), SmemLayoutB{});
      constexpr int local_ldb = trans_B ? K : N;
      return get_region_tensor<0, N, K, trans_B, local_ldb>(all);
    } else {
      Tensor all = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                               SmemLayoutB{});
      return get_region_tensor<offset_b, N, K, trans_B, ldb>(all);
    }
  }

  static MUTE_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
    if (!in_subgroup()) {
      return;
    }
    const int tid = local_thread_idx();

    Tensor sA = make_shared_a_tensor(pA);
    Tensor sB = make_shared_b_tensor(pB);
    TileMma tiled_mma;
    if constexpr (subgroup_size > 0) {
      if (tid >= int(size(tiled_mma))) {
        return;
      }
    }
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrA = thr_mma.partition_fragment_A(sA);
    Tensor tCrB = thr_mma.partition_fragment_B(sB);
    Tensor tCsA = thr_copy_A.partition_S(sA);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor tCrA_copy_view = thr_copy_A.retile_D(tCrA);
    Tensor tCrB_copy_view = thr_copy_B.retile_D(tCrB);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    // when layout is KxN and n_warp is 1, there seem to be a bug, use this as a
    // workaround
    auto tCrA_view = make_tensor(tCrA.data(), remove_swizzle(tCrA.layout()));
    auto tCrB_view = make_tensor(tCrB.data(), remove_swizzle(tCrB.layout()));
    if constexpr (clear_accum) {
      clear(acc);
    }
    MUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      copy(tiled_copy_A, tCsA(_, _, k), tCrA_copy_view(_, _, k));
      copy(tiled_copy_B, tCsB(_, _, k), tCrB_copy_view(_, _, k));
      gemm(tiled_mma, tCrA_view(_, _, k), tCrB_view(_, _, k), acc);
    }
  }

  static MUTE_DEVICE void body_rs(A_type_raw *pA, B_type_raw *pB,
                                  C_type_raw *pC) {
    if (!in_subgroup()) {
      return;
    }
    const int tid = local_thread_idx();
    Tensor sB = make_shared_b_tensor(pB);
    TileMmaRS tiled_mma;
    if constexpr (subgroup_size > 0) {
      if (tid >= int(size(tiled_mma))) {
        return;
      }
    }
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_B = make_tiled_copy_B(SmemCopyB{}, tiled_mma);
    auto thr_copy_B = tiled_copy_B.get_thread_slice(tid);

    Tensor tCrB_full = thr_mma.partition_fragment_B(sB);
    Tensor tCsB = thr_copy_B.partition_S(sB);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrA =
        make_tensor(make_rmem_ptr(reinterpret_cast<A_type *>(pA)),
                    partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));
    if constexpr (clear_accum) {
      clear(acc);
    }
    constexpr int k_slices = decltype(size<2>(tCrB_full))::value;
    with_full_k_buffer(tCrB_full, thr_copy_B, tiled_copy_B, tCsB,
                       [&](auto load_b, auto fragment_b) {
                         load_b(Int<0>{});
                         for_each(make_seq<k_slices>{}, [&](auto k) {
                           constexpr int next = decltype(k)::value + 1;
                           if constexpr (next < k_slices) {
                             load_b(Int<next>{});
                           }
                           gemm(tiled_mma, tCrA(_, _, k), fragment_b(k), acc);
                         });
                       });
  }

  static MUTE_DEVICE void body_sr(A_type_raw *pA, B_type_raw *pB,
                                  C_type_raw *pC) {
    if (!in_subgroup()) {
      return;
    }
    const int tid = local_thread_idx();
    Tensor sA = make_shared_a_tensor(pA);
    TileMmaSR tiled_mma;
    if constexpr (subgroup_size > 0) {
      if (tid >= int(size(tiled_mma))) {
        return;
      }
    }
    auto thr_mma = tiled_mma.get_thread_slice(tid);
    auto tiled_copy_A = make_tiled_copy_A(SmemCopyA{}, tiled_mma);
    auto thr_copy_A = tiled_copy_A.get_thread_slice(tid);

    Tensor tCrA_full = thr_mma.partition_fragment_A(sA);
    Tensor tCsA = thr_copy_A.partition_S(sA);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrB =
        make_tensor(make_rmem_ptr(reinterpret_cast<B_type *>(pB)),
                    partition_shape_B(tiled_mma, Shape<Int<N>, Int<K>>{}));
    if constexpr (clear_accum) {
      clear(acc);
    }
    constexpr int k_slices = decltype(size<2>(tCrA_full))::value;
    with_full_k_buffer(tCrA_full, thr_copy_A, tiled_copy_A, tCsA,
                       [&](auto load_a, auto fragment_a) {
                         load_a(Int<0>{});
                         for_each(make_seq<k_slices>{}, [&](auto k) {
                           constexpr int next = decltype(k)::value + 1;
                           if constexpr (next < k_slices) {
                             load_a(Int<next>{});
                           }
                           gemm(tiled_mma, fragment_a(k), tCrB(_, _, k), acc);
                         });
                       });
  }

  static MUTE_DEVICE void body_rr(A_type_raw *pA, B_type_raw *pB,
                                  C_type_raw *pC) {
    if (!in_subgroup()) {
      return;
    }
    const int tid = local_thread_idx();
    TileMmaRR tiled_mma;
    if constexpr (subgroup_size > 0) {
      if (tid >= int(size(tiled_mma))) {
        return;
      }
    }

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));
    Tensor tCrA =
        make_tensor(make_rmem_ptr(reinterpret_cast<A_type *>(pA)),
                    partition_shape_A(tiled_mma, Shape<Int<M>, Int<K>>{}));
    Tensor tCrB =
        make_tensor(make_rmem_ptr(reinterpret_cast<B_type *>(pB)),
                    partition_shape_B(tiled_mma, Shape<Int<N>, Int<K>>{}));

    if constexpr (clear_accum) {
      clear(acc);
    }
    MUTE_UNROLL
    for (int k = 0; k < size<2>(tCrA); ++k) {
      gemm(tiled_mma, tCrA(_, _, k), tCrB(_, _, k), acc);
    }
  }
};

} // namespace mute::tl_mma

namespace tl::tl_mma {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, int inst_m = 0, int inst_n = 0, int inst_k = 0,
          int subgroup_size = 0, int subgroup_offset = 0,
          bool fragment_inverse_shared_a = false,
          bool fragment_inverse_shared_b = false,
          bool fragment_packed_shared_a = false,
          bool fragment_packed_shared_b = false,
          int fragment_shared_a_physical_offset = 0,
          int fragment_shared_b_physical_offset = 0,
          int fragment_shared_a_atom_x_stride = 0,
          int fragment_shared_a_atom_k_stride = 0,
          int fragment_shared_b_atom_x_stride = 0,
          int fragment_shared_b_atom_k_stride = 0, typename A_type,
          typename B_type, typename C_type>
MUTLASS_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {
  using MMA = mute::tl_mma::GemmTensorOp<
      M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, clear_accum, lda, ldb,
      offset_a, offset_b, inst_m, inst_n, inst_k, subgroup_size,
      subgroup_offset, A_type, B_type, C_type, fragment_inverse_shared_a,
      fragment_inverse_shared_b, fragment_packed_shared_a,
      fragment_packed_shared_b, fragment_shared_a_physical_offset,
      fragment_shared_b_physical_offset, fragment_shared_a_atom_x_stride,
      fragment_shared_a_atom_k_stride, fragment_shared_b_atom_x_stride,
      fragment_shared_b_atom_k_stride>;
  MMA::body(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, int inst_m = 0, int inst_n = 0, int inst_k = 0,
          int subgroup_size = 0, int subgroup_offset = 0,
          bool fragment_inverse_shared_a = false,
          bool fragment_inverse_shared_b = false,
          bool fragment_packed_shared_a = false,
          bool fragment_packed_shared_b = false,
          int fragment_shared_a_physical_offset = 0,
          int fragment_shared_b_physical_offset = 0,
          int fragment_shared_a_atom_x_stride = 0,
          int fragment_shared_a_atom_k_stride = 0,
          int fragment_shared_b_atom_x_stride = 0,
          int fragment_shared_b_atom_k_stride = 0, typename A_type,
          typename B_type, typename C_type>
MUTLASS_DEVICE void gemm_rs(A_type *pA, B_type *pB, C_type *accum) {
  using MMA = mute::tl_mma::GemmTensorOp<
      M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, clear_accum, lda, ldb,
      offset_a, offset_b, inst_m, inst_n, inst_k, subgroup_size,
      subgroup_offset, A_type, B_type, C_type, fragment_inverse_shared_a,
      fragment_inverse_shared_b, fragment_packed_shared_a,
      fragment_packed_shared_b, fragment_shared_a_physical_offset,
      fragment_shared_b_physical_offset, fragment_shared_a_atom_x_stride,
      fragment_shared_a_atom_k_stride, fragment_shared_b_atom_x_stride,
      fragment_shared_b_atom_k_stride>;
  MMA::body_rs(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, int inst_m = 0, int inst_n = 0, int inst_k = 0,
          int subgroup_size = 0, int subgroup_offset = 0,
          bool fragment_inverse_shared_a = false,
          bool fragment_inverse_shared_b = false,
          bool fragment_packed_shared_a = false,
          bool fragment_packed_shared_b = false,
          int fragment_shared_a_physical_offset = 0,
          int fragment_shared_b_physical_offset = 0,
          int fragment_shared_a_atom_x_stride = 0,
          int fragment_shared_a_atom_k_stride = 0,
          int fragment_shared_b_atom_x_stride = 0,
          int fragment_shared_b_atom_k_stride = 0, typename A_type,
          typename B_type, typename C_type>
MUTLASS_DEVICE void gemm_sr(A_type *pA, B_type *pB, C_type *accum) {
  using MMA = mute::tl_mma::GemmTensorOp<
      M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, clear_accum, lda, ldb,
      offset_a, offset_b, inst_m, inst_n, inst_k, subgroup_size,
      subgroup_offset, A_type, B_type, C_type, fragment_inverse_shared_a,
      fragment_inverse_shared_b, fragment_packed_shared_a,
      fragment_packed_shared_b, fragment_shared_a_physical_offset,
      fragment_shared_b_physical_offset, fragment_shared_a_atom_x_stride,
      fragment_shared_a_atom_k_stride, fragment_shared_b_atom_x_stride,
      fragment_shared_b_atom_k_stride>;
  MMA::body_sr(pA, pB, accum);
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, int inst_m = 0, int inst_n = 0, int inst_k = 0,
          int subgroup_size = 0, int subgroup_offset = 0, typename A_type,
          typename B_type, typename C_type>
MUTLASS_DEVICE void gemm_rr(A_type *pA, B_type *pB, C_type *accum) {
  using MMA = mute::tl_mma::GemmTensorOp<
      M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, clear_accum, lda, ldb,
      offset_a, offset_b, inst_m, inst_n, inst_k, subgroup_size,
      subgroup_offset, A_type, B_type, C_type>;
  MMA::body_rr(pA, pB, accum);
}

} // namespace tl::tl_mma

namespace tl {
namespace tl_gemmsp {

template <typename A, typename MetaT> struct SparseDecodeTraits {
  using Meta = typename std::remove_cv<MetaT>::type;
  static_assert(sizeof(A) == 1 || sizeof(A) == 2 || sizeof(A) == 4,
                "gemm_sp register decode only supports 8/16/32-bit A.");
  static_assert(std::is_integral_v<Meta>,
                "gemm_sp expects integral sparse metadata.");
  static constexpr int kBits = sizeof(A) * 8;
  static constexpr int kMetaBits = sizeof(Meta) * 8;
  static constexpr int kStep = kBits == 32 ? 2 : 4;
  static constexpr int kCompressedPerBlock = kBits == 32 ? 1 : 2;
  static constexpr int kExpandedPerBlock = kBits == 32 ? 2 : 4;
  static constexpr int kSelectorBits = 4;
  static constexpr int kSelectorsPerMeta = kMetaBits / kSelectorBits;
  static_assert(kMetaBits % kSelectorBits == 0,
                "gemm_sp sparse metadata width must be divisible by 4 bits.");
  static constexpr int kDecompressedPerMeta =
      kSelectorsPerMeta * kExpandedPerBlock;
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_E, bool trans_B, bool clear_accum, int lda, int ldb,
          int offset_a, int offset_b, typename A_type, typename B_type,
          typename C_type, typename E_type>
class GemmSPSS {
public:
  static_assert(offset_a == 0 && offset_b == 0,
                "Sparse FMA expects zero A/B offsets.");
  static_assert((trans_A && lda == M) || (!trans_A && lda == K) || lda == 0,
                "Sparse FMA expects default contiguous stride for A.");
  static_assert((trans_B && ldb == K) || (!trans_B && ldb == N) || ldb == 0,
                "Sparse FMA expects default contiguous stride for B.");

  static constexpr int kBStride = ldb == 0 ? (trans_B ? K : N) : ldb;

  template <typename MetaT>
  static TL_DEVICE uint32_t load_sparse_meta(const MetaT *pE, int row,
                                             int meta_col, int meta_cols) {
    using Meta = typename std::remove_cv<MetaT>::type;
    static_assert(std::is_integral_v<Meta>,
                  "gemm_sp expects integral sparse metadata.");
    return static_cast<uint32_t>(
        pE[trans_E ? (meta_col * M + row) : (row * meta_cols + meta_col)]);
  }

  template <typename A, typename MetaT>
  static TL_DEVICE A sparse_decompress_value(const A *pA, const MetaT *pE,
                                             int m, int k) {
    using Traits = SparseDecodeTraits<A, MetaT>;
    const int meta_cols = K / Traits::kDecompressedPerMeta;
    const uint32_t meta =
        load_sparse_meta(pE, m, k / Traits::kDecompressedPerMeta, meta_cols);
    const int selector =
        (meta >> (((k % Traits::kDecompressedPerMeta) / Traits::kStep) * 4)) &
        0xf;
    int idx0 = selector & 0x3;
    int idx1 = selector >> 2;
    if constexpr (Traits::kCompressedPerBlock == 1) {
      idx0 /= 2;
    }
    const int in_block = k % Traits::kExpandedPerBlock;
    const bool take0 = in_block == idx0;
    const bool take1 = Traits::kCompressedPerBlock > 1 && in_block == idx1;
    if (!take0 && !take1) {
      return A{};
    }
    const int comp_group = k / Traits::kExpandedPerBlock;
    const int comp_col =
        comp_group * Traits::kCompressedPerBlock + (take1 ? 1 : 0);
    return pA[trans_A ? (m + comp_col * M) : (m * (K / 2) + comp_col)];
  }

  static TL_DEVICE void body(A_type *pA, B_type *pB, C_type *accum, E_type *pE,
                             int thread_idx_base = 0) {
    (void)num_warp_m;
    (void)num_warp_n;
    (void)thread_idx_base;

    const int tid = threadIdx.x;
    const int num_threads = blockDim.x;
    const int total = M * N;
    const int trip = (total + num_threads - 1) / num_threads;

    for (int idx_iter = 0; idx_iter < trip; ++idx_iter) {
      const int linear = idx_iter * num_threads + tid;
      if (linear >= total) {
        continue;
      }
      const int m = linear / N;
      const int n = linear - m * N;
      C_type sum = clear_accum ? C_type{} : accum[idx_iter];
      for (int k = 0; k < K; ++k) {
        const A_type a_val = sparse_decompress_value(pA, pE, m, k);
        const int b_index = trans_B ? (offset_b + n * kBStride + k)
                                    : (offset_b + k * kBStride + n);
        sum += static_cast<C_type>(a_val) * static_cast<C_type>(pB[b_index]);
      }
      accum[idx_iter] = sum;
    }
  }
};
} // namespace tl_gemmsp

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_E, bool trans_B, bool clear_accum = false, int lda = 0,
          int ldb = 0, int offset_a = 0, int offset_b = 0,
          bool use_sqmma = false, int inst_m = 0, int inst_n = 0,
          int inst_k = 0, int wg_wait = 0, int B_SG = 16, int B_SS = 256,
          int B_SL = 256, typename A_type, typename B_type, typename C_type,
          typename E_type>
TL_DEVICE void gemm_sp_ss(A_type *pA, B_type *pB, C_type *accum, E_type *pE,
                          int thread_idx_base = 0) {
  static_assert(wg_wait == 0,
                "Sparse FMA currently expects wg_wait to stay at 0.");
  (void)use_sqmma;
  (void)inst_m;
  (void)inst_n;
  (void)inst_k;
  (void)B_SG;
  (void)B_SS;
  (void)B_SL;

  using MMA =
      tl_gemmsp::GemmSPSS<M, N, K, num_warp_m, num_warp_n, trans_A, trans_E,
                          trans_B, clear_accum, lda, ldb, offset_a, offset_b,
                          A_type, B_type, C_type, E_type>;
  MMA::body(pA, pB, accum, pE, thread_idx_base);
}

} // namespace tl
