#pragma once

// clang-format off
#include "common.h"
#include "intrin.h"

#include <algorithm>
#include <mute/algorithm/clear.hpp>
#include <mute/arch/mma_mp31.hpp>
#include <mute/tensor.hpp>
#include <mute/atom/mma_atom.hpp>
#include <mute/underscore.hpp>
#include <mutlass/gemm/collective/collective_builder.hpp>
// clang-format on

namespace mute {

namespace tl_wmma {

using _X = Underscore;

template <typename Impl, int atom_m, int atom_n, int atom_k>
struct WmmaInstruction {
  using MMA = MMA_Atom<Impl>;
  static constexpr int kAtomM = atom_m;
  static constexpr int kAtomN = atom_n;
  static constexpr int kAtomK = atom_k;
  static constexpr bool kSupported = true;
};

template <template <TCE::Major, TCE::Major> class Impl, bool trans_A,
          bool trans_B, int atom_m, int atom_n, int atom_k>
using WmmaInstructionFor =
    WmmaInstruction<Impl<trans_A ? TCE::Major::MN : TCE::Major::K,
                         trans_B ? TCE::Major::MN : TCE::Major::K>,
                    atom_m, atom_n, atom_k>;

struct UnsupportedWmmaInstruction {
  using MMA = typename WmmaInstructionFor<MP31_16x8x8_F32F16F16F32, false,
                                          false, 16, 8, 8>::MMA;
  static constexpr int kAtomM = 16;
  static constexpr int kAtomN = 8;
  static constexpr int kAtomK = 8;
  static constexpr bool kSupported = false;
};

template <typename A_type, typename B_type, typename C_type>
struct WmmaOpSelectorImpl {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    return UnsupportedWmmaInstruction{};
  }
};

template <> struct WmmaOpSelectorImpl<half_t, half_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 8) {
      return WmmaInstructionFor<MP31_16x8x8_F32F16F16F32, trans_A, trans_B, 16,
                                8, 8>{};
    } else if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32F16F16F32, trans_A, trans_B, 16,
                                8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32F16F16F32, trans_A, trans_B, 8,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32F16F16F32, trans_A, trans_B,
                                16, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32F16F16F32, trans_A, trans_B,
                                16, 16, 32>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <> struct WmmaOpSelectorImpl<bfloat16_t, bfloat16_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 8) {
      return WmmaInstructionFor<MP31_16x8x8_F32BF16BF16F32, trans_A, trans_B,
                                16, 8, 8>{};
    } else if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32BF16BF16F32, trans_A, trans_B,
                                16, 8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32BF16BF16F32, trans_A, trans_B,
                                8, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32BF16BF16F32, trans_A, trans_B,
                                16, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32BF16BF16F32, trans_A, trans_B,
                                16, 16, 32>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <> struct WmmaOpSelectorImpl<tfloat32_t, tfloat32_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 4) {
      return WmmaInstructionFor<MP31_16x8x4_F32TF32TF32F32, trans_A, trans_B,
                                16, 8, 4>{};
    } else if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 8) {
      return WmmaInstructionFor<MP31_16x8x8_F32TF32TF32F32, trans_A, trans_B,
                                16, 8, 8>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32TF32TF32F32, trans_A, trans_B,
                                16, 16, 16>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <> struct WmmaOpSelectorImpl<int8_t, int8_t, int32_t> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_S32S8S8S32, trans_A, trans_B, 16,
                                8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_S32S8S8S32, trans_A, trans_B, 8,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_S32S8S8S32, trans_A, trans_B, 16,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_S32S8S8S32, trans_A, trans_B, 16,
                                16, 32>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 64) {
      return WmmaInstructionFor<MP31_16x16x64_S32S8S8S32, trans_A, trans_B, 16,
                                16, 64>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <> struct WmmaOpSelectorImpl<uint8_t, uint8_t, uint32_t> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_U32U8U8U32, trans_A, trans_B, 16,
                                8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_U32U8U8U32, trans_A, trans_B, 8,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_U32U8U8U32, trans_A, trans_B, 16,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_U32U8U8U32, trans_A, trans_B, 16,
                                16, 32>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 64) {
      return WmmaInstructionFor<MP31_16x16x64_U32U8U8U32, trans_A, trans_B, 16,
                                16, 64>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <> struct WmmaOpSelectorImpl<half_t, int8_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32F16S8F32, trans_A, trans_B, 16,
                                8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32F16S8F32, trans_A, trans_B, 8,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32F16S8F32, trans_A, trans_B, 16,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32F16S8F32, trans_A, trans_B, 16,
                                16, 32>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <> struct WmmaOpSelectorImpl<bfloat16_t, int8_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32BF16S8F32, trans_A, trans_B, 16,
                                8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32BF16S8F32, trans_A, trans_B, 8,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32BF16S8F32, trans_A, trans_B,
                                16, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32BF16S8F32, trans_A, trans_B,
                                16, 16, 32>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <> struct WmmaOpSelectorImpl<int8_t, half_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32S8F16F32, trans_A, trans_B, 16,
                                8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32S8F16F32, trans_A, trans_B, 8,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32S8F16F32, trans_A, trans_B, 16,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32S8F16F32, trans_A, trans_B, 16,
                                16, 32>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <> struct WmmaOpSelectorImpl<int8_t, bfloat16_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32S8BF16F32, trans_A, trans_B, 16,
                                8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32S8BF16F32, trans_A, trans_B, 8,
                                16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32S8BF16F32, trans_A, trans_B,
                                16, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32S8BF16F32, trans_A, trans_B,
                                16, 16, 32>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <>
struct WmmaOpSelectorImpl<mutlass::float_e4m3_t, mutlass::float_e4m3_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32E4M3E4M3F32, trans_A, trans_B,
                                16, 8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32E4M3E4M3F32, trans_A, trans_B,
                                8, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32E4M3E4M3F32, trans_A, trans_B,
                                16, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32E4M3E4M3F32, trans_A, trans_B,
                                16, 16, 32>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 64) {
      return WmmaInstructionFor<MP31_16x16x64_F32E4M3E4M3F32, trans_A, trans_B,
                                16, 16, 64>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <>
struct WmmaOpSelectorImpl<mutlass::float_e5m2_t, mutlass::float_e5m2_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32E5M2E5M2F32, trans_A, trans_B,
                                16, 8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32E5M2E5M2F32, trans_A, trans_B,
                                8, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32E5M2E5M2F32, trans_A, trans_B,
                                16, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32E5M2E5M2F32, trans_A, trans_B,
                                16, 16, 32>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 64) {
      return WmmaInstructionFor<MP31_16x16x64_F32E5M2E5M2F32, trans_A, trans_B,
                                16, 16, 64>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <>
struct WmmaOpSelectorImpl<mutlass::float_e4m3_t, mutlass::float_e5m2_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32E4M3E5M2F32, trans_A, trans_B,
                                16, 8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32E4M3E5M2F32, trans_A, trans_B,
                                8, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32E4M3E5M2F32, trans_A, trans_B,
                                16, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32E4M3E5M2F32, trans_A, trans_B,
                                16, 16, 32>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 64) {
      return WmmaInstructionFor<MP31_16x16x64_F32E4M3E5M2F32, trans_A, trans_B,
                                16, 16, 64>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <>
struct WmmaOpSelectorImpl<mutlass::float_e5m2_t, mutlass::float_e4m3_t, float> {
  template <bool trans_A, bool trans_B, int inst_m, int inst_n, int inst_k>
  static TL_HOST_DEVICE constexpr auto select() {
    if constexpr (inst_m == 16 && inst_n == 8 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x8x16_F32E5M2E4M3F32, trans_A, trans_B,
                                16, 8, 16>{};
    } else if constexpr (inst_m == 8 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_8x16x16_F32E5M2E4M3F32, trans_A, trans_B,
                                8, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 16) {
      return WmmaInstructionFor<MP31_16x16x16_F32E5M2E4M3F32, trans_A, trans_B,
                                16, 16, 16>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 32) {
      return WmmaInstructionFor<MP31_16x16x32_F32E5M2E4M3F32, trans_A, trans_B,
                                16, 16, 32>{};
    } else if constexpr (inst_m == 16 && inst_n == 16 && inst_k == 64) {
      return WmmaInstructionFor<MP31_16x16x64_F32E5M2E4M3F32, trans_A, trans_B,
                                16, 16, 64>{};
    } else {
      return UnsupportedWmmaInstruction{};
    }
  }
};

template <typename A_type, typename B_type, typename C_type,
          class AtomShape_MNK, TCE::Major MajorA = TCE::Major::K,
          TCE::Major MajorB = TCE::Major::K>
TL_HOST_DEVICE constexpr auto wmma_op_selector() {
  static_assert(is_static<AtomShape_MNK>::value,
                "AtomShape_MNK must be static.");
  static_assert(rank(AtomShape_MNK{}) == 3, "AtomShape_MNK must be rank 3.");
  constexpr bool trans_A = MajorA == TCE::Major::MN;
  constexpr bool trans_B = MajorB == TCE::Major::MN;
  constexpr int atom_m = size<0>(AtomShape_MNK{});
  constexpr int atom_n = size<1>(AtomShape_MNK{});
  constexpr int atom_k = size<2>(AtomShape_MNK{});
  using Selector = WmmaOpSelectorImpl<A_type, B_type, C_type>;
  using Instruction =
      decltype(Selector::template select<trans_A, trans_B, atom_m, atom_n,
                                         atom_k>());
  static_assert(Instruction::kSupported,
                "Unsupported PH1 WMMA configuration for gemm_ss");
  return Instruction{};
}

template <int N, int num_warp_n, bool transpose> struct SelectCopy {
  using type = DefaultCopy;
};

template <int N, int K, bool K_inner, int leading_dim>
struct LinearOperandLayout {
  using type = typename std::conditional<
      K_inner,
      Layout<Shape<Int<N>, Int<leading_dim>>, Shape<Int<leading_dim>, _1>>,
      Layout<Shape<Int<leading_dim>, Int<K>>,
             Shape<_1, Int<leading_dim>>>>::type;
};

template <typename Element, bool K_inner> struct LinearOperandCopy {
  using type = DefaultCopy;
};

template <> struct LinearOperandCopy<tfloat32_t, false> {
  using type = UniversalCopy<tfloat32_t>;
};

template <int Bits, int N, int K, bool K_inner, int num_warp_n, int leading_dim,
          typename Enable = void>
struct OperandTraits {
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
                     typename std::enable_if<leading_dim % 128 == 0>::type> {
  using LayoutAtom = decltype(composition(
      Swizzle<3, 4, 3>{}, Layout<Shape<_8, _128>, Stride<_128, _1>>{}));
  using Layout =
      decltype(tile_to_shape(LayoutAtom{}, Shape<Int<N>, Int<leading_dim>>{}));
  using Copy = typename SelectCopy<N, num_warp_n, true>::type;
};

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, int lda, int ldb, int offset_a,
          int offset_b, int inst_m, int inst_n, int inst_k, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
public:
  using A_type_mute = typename tl::to_mute_type<A_type_raw>::type;
  using B_type_mute = typename tl::to_mute_type<B_type_raw>::type;
  using A_type = conditional_t<std::is_same<A_type_mute, float>::value,
                               tfloat32_t, A_type_mute>;
  using B_type = conditional_t<std::is_same<B_type_mute, float>::value,
                               tfloat32_t, B_type_mute>;
  using C_type = C_type_raw;

  static_assert(num_warp_m > 0 && num_warp_n > 0,
                "PH1 WMMA requires positive warp partition counts");
  static_assert(M % num_warp_m == 0 && N % num_warp_n == 0,
                "PH1 WMMA requires block tiles to divide evenly across warps");
  static_assert(inst_m > 0 && inst_n > 0 && inst_k > 0,
                "PH1 WMMA requires lowering to provide an explicit "
                "instruction shape");
  static constexpr int kWarpShapeM = M / num_warp_m;
  static constexpr int kWarpShapeN = N / num_warp_n;
  static constexpr TCE::Major WmmaMajorA =
      trans_A ? TCE::Major::MN : TCE::Major::K;
  static constexpr TCE::Major WmmaMajorB =
      trans_B ? TCE::Major::MN : TCE::Major::K;
  using AtomShape_MNK = Shape<Int<inst_m>, Int<inst_n>, Int<inst_k>>;
  using Instruction =
      decltype(wmma_op_selector<A_type, B_type, C_type, AtomShape_MNK,
                                WmmaMajorA, WmmaMajorB>());

  // Keep PH1 WMMA shared-memory staging linear for now so it matches the
  // frontend layout hook; the shared->register retile is still handled by the
  // tiled copy path below.
  using SmemLayoutA = typename LinearOperandLayout<M, K, !trans_A, lda>::type;
  using SmemLayoutB = typename LinearOperandLayout<N, K, trans_B, ldb>::type;
  using SmemCopyA =
      Copy_Atom<typename LinearOperandCopy<A_type, !trans_A>::type, A_type>;
  using SmemCopyB =
      Copy_Atom<typename LinearOperandCopy<B_type, trans_B>::type, B_type>;
  static_assert(Instruction::kSupported,
                "Unsupported PH1 WMMA configuration for gemm_ss");
  static_assert(
      kWarpShapeM % Instruction::kAtomM == 0,
      "PH1 WMMA requires the warp-local M tile to align with the atom "
      "shape");
  static_assert(
      kWarpShapeN % Instruction::kAtomN == 0,
      "PH1 WMMA requires the warp-local N tile to align with the atom "
      "shape");
  static_assert(K % Instruction::kAtomK == 0,
                "PH1 WMMA requires K to align with the selected atom shape");
  using TileMma =
      TiledMMA<typename Instruction::MMA,
               Layout<Shape<Int<num_warp_m>, Int<num_warp_n>, _1>>,
               Tile<_X, Int<std::min(num_warp_n *Instruction::kAtomN, N)>, _X>>;

  template <class... Args>
  static TL_DEVICE auto remove_swizzle(Layout<Args...> const &layout) {
    return layout;
  }

  template <class... Args>
  static TL_DEVICE auto remove_swizzle(ComposedLayout<Args...> const &layout) {
    if constexpr (sizeof(A_type) == 2) {
      return layout.layout_b();
    } else {
      return layout;
    }
  }

  template <int offset, int NN, int KK, bool trans, int lddim, typename Engine0,
            typename Layout0>
  static TL_DEVICE auto get_region_tensor(Tensor<Engine0, Layout0> &sa) {
    if constexpr (offset == 0) {
      using NewLayout =
          Layout<Shape<Int<NN>, Int<KK>>,
                 Stride<_1, conditional_t<trans, Int<NN>, Int<lddim>>>>;
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

  static TL_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA_all = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                                SmemLayoutA{});
    Tensor sB_all = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                                SmemLayoutB{});
    Tensor sA = get_region_tensor<offset_a, M, K, !trans_A, lda>(sA_all);
    Tensor sB = get_region_tensor<offset_b, N, K, trans_B, ldb>(sB_all);
    TileMma tiled_mma;
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

  static TL_DEVICE void body_rr(A_type_raw *pA, B_type_raw *pB,
                                C_type_raw *pC) {
    TileMma tiled_mma;
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

} // namespace tl_wmma

namespace tl_sqmma {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum, typename A_type_raw,
          typename B_type_raw, typename C_type_raw>
class GemmTensorOp {
public:
  using A_type_mute = typename tl::to_mute_type<A_type_raw>::type;
  using B_type_mute = typename tl::to_mute_type<B_type_raw>::type;
  using A_type = conditional_t<std::is_same<A_type_mute, float>::value,
                               tfloat32_t, A_type_mute>;
  using B_type = conditional_t<std::is_same<B_type_mute, float>::value,
                               tfloat32_t, B_type_mute>;
  using C_type = C_type_raw;

  static constexpr TCE::Major SqmmaMajorA =
      trans_A ? TCE::Major::MN : TCE::Major::K;
  static constexpr TCE::Major SqmmaMajorB =
      trans_B ? TCE::Major::K : TCE::Major::MN;

  // Tile handled by one squad warp
  using AtomShape_MNK =
      Shape<Int<M / (num_warp_m / 4)>, Int<N / num_warp_n>, Int<K>>;
  using TileShape_MNK = Shape<Int<M>, Int<N>, Int<K>>;

  using SqmmaOp =
      decltype(mute::MP31::SQMMA::ss_op_selector<A_type, B_type, C_type,
                                                 AtomShape_MNK, SqmmaMajorA,
                                                 SqmmaMajorB>());
  using SqmmaTraits = MMA_Traits<SqmmaOp>;
  using InstructionShape_MNK = typename SqmmaTraits::Shape_MNK;

  static_assert(size<0>(TileShape_MNK{}) % size<0>(InstructionShape_MNK{}) == 0,
                "TileM must align to SQMMA M.");
  static_assert(size<1>(TileShape_MNK{}) % size<1>(InstructionShape_MNK{}) == 0,
                "TileN must align to SQMMA N.");
  static_assert(size<2>(TileShape_MNK{}) % size<2>(InstructionShape_MNK{}) == 0,
                "TileK must align to SQMMA K.");

  using AtomLayout = Layout<Shape<Int<num_warp_m / 4>, Int<num_warp_n>, _1>>;

  using TiledMma = decltype(make_tiled_mma(SqmmaOp{}, AtomLayout{}));

  using SmemLayoutAtomA =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_A<
               SqmmaMajorA, A_type, SqmmaOp, TileShape_MNK>());
  using SmemLayoutAtomB =
      decltype(mutlass::gemm::collective::detail::ss_smem_selector_B<
               SqmmaMajorB, B_type, SqmmaOp, TileShape_MNK>());

  using SmemLayoutA =
      decltype(tile_to_shape(SmemLayoutAtomA{}, Shape<Int<M>, Int<K>, _1>{}));
  using SmemLayoutB =
      decltype(tile_to_shape(SmemLayoutAtomB{}, Shape<Int<N>, Int<K>, _1>{}));

  static_assert(num_warp_m % 4 == 0,
                "num_warp_m must be a multiple of 4 for sqmma");

  template <int wg_wait = 0>
  static TL_DEVICE void body(A_type_raw *pA, B_type_raw *pB, C_type_raw *pC) {
    const int tid = threadIdx.x;
    Tensor sA = make_tensor(make_smem_ptr(reinterpret_cast<A_type *>(pA)),
                            SmemLayoutA{});
    Tensor sB = make_tensor(make_smem_ptr(reinterpret_cast<B_type *>(pB)),
                            SmemLayoutB{});
    TiledMma tiled_mma;
    auto thr_mma = tiled_mma.get_thread_slice(tid);

    // Allocate registers for pipelining
    Tensor tCsA = thr_mma.partition_A(sA); // (MMA,MMA_M,MMA_K,PIPE)
    Tensor tCsB = thr_mma.partition_B(sB); // (MMA,MMA_N,MMA_K,PIPE)

    Tensor tCrA = thr_mma.make_fragment_A(tCsA);
    Tensor tCrB = thr_mma.make_fragment_B(tCsB);

    Tensor acc =
        make_tensor(make_rmem_ptr(reinterpret_cast<C_type *>(pC)),
                    partition_shape_C(tiled_mma, Shape<Int<M>, Int<N>>{}));

    if constexpr (clear_accum) {
      tiled_mma.accumulate_ = mute::MP31::SQMMA::ScaleOut::Zero;
    }

    for (int k_block = 0; k_block < size<2>(tCrA); ++k_block) {
      // (V,M) x (V,N) => (V,M,N)
      gemm(tiled_mma, tCrA(_, _, k_block, 0), tCrB(_, _, k_block, 0), acc);
      tiled_mma.accumulate_ = mute::MP31::SQMMA::ScaleOut::One;
    }

    if constexpr (wg_wait >= 0) {
      warpsquad_wait<wg_wait>();
    }
  }
};

} // namespace tl_sqmma

} // namespace mute

namespace tl {

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, int lda = 0, int ldb = 0,
          int offset_a = 0, int offset_b = 0, bool use_sqmma = true,
          int inst_m = 0, int inst_n = 0, int inst_k = 0, int wg_wait = 0,
          typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_ss(A_type *pA, B_type *pB, C_type *accum) {
  if constexpr (use_sqmma) {
    static_assert((trans_A && lda == M) || (!trans_A && lda == K),
                  "SQMMA doesn't support custom stride for A");
    static_assert((trans_B && ldb == K) || (!trans_B && ldb == N),
                  "SQMMA doesn't support custom stride for B");
    static_assert(offset_a == 0 && offset_b == 0,
                  "offset_a and offset_b must be zero for SQMMA");
    using MMA = mute::tl_sqmma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n,
                                             trans_A, trans_B, clear_accum,
                                             A_type, B_type, C_type>;
    MMA::template body<wg_wait>(pA, pB, accum);
  } else {
    using MMA =
        mute::tl_wmma::GemmTensorOp<M, N, K, num_warp_m, num_warp_n, trans_A,
                                    trans_B, clear_accum, lda, ldb, offset_a,
                                    offset_b, inst_m, inst_n, inst_k, A_type,
                                    B_type, C_type>;
    MMA::body(pA, pB, accum);
  }
}

template <int M, int N, int K, int num_warp_m, int num_warp_n, bool trans_A,
          bool trans_B, bool clear_accum = false, int lda = 0, int ldb = 0,
          int offset_a = 0, int offset_b = 0, bool use_sqmma = true,
          int inst_m = 0, int inst_n = 0, int inst_k = 0, int wg_wait = 0,
          typename A_type, typename B_type, typename C_type>
TL_DEVICE void gemm_rr(A_type *pA, B_type *pB, C_type *accum) {
  static_assert(!use_sqmma, "PH1 SQMMA does not support gemm_rr");
  static_assert(wg_wait == 0,
                "PH1 WMMA gemm_rr currently expects wg_wait to stay at 0");
  using MMA = mute::tl_wmma::GemmTensorOp<
      M, N, K, num_warp_m, num_warp_n, trans_A, trans_B, clear_accum, lda, ldb,
      offset_a, offset_b, inst_m, inst_n, inst_k, A_type, B_type, C_type>;
  MMA::body_rr(pA, pB, accum);
}

template <int num_mma>
TL_DEVICE /**
           * Wait for all WMMA/MMA warps in the current warp-group to
           * synchronize.
           *
           * Blocks until the warp-group-wide rendezvous for `num_mma` MMA lanes
           * completes, ensuring all participating warps have arrived before
           * proceeding.
           */
    void
    wait_wgmma() {
  mute::warpsquad_wait<num_mma>();
}

} // namespace tl
