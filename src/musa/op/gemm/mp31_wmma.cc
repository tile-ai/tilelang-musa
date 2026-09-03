/*!
 * \file tl/musa/op/gemm/mp31_wmma.cc
 * \brief MP31 WMMA implementation hooks for the explicit wmma_gemm op.
 */

#include "musa/op/gemm/mp31_wmma.h"

#include "musa/target_utils.h"
#include "op/utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>

#include <array>
#include <cmath>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

namespace tvm {
namespace tl {
namespace musa {
namespace mp31 {
namespace {

constexpr const char *kWMMA = "musa.wmma";

enum class WMMATypeClass : uint8_t {
  kFP16,
  kFP16Accum,
  kBF16,
  kTF32,
  kS8,
  kU8,
  kFP8,
  kF16S8,
  kBF16S8,
  kS8F16,
  kS8BF16,
  kF16S4,
  kBF16S4,
  kS4F16,
  kS4BF16,
};

struct WMMAShapeMNK {
  int m;
  int n;
  int k;
};

struct WMMATile {
  int m;
  int n;
};

bool IsMP31FP8(DataType dtype) {
  return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2();
}

std::optional<WMMATypeClass> GetWMMATypeClass(const GemmNode &op) {
  const DataType a = op.a_->dtype;
  const DataType b = op.b_->dtype;
  const DataType c = op.c_->dtype;
  if (a == DataType::Float(16) && b == DataType::Float(16) &&
      c == DataType::Float(32))
    return WMMATypeClass::kFP16;
  if (a == DataType::Float(16) && b == DataType::Float(16) &&
      c == DataType::Float(16))
    return WMMATypeClass::kFP16Accum;
  if (a == DataType::BFloat(16) && b == DataType::BFloat(16) &&
      c == DataType::Float(32))
    return WMMATypeClass::kBF16;
  if (a.is_tfloat32() && b.is_tfloat32() && c == DataType::Float(32))
    return WMMATypeClass::kTF32;
  if (a == DataType::Int(8) && b == DataType::Int(8) && c == DataType::Int(32))
    return WMMATypeClass::kS8;
  if (a == DataType::UInt(8) && b == DataType::UInt(8) &&
      (c == DataType::Int(32) || c == DataType::UInt(32)))
    return WMMATypeClass::kU8;
  if (IsMP31FP8(a) && IsMP31FP8(b) && c == DataType::Float(32))
    return WMMATypeClass::kFP8;
  if (a == DataType::Float(16) && b == DataType::Int(8) &&
      c == DataType::Float(32))
    return WMMATypeClass::kF16S8;
  if (a == DataType::BFloat(16) && b == DataType::Int(8) &&
      c == DataType::Float(32))
    return WMMATypeClass::kBF16S8;
  if (a == DataType::Int(8) && b == DataType::Float(16) &&
      c == DataType::Float(32))
    return WMMATypeClass::kS8F16;
  if (a == DataType::Int(8) && b == DataType::BFloat(16) &&
      c == DataType::Float(32))
    return WMMATypeClass::kS8BF16;
  if (a == DataType::Float(16) && b == DataType::Int(4) &&
      c == DataType::Float(32))
    return WMMATypeClass::kF16S4;
  if (a == DataType::BFloat(16) && b == DataType::Int(4) &&
      c == DataType::Float(32))
    return WMMATypeClass::kBF16S4;
  if (a == DataType::Int(4) && b == DataType::Float(16) &&
      c == DataType::Float(32))
    return WMMATypeClass::kS4F16;
  if (a == DataType::Int(4) && b == DataType::BFloat(16) &&
      c == DataType::Float(32))
    return WMMATypeClass::kS4BF16;
  return std::nullopt;
}

const std::vector<WMMAShapeMNK> &GetWMMACandidates(WMMATypeClass type) {
  static const std::vector<WMMAShapeMNK> kF16 = {
      {16, 16, 32}, {16, 16, 16}, {16, 8, 16}, {8, 16, 16}, {16, 8, 8}};
  static const std::vector<WMMAShapeMNK> kTF32 = {
      {16, 16, 16}, {16, 8, 8}, {16, 8, 4}};
  static const std::vector<WMMAShapeMNK> kEightBit = {
      {16, 16, 64}, {16, 16, 32}, {16, 16, 16}, {16, 8, 16}, {8, 16, 16}};
  static const std::vector<WMMAShapeMNK> kMixed = {
      {16, 16, 32}, {16, 16, 16}, {16, 8, 16}, {8, 16, 16}};
  static const std::vector<WMMAShapeMNK> kInt4 = {{16, 16, 32}};
  switch (type) {
  case WMMATypeClass::kFP16:
  case WMMATypeClass::kBF16:
    return kF16;
  case WMMATypeClass::kFP16Accum:
    return kF16;
  case WMMATypeClass::kTF32:
    return kTF32;
  case WMMATypeClass::kS8:
  case WMMATypeClass::kU8:
  case WMMATypeClass::kFP8:
    return kEightBit;
  case WMMATypeClass::kF16S8:
  case WMMATypeClass::kBF16S8:
  case WMMATypeClass::kS8F16:
  case WMMATypeClass::kS8BF16:
    return kMixed;
  case WMMATypeClass::kF16S4:
  case WMMATypeClass::kBF16S4:
  case WMMATypeClass::kS4F16:
  case WMMATypeClass::kS4BF16:
    return kInt4;
  }
  ICHECK(false) << "Unknown MP31 WMMA type class";
  return kF16;
}

WMMATile GetWMMATile(const GemmNode &op, int block_size, int warp_m, int warp_n,
                     const Target &target) {
  ICHECK(TargetIsMP31(target)) << "WMMA is only supported on MP31";
  ICHECK(IsFragmentBuffer(op.a_) && IsFragmentBuffer(op.b_) &&
         IsFragmentBuffer(op.c_))
      << "MP31 WMMA requires A/B/C to be local fragments (rr path)";
  const int warp_size = TargetMUSAGetWarpSize(target);
  ICHECK_GT(block_size, 0);
  ICHECK_EQ(block_size % warp_size, 0)
      << "MP31 WMMA block threads must be divisible by warp size " << warp_size;
  ICHECK_GT(warp_m, 0);
  ICHECK_GT(warp_n, 0);
  ICHECK_EQ(warp_m * warp_n, block_size / warp_size);
  ICHECK_EQ(op.m_ % warp_m, 0);
  ICHECK_EQ(op.n_ % warp_n, 0);
  return {op.m_ / warp_m, op.n_ / warp_n};
}

std::optional<std::array<int, 3>> SelectNativeShape(const GemmNode &op,
                                                    const WMMATile &tile) {
  auto type = GetWMMATypeClass(op);
  if (!type.has_value())
    return std::nullopt;
  for (const WMMAShapeMNK &candidate : GetWMMACandidates(*type)) {
    if (tile.m % candidate.m == 0 && tile.n % candidate.n == 0 &&
        op.k_ % candidate.k == 0)
      return std::array<int, 3>{candidate.m, candidate.n, candidate.k};
  }
  return std::nullopt;
}

} // namespace

bool WMMA::IsInstruction(const String &gemm_inst) { return gemm_inst == kWMMA; }

String WMMA::SelectInst(const GemmNode &op, int block_size, Target target) {
  ICHECK(TargetIsMP31(target)) << "WMMA is only supported on MP31";
  auto [m_warp, n_warp] = ComputeWarpPartition(*op.policy_.get(), op.m_, op.n_,
                                               block_size, target, kWMMA);
  (void)GetInstShape(op, block_size, m_warp, n_warp, target);
  return kWMMA;
}

std::pair<int, int> WMMA::ComputeWarpPartition(const GemmWarpPolicyNode &policy,
                                               int M, int N, int block_size,
                                               Target target,
                                               String gemm_inst) {
  ICHECK(IsInstruction(gemm_inst))
      << "Unsupported MP31 WMMA instruction: " << gemm_inst;
  ICHECK(TargetIsMP31(target)) << "WMMA is only supported on MP31";
  const int warp_size = TargetMUSAGetWarpSize(target);
  ICHECK_GT(block_size, 0);
  ICHECK_EQ(block_size % warp_size, 0)
      << "MP31 WMMA block threads must be divisible by warp size " << warp_size;
  const int num_warps = block_size / warp_size;
  ICHECK_EQ(M % 4, 0) << "MP31 WMMA M must be divisible by 4";
  ICHECK_EQ(N % 8, 0) << "MP31 WMMA N must be divisible by 8";

  auto valid = [&](int m_warp, int n_warp) {
    return m_warp > 0 && n_warp > 0 && m_warp * n_warp == num_warps &&
           M % m_warp == 0 && N % n_warp == 0;
  };
  int m_warp = 0;
  int n_warp = 0;
  if (policy.IsFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    ICHECK(valid(m_warp, n_warp));
  } else if (policy.IsFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    ICHECK(valid(m_warp, n_warp));
  } else if (policy.IsSquare()) {
    const float ideal = N > 0 ? static_cast<float>(M) / N : 1.0f;
    float best = std::numeric_limits<float>::max();
    for (int m = 1; m <= num_warps; ++m) {
      if (num_warps % m != 0)
        continue;
      int n = num_warps / m;
      if (!valid(m, n))
        continue;
      float tile_ratio =
          (static_cast<float>(M) / m) / (static_cast<float>(N) / n);
      float score = std::abs(tile_ratio - ideal);
      if (score < best) {
        best = score;
        m_warp = m;
        n_warp = n;
      }
    }
  } else {
    ICHECK(0) << "MP31 WMMA does not support Free warp policy";
  }
  ICHECK_GT(m_warp, 0) << "No valid MP31 WMMA warp partition for M=" << M
                       << ", N=" << N << ", threads=" << block_size;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

Array<Integer> WMMA::GetInstShape(const GemmNode &op, int block_size,
                                  int m_warp, int n_warp,
                                  const Target &target) {
  const WMMATile tile = GetWMMATile(op, block_size, m_warp, n_warp, target);
  auto shape = SelectNativeShape(op, tile);
  ICHECK(shape.has_value())
      << "No native MP31 WMMA instruction can cover M=" << op.m_
      << ", N=" << op.n_ << ", K=" << op.k_ << " with m_warp=" << m_warp
      << ", n_warp=" << n_warp;
  return {Integer((*shape)[0]), Integer((*shape)[1]), Integer((*shape)[2])};
}

bool WMMA::ReuseExistingSharedLayout(String gemm_inst) {
  ICHECK(IsInstruction(gemm_inst))
      << "Unsupported MP31 WMMA instruction: " << gemm_inst;
  return false;
}

} // namespace mp31
} // namespace musa

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def(
      "tl.get_mp31_wmma_inst_shape",
      [](Gemm gemm, int block_size, int m_warp, int n_warp, Target target) {
        return musa::mp31::WMMA::GetInstShape(*gemm.operator->(), block_size,
                                              m_warp, n_warp, target);
      });
}

} // namespace tl
} // namespace tvm
