/*!
 * \file tl/backend/musa/op/gemm.cc
 * \brief MUSA implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "backend/musa/op/gemm.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/utils.h"

#include <tvm/tir/transform.h>

#include <algorithm>
#include <cmath>
#include <cstdint>
#include <cstdlib>
#include <limits>
#include <string>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace musa {

bool IsPH1SupportedFp8(DataType dtype) {
  return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2();
}

Layout MakeTransposedPH1SqmmaOperandLayout(int actual_rows, int actual_cols,
                                           int logical_rows, int logical_cols,
                                           int element_bits, bool k_inner) {
  auto base = makeGemmABLayoutPH1(logical_rows, logical_cols, logical_cols,
                                  element_bits, k_inner);
  auto mapped = base->Forward({InputPlaceholder(1), InputPlaceholder(0)});
  return Layout(Array<PrimExpr>{Integer(actual_rows), Integer(actual_cols)},
                mapped);
}

namespace {

std::string EnvString(const char *name) {
  const char *value = std::getenv(name);
  return value == nullptr ? std::string() : std::string(value);
}

std::pair<int, int> ComputeWarpGroupPartition(const GemmWarpPolicyNode &policy,
                                              int M, int N, int num_warps,
                                              int k_m_per_warp,
                                              int k_n_per_warp) {
  ICHECK(num_warps % 4 == 0) << "Warp-Group MMA requires 128*k threads.";

  int m_warp = 1, n_warp = 1;
  constexpr int kGroup = 4;

  ICHECK(M % k_m_per_warp == 0)
      << "M must be divisible by " << k_m_per_warp << ", but got " << M;
  ICHECK(N % k_n_per_warp == 0)
      << "N must be divisible by " << k_n_per_warp << ", but got " << N;

  m_warp = kGroup;
  n_warp = num_warps / m_warp;

  if (policy.isFullRow()) {
    for (int cand = num_warps; cand >= kGroup; cand -= kGroup) {
      if (M % (cand * k_m_per_warp) == 0) {
        m_warp = cand;
        n_warp = num_warps / m_warp;
        break;
      }
    }
  } else if (policy.isFullCol()) {
    int cand_n = n_warp;
    if (N % (cand_n * k_n_per_warp) != 0) {
      int max_n = N / k_n_per_warp;
      for (int n = std::min(cand_n, max_n); n >= 1; --n) {
        if (num_warps % n == 0 && (num_warps / n) % kGroup == 0) {
          n_warp = n;
          m_warp = num_warps / n_warp;
          break;
        }
      }
    }
  } else if (policy.isSquare()) {
    int max_m = M / k_m_per_warp;
    int max_n = N / k_n_per_warp;

    float ideal = N > 0 ? static_cast<float>(M) / N : 1.f;
    float best_score = std::numeric_limits<float>::max();
    int best_m = kGroup, best_n = n_warp;

    for (int m = kGroup; m <= num_warps && m <= max_m; m += kGroup) {
      if (num_warps % m)
        continue;
      int n = num_warps / m;
      if (n > max_n)
        continue;

      float m_per_warp = static_cast<float>(M) / (m * k_m_per_warp);
      float n_per_warp = static_cast<float>(N) / (n * k_n_per_warp);
      float score = std::abs(m_per_warp / n_per_warp - ideal);

      if (score < best_score) {
        best_score = score;
        best_m = m;
        best_n = n;
      }
    }
    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                            int num_warps, int k_m_per_warp, int k_n_per_warp) {
  int m_warp = 1, n_warp = 1;

  ICHECK(M % k_m_per_warp == 0)
      << "M must be divisible by " << k_m_per_warp << ", but got " << M;
  ICHECK(N % k_n_per_warp == 0)
      << "N must be divisible by " << k_n_per_warp << ", but got " << N;

  if (policy.isFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * k_m_per_warp) != 0) {
      int max_m_warps = M / k_m_per_warp;
      m_warp = max_m_warps;
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (policy.isFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * k_n_per_warp) != 0) {
      int max_n_warps = N / k_n_per_warp;
      n_warp = max_n_warps;
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (policy.isSquare()) {
    int max_m_warps = M / k_m_per_warp;
    float ideal_ratio = N > 0 ? static_cast<float>(M) / N : 1.0f;

    int best_m = 1;
    int best_n = 1;
    float best_balance = std::numeric_limits<float>::max();
    for (int m = 1; m <= max_m_warps && m <= num_warps; m++) {
      int n = num_warps / m;
      float m_per_warp = static_cast<float>(M) / (m * k_m_per_warp);
      float n_per_warp = static_cast<float>(N) / (n * k_n_per_warp);
      if (m_per_warp < 1 || n_per_warp < 1)
        continue;
      if (m * n != num_warps)
        continue;
      float balance = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (balance < best_balance) {
        best_balance = balance;
        best_m = m;
        best_n = n;
      }
    }

    m_warp = best_m;
    n_warp = best_n;
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

std::optional<std::array<int, 3>>
SelectSQMMAInstShape(const GemmNode &op, int block_size, Target target) {
  if (!TargetIsPH1(target)) {
    return std::nullopt;
  }
  if (op.a_.scope() != "shared.dyn" && op.a_.scope() != "shared") {
    return std::nullopt;
  }
  if (op.b_.scope() != "shared.dyn" && op.b_.scope() != "shared") {
    return std::nullopt;
  }
  if (op.c_.scope() != "local.fragment") {
    return std::nullopt;
  }
  int warp_size = TargetGetWarpSize(target);
  if (block_size % warp_size != 0) {
    return std::nullopt;
  }
  int num_warps = block_size / warp_size;
  if (num_warps % 4 != 0) {
    return std::nullopt;
  }
  auto warp_parts = op.policy_->computeWarpPartition(
      op.m_, op.n_, block_size, target, kGemmInstMusaSQMMA);
  int warp_m = warp_parts.first;
  int warp_n = warp_parts.second;
  if (warp_m <= 0 || warp_n <= 0) {
    return std::nullopt;
  }
  if (warp_m % 4 != 0) {
    return std::nullopt;
  }
  int warp_groups_m = warp_m / 4;
  if (warp_groups_m <= 0) {
    return std::nullopt;
  }
  if (op.m_ % (warp_m * 4) != 0) {
    return std::nullopt;
  }
  if (op.n_ % (warp_n * 8) != 0) {
    return std::nullopt;
  }
  int64_t atom_m = op.m_ / warp_groups_m;
  int64_t atom_n = op.n_ / warp_n;

  const auto &a_dtype = op.a_->dtype;
  const auto &b_dtype = op.b_->dtype;
  const auto &c_dtype = op.c_->dtype;
  const bool major_a_is_k = !op.transA_;
  const bool major_b_is_k = op.transB_;

  enum class SqmmaTypeClass : uint8_t {
    kInt8,
    kUInt8,
    kFP16,
    kBF16,
    kTF32,
    kFP8
  };
  std::optional<SqmmaTypeClass> type_class = std::nullopt;

  if (a_dtype == DataType::Float(16) && b_dtype == DataType::Float(16) &&
      c_dtype == DataType::Float(32)) {
    type_class = SqmmaTypeClass::kFP16;
  } else if (a_dtype == DataType::BFloat(16) &&
             b_dtype == DataType::BFloat(16) &&
             c_dtype == DataType::Float(32)) {
    type_class = SqmmaTypeClass::kBF16;
  } else if (a_dtype == DataType::Float(32) && b_dtype == DataType::Float(32) &&
             c_dtype == DataType::Float(32)) {
    type_class = SqmmaTypeClass::kTF32;
  } else if (IsPH1SupportedFp8(a_dtype) && IsPH1SupportedFp8(b_dtype) &&
             c_dtype == DataType::Float(32)) {
    type_class = SqmmaTypeClass::kFP8;
  } else if (a_dtype == DataType::Int(8) && b_dtype == DataType::Int(8) &&
             c_dtype == DataType::Int(32)) {
    type_class = SqmmaTypeClass::kInt8;
  } else if (a_dtype == DataType::UInt(8) && b_dtype == DataType::UInt(8) &&
             c_dtype == DataType::UInt(32)) {
    type_class = SqmmaTypeClass::kUInt8;
  } else {
    return std::nullopt;
  }

  if ((*type_class == SqmmaTypeClass::kFP16 ||
       *type_class == SqmmaTypeClass::kBF16) &&
      op.transA_ && warp_groups_m > 1) {
    return std::nullopt;
  }
  if (*type_class == SqmmaTypeClass::kTF32 &&
      (op.m_ > 128 || op.n_ > 128 || op.k_ > 32)) {
    return std::nullopt;
  }

  auto select_inst_n = [](int inst_m, SqmmaTypeClass tc) -> std::vector<int> {
    if (tc == SqmmaTypeClass::kTF32) {
      if (inst_m == 128)
        return {128, 64};
      if (inst_m == 64)
        return {64, 32, 16};
      if (inst_m == 32)
        return {64, 32};
      if (inst_m == 16)
        return {64};
      return {};
    }
    if (inst_m == 128)
      return {128, 64, 32};
    if (inst_m == 64)
      return {128, 64, 32, 16};
    if (inst_m == 32)
      return {128, 64, 32};
    if (inst_m == 16)
      return {64};
    return {};
  };

  std::vector<int> inst_k_candidates;
  if (*type_class == SqmmaTypeClass::kFP16 ||
      *type_class == SqmmaTypeClass::kBF16) {
    inst_k_candidates = {64, 32, 16};
  } else if (*type_class == SqmmaTypeClass::kTF32) {
    inst_k_candidates = {32, 16, 8};
  } else {
    inst_k_candidates = {128, 64, 32};
  }

  for (int inst_m : {128, 64, 32, 16}) {
    if (atom_m % inst_m != 0)
      continue;
    if (*type_class == SqmmaTypeClass::kTF32 && inst_m == 128 &&
        (!major_a_is_k || !major_b_is_k)) {
      continue;
    }
    for (int inst_n : select_inst_n(inst_m, *type_class)) {
      if (atom_n % inst_n != 0)
        continue;
      for (int inst_k : inst_k_candidates) {
        if (op.k_ % inst_k == 0) {
          return std::array<int, 3>{inst_m, inst_n, inst_k};
        }
      }
    }
  }

  return std::nullopt;
}

std::optional<std::array<int, 3>>
SelectPH1WmmaInstShape(const GemmNode &op, int block_size, Target target) {
  if (!TargetIsPH1(target)) {
    return std::nullopt;
  }
  const bool a_is_shared =
      op.a_.scope() == "shared.dyn" || op.a_.scope() == "shared";
  const bool b_is_shared =
      op.b_.scope() == "shared.dyn" || op.b_.scope() == "shared";
  const bool a_is_fragment = IsFragmentBuffer(op.a_);
  const bool b_is_fragment = IsFragmentBuffer(op.b_);
  if (!((a_is_shared && b_is_shared) || (a_is_fragment && b_is_fragment))) {
    return std::nullopt;
  }
  if (op.c_.scope() != "local.fragment") {
    return std::nullopt;
  }

  int warp_size = TargetGetWarpSize(target);
  if (block_size % warp_size != 0) {
    return std::nullopt;
  }

  if (op.m_ % 4 != 0 || op.n_ % 8 != 0) {
    return std::nullopt;
  }

  auto warp_parts = op.policy_->computeWarpPartition(
      op.m_, op.n_, block_size, target, kGemmInstMusaPH1WMMA);
  int warp_m = warp_parts.first;
  int warp_n = warp_parts.second;
  if (warp_m <= 0 || warp_n <= 0) {
    return std::nullopt;
  }
  if (op.m_ % warp_m != 0 || op.n_ % warp_n != 0) {
    return std::nullopt;
  }

  int warp_tile_m = op.m_ / warp_m;
  int warp_tile_n = op.n_ / warp_n;

  const auto &a_dtype = op.a_->dtype;
  const auto &b_dtype = op.b_->dtype;
  const auto &c_dtype = op.c_->dtype;

  enum class Ph1WmmaTypeClass : uint8_t {
    kF16F16F32,
    kBF16BF16F32,
    kTF32TF32F32,
    kS8S8S32,
    kU8U8U32,
    kF16S8F32,
    kBF16S8F32,
    kS8F16F32,
    kS8BF16F32,
    kFP8F32,
  };
  std::optional<Ph1WmmaTypeClass> type_class = std::nullopt;

  if (a_dtype == DataType::Float(16) && b_dtype == DataType::Float(16) &&
      c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kF16F16F32;
  } else if (a_dtype == DataType::BFloat(16) &&
             b_dtype == DataType::BFloat(16) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kBF16BF16F32;
  } else if (a_dtype == DataType::Float(32) && b_dtype == DataType::Float(32) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kTF32TF32F32;
  } else if (a_dtype == DataType::Int(8) && b_dtype == DataType::Int(8) &&
             c_dtype == DataType::Int(32)) {
    type_class = Ph1WmmaTypeClass::kS8S8S32;
  } else if (a_dtype == DataType::UInt(8) && b_dtype == DataType::UInt(8) &&
             c_dtype == DataType::UInt(32)) {
    type_class = Ph1WmmaTypeClass::kU8U8U32;
  } else if (a_dtype == DataType::Float(16) && b_dtype == DataType::Int(8) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kF16S8F32;
  } else if (a_dtype == DataType::BFloat(16) && b_dtype == DataType::Int(8) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kBF16S8F32;
  } else if (a_dtype == DataType::Int(8) && b_dtype == DataType::Float(16) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kS8F16F32;
  } else if (a_dtype == DataType::Int(8) && b_dtype == DataType::BFloat(16) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kS8BF16F32;
  } else if (IsPH1SupportedFp8(a_dtype) && IsPH1SupportedFp8(b_dtype) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kFP8F32;
  } else {
    return std::nullopt;
  }

  auto select_inst = [&](const std::vector<std::array<int, 3>> &candidates)
      -> std::optional<std::array<int, 3>> {
    for (const auto &inst : candidates) {
      if (warp_tile_m % inst[0] == 0 && warp_tile_n % inst[1] == 0 &&
          op.k_ % inst[2] == 0) {
        return inst;
      }
    }
    return std::nullopt;
  };

  if (*type_class == Ph1WmmaTypeClass::kF16F16F32 ||
      *type_class == Ph1WmmaTypeClass::kBF16BF16F32) {
    return select_inst(
        {{16, 16, 32}, {16, 16, 16}, {16, 8, 16}, {8, 16, 16}, {16, 8, 8}});
  }
  if (*type_class == Ph1WmmaTypeClass::kTF32TF32F32) {
    return select_inst({{16, 16, 16}, {16, 8, 8}, {16, 8, 4}});
  }
  if (*type_class == Ph1WmmaTypeClass::kS8S8S32 ||
      *type_class == Ph1WmmaTypeClass::kU8U8U32 ||
      *type_class == Ph1WmmaTypeClass::kFP8F32) {
    return select_inst(
        {{16, 16, 64}, {16, 16, 32}, {16, 16, 16}, {16, 8, 16}, {8, 16, 16}});
  }
  return select_inst({{16, 16, 32}, {16, 16, 16}, {16, 8, 16}, {8, 16, 16}});
}

bool AllowSQMMA(const GemmNode &op, int block_size, Target target) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  if (ctxt->GetConfig(kDisableSQMMA, Optional<Bool>()).value_or(false)) {
    return false;
  }
  return SelectSQMMAInstShape(op, block_size, target).has_value();
}

bool AllowPH1Wmma(const GemmNode &op, int block_size, Target target) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  if (ctxt->GetConfig(kDisablePH1WMMA, Optional<Bool>()).value_or(false)) {
    return false;
  }
  return SelectPH1WmmaInstShape(op, block_size, target).has_value();
}

std::optional<std::array<int, 3>>
SelectQY2MmaInstShape(const GemmNode &op, int block_size, Target target) {
  if (!TargetIsQY2(target)) {
    return std::nullopt;
  }
  const bool a_is_shared =
      op.a_.scope() == "shared.dyn" || op.a_.scope() == "shared";
  const bool b_is_shared =
      op.b_.scope() == "shared.dyn" || op.b_.scope() == "shared";
  const bool a_is_fragment = IsFragmentBuffer(op.a_);
  const bool b_is_fragment = IsFragmentBuffer(op.b_);
  if (!((a_is_shared && b_is_shared) || a_is_fragment || b_is_fragment)) {
    return std::nullopt;
  }
  if (op.c_.scope() != "local.fragment") {
    return std::nullopt;
  }
  const bool is_f16_mma = op.a_->dtype == DataType::Float(16) &&
                          op.b_->dtype == DataType::Float(16) &&
                          op.c_->dtype == DataType::Float(32);
  const bool is_bf16_mma = op.a_->dtype == DataType::BFloat(16) &&
                           op.b_->dtype == DataType::BFloat(16) &&
                           op.c_->dtype == DataType::Float(32);
  enum class Qy2MmaTypeClass : uint8_t {
    kF16F16F32,
    kBF16BF16F32,
  };
  std::optional<Qy2MmaTypeClass> type_class = std::nullopt;
  if (is_f16_mma) {
    type_class = Qy2MmaTypeClass::kF16F16F32;
  } else if (is_bf16_mma) {
    type_class = Qy2MmaTypeClass::kBF16BF16F32;
  } else {
    return std::nullopt;
  }
  if (op.k_ % 16 != 0) {
    return std::nullopt;
  }
  auto select_inst = [&](const std::vector<std::array<int, 3>> &candidates)
      -> std::optional<std::array<int, 3>> {
    for (const auto &inst : candidates) {
      if (op.m_ % inst[0] == 0 && op.n_ % inst[1] == 0 &&
          op.k_ % inst[2] == 0) {
        return inst;
      }
    }
    return std::nullopt;
  };
  auto select_qy2_inst = [&](const std::vector<std::array<int, 3>> &candidates)
      -> std::optional<std::array<int, 3>> {
    switch (*type_class) {
    case Qy2MmaTypeClass::kF16F16F32:
    case Qy2MmaTypeClass::kBF16BF16F32:
      return select_inst(candidates);
    }
    return std::nullopt;
  };

  const std::string qy2_shape = EnvString("TILELANG_MUSA_MP22_MMA_SHAPE");
  if (qy2_shape == "m8n32k16") {
    int warp_size = TargetGetWarpSize(target);
    if (block_size % warp_size != 0) {
      return std::nullopt;
    }
    return select_qy2_inst({{8, 32, 16}});
  }
  if (qy2_shape == "m32n32k16") {
    return std::nullopt;
  }
  if (qy2_shape != "m16n16k16" && op.m_ % 32 == 0 && op.n_ % 32 == 0) {
    return std::nullopt;
  }
  int warp_size = TargetGetWarpSize(target);
  if (block_size % warp_size != 0) {
    return std::nullopt;
  }
  return select_qy2_inst({{16, 16, 16}});
}

} // namespace

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    if (TargetIsQY2(target)) {
      if (SelectQY2MmaInstShape(op, block_size, target).has_value()) {
        return kGemmInstMusaQY2MMA;
      }
      return kGemmInstMusaMMA;
    }
    ICHECK(TargetIsPH1(target))
        << "Unsupported MUSA target for gemm: " << target->str();
    if (AllowSQMMA(op, block_size, target)) {
      return kGemmInstMusaSQMMA;
    }
    if (AllowPH1Wmma(op, block_size, target)) {
      return kGemmInstMusaPH1WMMA;
    }
    return kGemmInstMusaFMA;
  }

  static std::optional<std::array<int, 3>> SelectInstShape(const GemmNode &op,
                                                           int block_size,
                                                           Target target,
                                                           String gemm_inst) {
    if (gemm_inst == kGemmInstMusaSQMMA) {
      return SelectSQMMAInstShape(op, block_size, target);
    }
    if (gemm_inst == kGemmInstMusaPH1WMMA) {
      return SelectPH1WmmaInstShape(op, block_size, target);
    }
    if (gemm_inst == kGemmInstMusaQY2MMA) {
      return SelectQY2MmaInstShape(op, block_size, target);
    }
    return std::nullopt;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst,
                       std::optional<std::array<int, 3>> mma_inst_shape) {
    int num_warps = block_size / TargetGetWarpSize(target);

    int k_m_per_warp = 4;
    int k_n_per_warp = 8;
    if (TargetIsQY2(target)) {
      if (gemm_inst == kGemmInstMusaQY2MMA) {
        ICHECK(mma_inst_shape.has_value())
            << "QY2 extended MMA warp partition requires an MMA instruction "
               "shape.";
        k_m_per_warp = (*mma_inst_shape)[0];
        k_n_per_warp = (*mma_inst_shape)[1];
      } else {
        k_m_per_warp = 32;
        k_n_per_warp = 32;
      }
    } else {
      ICHECK(TargetIsPH1(target))
          << "Unsupported MUSA target for gemm: " << target->str();
      if (gemm_inst == kGemmInstMusaFMA) {
        k_m_per_warp = 1;
        k_n_per_warp = 1;
      } else if (gemm_inst == kGemmInstMusaPH1WMMA) {
        k_m_per_warp = 8;
        k_n_per_warp = 8;
      }
    }

    if (gemm_inst == kGemmInstMusaSQMMA) {
      return ComputeWarpGroupPartition(policy, M, N, num_warps, k_m_per_warp,
                                       k_n_per_warp);
    }
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, k_m_per_warp,
                                       k_n_per_warp);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    (void)gemm_inst;
    return false;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kGemmInstMusaSQMMA) {
      return "sqmma";
    }
    if (gemm_inst == kGemmInstMusaPH1WMMA) {
      return "wmma";
    }
    if (gemm_inst == kGemmInstMusaQY2MMA || gemm_inst == kGemmInstMusaMMA) {
      return "mma";
    }
    if (gemm_inst == kGemmInstMusaFMA) {
      return "fma";
    }
    return "unknown";
  }
};

} // namespace musa

namespace {

bool MatchMUSAGemmTarget(Target target) { return TargetIsMusa(target); }

bool RegisterMUSAGemm() {
  RegisterGemmImpl(GemmImpl{
      "musa.Gemm",
      MatchMUSAGemmTarget,
      musa::Gemm::SelectInst,
      musa::Gemm::ComputeWarpPartition,
      musa::Gemm::ReuseExistingSharedLayout,
      musa::Gemm::InstructionKind,
      musa::Gemm::SelectInstShape,
  });
  return true;
}

const bool musa_gemm_registered = RegisterMUSAGemm();

} // namespace

} // namespace tl
} // namespace tvm
