/*!
 * \file tl/backend/musa/op/gemm.cc
 * \brief MUSA implementation for tl.gemm instruction selection.
 */

#include "op/gemm.h"

#include "backend/common/target_utils.h"
#include "backend/musa/op/gemm.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "support/check.h"

#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/transform.h>

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

using namespace tirx;
using namespace ffi;

namespace musa {

bool IsPH1SupportedFp8(DataType dtype) {
  return dtype.is_float8_e4m3() || dtype.is_float8_e4m3fn() ||
         dtype.is_float8_e5m2();
}

bool IsPH1SupportedTf32(DataType dtype) {
  // Tensor instructions require an explicit TF32 dtype to reduce precision.
  return dtype.is_tfloat32();
}

Layout MakeTransposedPH1SqmmaOperandLayout(int actual_rows, int actual_cols,
                                           int logical_rows, int logical_cols,
                                           int element_bits, bool k_inner) {
  auto base = MakeSqmmaABLayout(logical_rows, logical_cols, logical_cols,
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

  if (policy.IsFullRow()) {
    for (int cand = num_warps; cand >= kGroup; cand -= kGroup) {
      if (M % (cand * k_m_per_warp) == 0) {
        m_warp = cand;
        n_warp = num_warps / m_warp;
        break;
      }
    }
  } else if (policy.IsFullCol()) {
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
  } else if (policy.IsSquare()) {
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

  if (policy.IsFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * k_m_per_warp) != 0) {
      int max_m_warps = M / k_m_per_warp;
      m_warp = max_m_warps;
      n_warp = num_warps / m_warp;
      if (n_warp == 0)
        n_warp = 1;
    }
  } else if (policy.IsFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * k_n_per_warp) != 0) {
      int max_n_warps = N / k_n_per_warp;
      n_warp = max_n_warps;
      m_warp = num_warps / n_warp;
      if (m_warp == 0)
        m_warp = 1;
    }
  } else if (policy.IsSquare()) {
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
  auto warp_parts = op.policy_->ComputeWarpPartition(
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
  } else if (IsPH1SupportedTf32(a_dtype) && IsPH1SupportedTf32(b_dtype) &&
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
SelectMP31WmmaInstShape(const GemmNode &op, int block_size, Target target) {
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

  auto warp_parts = op.policy_->ComputeWarpPartition(
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
  } else if (IsPH1SupportedTf32(a_dtype) && IsPH1SupportedTf32(b_dtype) &&
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

std::optional<std::array<int, 3>>
SelectWmmaInstShape(const GemmNode &op, int block_size, Target target) {
  if (TargetIsPH1(target)) {
    return SelectMP31WmmaInstShape(op, block_size, target);
  }
  return std::nullopt;
}

bool AllowSQMMA(const GemmNode &op, int block_size, Target target) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  if (ctxt->GetConfig(kDisableSQMMA, Optional<Bool>()).value_or(false)) {
    return false;
  }
  return SelectSQMMAInstShape(op, block_size, target).has_value();
}

bool AllowWmma(const GemmNode &op, int block_size, Target target) {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  if (ctxt->GetConfig(kDisableWMMA, Optional<Bool>()).value_or(false)) {
    return false;
  }
  return SelectWmmaInstShape(op, block_size, target).has_value();
}

} // namespace

int GetQY2MmaWarpSize(const std::optional<std::array<int, 3>> &mma_inst_shape) {
  if (!mma_inst_shape.has_value()) {
    return 128;
  }
  const auto &shape = *mma_inst_shape;
  if ((shape[0] == 8 && shape[1] == 32) || (shape[0] == 16 && shape[1] == 16)) {
    ICHECK_EQ(shape[2], 16) << "Unsupported QY2 MMA instruction shape: m"
                            << shape[0] << "n" << shape[1] << "k" << shape[2];
    return 32;
  }
  ICHECK(shape[0] == 32 && shape[1] == 32 &&
         (shape[2] == 8 || shape[2] == 16 || shape[2] == 32))
      << "Unsupported QY2 MMA instruction shape: m" << shape[0] << "n"
      << shape[1] << "k" << shape[2];
  return 128;
}

namespace {

std::vector<std::pair<int, int>>
GetQY2MmaWarpPartitions(int M, int N, int block_size,
                        const std::array<int, 3> &inst_shape) {
  std::vector<std::pair<int, int>> partitions;
  const int warp_size = GetQY2MmaWarpSize(inst_shape);
  if (block_size <= 0 || block_size % warp_size != 0 ||
      M % inst_shape[0] != 0 || N % inst_shape[1] != 0) {
    return partitions;
  }

  const int num_warps = block_size / warp_size;
  for (int m_warp = 1; m_warp <= num_warps; ++m_warp) {
    if (num_warps % m_warp != 0) {
      continue;
    }
    const int n_warp = num_warps / m_warp;
    if (M % (m_warp * inst_shape[0]) == 0 &&
        N % (n_warp * inst_shape[1]) == 0) {
      partitions.emplace_back(m_warp, n_warp);
    }
  }
  return partitions;
}

std::pair<int, int>
SelectQY2MmaWarpPartition(const GemmWarpPolicy &policy, int M, int N,
                          const std::array<int, 3> &inst_shape,
                          const std::vector<std::pair<int, int>> &partitions) {
  ICHECK(!partitions.empty()) << "No valid QY2 MMA warp partition";

  std::pair<int, int> selected = partitions.front();
  if (policy->IsFullRow()) {
    for (const auto &partition : partitions) {
      if (partition.first > selected.first) {
        selected = partition;
      }
    }
  } else if (policy->IsFullCol()) {
    for (const auto &partition : partitions) {
      if (partition.second > selected.second) {
        selected = partition;
      }
    }
  } else if (policy->IsSquare()) {
    const float ideal_ratio =
        N > 0 ? static_cast<float>(M) / static_cast<float>(N) : 1.0f;
    float best_score = std::numeric_limits<float>::max();
    for (const auto &[m_warp, n_warp] : partitions) {
      const float m_per_warp = static_cast<float>(M) / (m_warp * inst_shape[0]);
      const float n_per_warp = static_cast<float>(N) / (n_warp * inst_shape[1]);
      const float score = std::abs(m_per_warp / n_per_warp - ideal_ratio);
      if (score < best_score) {
        best_score = score;
        selected = {m_warp, n_warp};
      }
    }
  } else {
    ICHECK(0) << "Unknown GemmWarpPolicy";
  }
  return selected;
}

} // namespace

std::optional<std::array<int, 3>>
SelectQY2MmaInstShape(const Buffer &a, const Buffer &b, const Buffer &c, int M,
                      int N, int K, int block_size, Target target) {
  if (!TargetIsQY2(target)) {
    return std::nullopt;
  }
  const std::string qy2_shape = EnvString("TILELANG_MUSA_MP22_MMA_SHAPE");
  ICHECK(qy2_shape.empty() || qy2_shape == "m8n32k16" ||
         qy2_shape == "m16n16k16" || qy2_shape == "m32n32k16")
      << "Invalid TILELANG_MUSA_MP22_MMA_SHAPE: " << qy2_shape
      << ". Expected m8n32k16, m16n16k16, or m32n32k16.";
  const bool a_is_shared = IsSharedBuffer(a);
  const bool b_is_shared = IsSharedBuffer(b);
  const bool a_is_fragment = IsFragmentBuffer(a);
  const bool b_is_fragment = IsFragmentBuffer(b);
  ICHECK((a_is_shared && b_is_shared) || a_is_fragment || b_is_fragment)
      << "QY2 MMA requires both A and B in shared scope, or at least one "
         "operand in local.fragment scope";
  ICHECK(IsFragmentBuffer(c))
      << "QY2 MMA requires C in local.fragment scope, got " << c.scope();
  const bool is_f16_mma = a->dtype == DataType::Float(16) &&
                          b->dtype == DataType::Float(16) &&
                          c->dtype == DataType::Float(32);
  const bool is_bf16_mma = a->dtype == DataType::BFloat(16) &&
                           b->dtype == DataType::BFloat(16) &&
                           c->dtype == DataType::Float(32);
  const bool is_tf32_mma =
      a->dtype == b->dtype &&
      (a->dtype == DataType::Float(32) || a->dtype.is_tfloat32()) &&
      c->dtype == DataType::Float(32);
  const bool is_i8_mma = a->dtype == DataType::Int(8) &&
                         b->dtype == DataType::Int(8) &&
                         c->dtype == DataType::Int(32);
  ICHECK(is_f16_mma || is_bf16_mma || is_tf32_mma || is_i8_mma)
      << "Unsupported QY2 MMA dtype combination: " << a->dtype << ", "
      << b->dtype << " -> " << c->dtype;
  const bool supports_extended_mma = is_f16_mma || is_bf16_mma;
  const std::array<int, 3> default_inst =
      is_i8_mma ? std::array<int, 3>{32, 32, 32}
                : (is_tf32_mma ? std::array<int, 3>{32, 32, 8}
                               : std::array<int, 3>{32, 32, 16});
  auto select_inst =
      [&](const std::array<int, 3> &inst) -> std::optional<std::array<int, 3>> {
    if (K % inst[2] == 0 &&
        !GetQY2MmaWarpPartitions(M, N, block_size, inst).empty()) {
      return inst;
    }
    return std::nullopt;
  };

  if (!qy2_shape.empty()) {
    // Wave32 overrides only apply to FP16/BF16 extended MMA. Other supported
    // dtype combinations keep their native m32 instruction.
    std::array<int, 3> requested_inst = default_inst;
    if (supports_extended_mma) {
      if (qy2_shape == "m16n16k16") {
        requested_inst = {16, 16, 16};
      } else if (qy2_shape == "m8n32k16") {
        requested_inst = {8, 32, 16};
      }
    }
    auto inst = select_inst(requested_inst);
    ICHECK(inst.has_value())
        << "Explicit QY2 MMA shape " << qy2_shape
        << " is incompatible with GEMM shape m" << M << "n" << N << "k" << K
        << " and block size " << block_size;
    return inst;
  }

  // The native m32 instruction uses one 128-thread warp. Its K dimension is
  // selected from the input dtype. Prefer it whenever both the tile shape and
  // block size support it. The extended m16n16k16 and m8n32k16 instructions
  // use 32-thread warps and serve as FP16/BF16 fallbacks.
  if (auto inst = select_inst(default_inst); inst.has_value()) {
    return inst;
  }
  if (supports_extended_mma) {
    if (auto inst = select_inst({16, 16, 16}); inst.has_value()) {
      return inst;
    }
    if (auto inst = select_inst({8, 32, 16}); inst.has_value()) {
      return inst;
    }
  }
  LOG(FATAL) << "No QY2 MMA instruction shape supports GEMM shape m" << M << "n"
             << N << "k" << K << " with block size " << block_size;
  return std::nullopt;
}

std::pair<int, int> ComputeQY2GemmWarpPartition(
    const GemmWarpPolicy &policy, int M, int N, int block_size, Target target,
    const std::optional<std::array<int, 3>> &mma_inst_shape) {
  ICHECK(TargetIsQY2(target))
      << "ComputeQY2GemmWarpPartition requires a QY2 target";
  const std::array<int, 3> inst_shape =
      mma_inst_shape.value_or(std::array<int, 3>{32, 32, 16});
  auto partitions = GetQY2MmaWarpPartitions(M, N, block_size, inst_shape);
  ICHECK(!partitions.empty())
      << "Cannot partition " << block_size << " QY2 threads for logical GEMM "
      << M << "x" << N << " with instruction m" << inst_shape[0] << "n"
      << inst_shape[1] << "k" << inst_shape[2];
  auto [m_warp, n_warp] =
      SelectQY2MmaWarpPartition(policy, M, N, inst_shape, partitions);
  policy->m_warp = m_warp;
  policy->n_warp = n_warp;
  return {m_warp, n_warp};
}

struct Gemm {
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    if (TargetIsQY2(target)) {
      return kGemmInstMusaMMA;
    }
    ICHECK(TargetIsPH1(target))
        << "Unsupported MUSA target for gemm: " << target->str();
    if (AllowSQMMA(op, block_size, target)) {
      return kGemmInstMusaSQMMA;
    }
    if (AllowWmma(op, block_size, target)) {
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
      return SelectWmmaInstShape(op, block_size, target);
    }
    if (TargetIsQY2(target) && gemm_inst == kGemmInstMusaMMA) {
      return SelectQY2MmaInstShape(op.a_, op.b_, op.c_, op.m_, op.n_, op.k_,
                                   block_size, target);
    }
    return std::nullopt;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    if (TargetIsQY2(target)) {
      LOG(WARNING)
          << "QY2 GEMM warp partition is using the default m32n32k16 "
             "To use another instruction shape, C++ callers should use "
             "musa::ComputeQY2GemmWarpPartition; Python callers should use "
             "tl.MUSAGemmComputeWarpPartition.";
      return ComputeQY2GemmWarpPartition(GetRef<GemmWarpPolicy>(&policy), M, N,
                                         block_size, target);
    }
    int num_warps = block_size / TargetGetWarpSize(target);

    int k_m_per_warp = 4;
    int k_n_per_warp = 8;
    ICHECK(TargetIsPH1(target))
        << "Unsupported MUSA target for gemm: " << target->str();
    if (gemm_inst == kGemmInstMusaFMA) {
      k_m_per_warp = 1;
      k_n_per_warp = 1;
    } else if (gemm_inst == kGemmInstMusaPH1WMMA) {
      k_m_per_warp = 8;
      k_n_per_warp = 8;
    }

    if (gemm_inst == kGemmInstMusaSQMMA) {
      return ComputeWarpGroupPartition(policy, M, N, num_warps, k_m_per_warp,
                                       k_n_per_warp);
    }
    return ComputeDefaultWarpPartition(policy, M, N, num_warps, k_m_per_warp,
                                       k_n_per_warp);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    return gemm_inst == kGemmInstMusaMMA;
  }

  static bool CanReuseExistingSharedLayout(const GemmNode &op,
                                           const LayoutMap &layout_map,
                                           const Buffer &buffer, int block_size,
                                           Target target, String gemm_inst) {
    if (!TargetIsQY2(target) || gemm_inst != kGemmInstMusaMMA) {
      return false;
    }
    const auto f = Function::GetGlobal("tl.musa.qy2.gemm.reuse_shared_layout");
    ICHECK(f.has_value())
        << "QY2 GEMM shared-layout reuse callback is not registered";
    return (*f)(GetRef<tvm::tl::Gemm>(&op), layout_map, buffer, target,
                block_size)
        .cast<bool>();
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
      musa::Gemm::SelectInstShape,
      musa::Gemm::CanReuseExistingSharedLayout,
  });
  return true;
}

const bool musa_gemm_registered = RegisterMUSAGemm();

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef()
      .def("tl.MUSAGemmComputeWarpPartition",
           [](GemmWarpPolicy policy, int M, int N, int block_size,
              Target target, Array<Integer> mma_inst_shape) {
             std::optional<std::array<int, 3>> inst_shape;
             if (!mma_inst_shape.empty()) {
               ICHECK_EQ(mma_inst_shape.size(), 3)
                   << "mma_inst_shape must contain [M, N, K]";
               inst_shape = std::array<int, 3>{
                   static_cast<int>(mma_inst_shape[0]->value),
                   static_cast<int>(mma_inst_shape[1]->value),
                   static_cast<int>(mma_inst_shape[2]->value)};
             }
             musa::ComputeQY2GemmWarpPartition(policy, M, N, block_size, target,
                                               inst_shape);
           })
      .def("tl.GemmSelectQY2MmaInstShape",
           [](Gemm gemm, int block_size, Target target) {
             Array<Integer> result;
             auto inst_shape = gemm->GetGemmInstructionShape(block_size, target,
                                                             kGemmInstMusaMMA);
             if (inst_shape.has_value()) {
               result.push_back(Integer((*inst_shape)[0]));
               result.push_back(Integer((*inst_shape)[1]));
               result.push_back(Integer((*inst_shape)[2]));
             }
             return result;
           });
}

} // namespace tl
} // namespace tvm
