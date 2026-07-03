/*!
 * \file tl/backend/musa/op/gemm_sp.cc
 * \brief MUSA implementation for tl.gemm_sp instruction selection.
 */

#include "op/gemm_sp.h"

#include "support/check.h"
#include "target/utils.h"

#include <tvm/runtime/logging.h>

#include <algorithm>
#include <cmath>
#include <limits>
#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace musa {

namespace {

constexpr const char *kMusaMMASP = "musa.mma.sp";

bool CheckMMASP(const GemmSPNode &op) {
  if (op.C->dtype == DataType::Float(16) ||
      op.C->dtype == DataType::Float(32)) {
    if (op.A->dtype == DataType::Float(16) &&
        op.B->dtype == DataType::Float(16)) {
      return op.K % 32 == 0;
    }
    if (op.A->dtype == DataType::BFloat(16) &&
        op.B->dtype == DataType::BFloat(16)) {
      return op.K % 32 == 0;
    }
    return false;
  }

  if (op.C->dtype == DataType::Int(32)) {
    if ((op.A->dtype == DataType::Int(8) || op.A->dtype == DataType::UInt(8)) &&
        (op.B->dtype == DataType::Int(8) || op.B->dtype == DataType::UInt(8))) {
      return op.K % 64 == 0;
    }
  }
  return false;
}

void FatalUnavailable(const GemmSPNode &op, Target target) {
  LOG(FATAL) << "T.gemm_sp on MUSA currently requires QY2 sparse MMA "
                "lowering. Got target="
             << target << ", A(scope=" << op.A.scope()
             << ", dtype=" << op.A->dtype << "), E(scope=" << op.E.scope()
             << ", dtype=" << op.E->dtype << "), B(scope=" << op.B.scope()
             << ", dtype=" << op.B->dtype << "), C(scope=" << op.C.scope()
             << ", dtype=" << op.C->dtype << "), M=" << op.M << ", N=" << op.N
             << ", K=" << op.K << ".";
}

std::pair<int, int>
ComputeDefaultWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
                            int num_warps) {
  int m_warp = 1, n_warp = 1;
  constexpr int kMPerWarp = 16;
  constexpr int kNPerWarp = 8;

  ICHECK(M % kMPerWarp == 0)
      << "M must be divisible by " << kMPerWarp << ", but got " << M;
  ICHECK(N % kNPerWarp == 0)
      << "N must be divisible by " << kNPerWarp << ", but got " << N;

  if (policy.isFullRow()) {
    m_warp = num_warps;
    n_warp = 1;
    if (M % (m_warp * kMPerWarp) != 0) {
      int max_m_warps = M / kMPerWarp;
      m_warp = std::min(num_warps, max_m_warps);
      while (m_warp > 1 && num_warps % m_warp != 0) {
        --m_warp;
      }
      n_warp = num_warps / m_warp;
    }
  } else if (policy.isFullCol()) {
    m_warp = 1;
    n_warp = num_warps;
    if (N % (n_warp * kNPerWarp) != 0) {
      int max_n_warps = N / kNPerWarp;
      n_warp = std::min(num_warps, max_n_warps);
      while (n_warp > 1 && num_warps % n_warp != 0) {
        --n_warp;
      }
      m_warp = num_warps / n_warp;
    }
  } else if (policy.isSquare()) {
    int max_m = M / kMPerWarp;
    int max_n = N / kNPerWarp;
    float ideal = N > 0 ? static_cast<float>(M) / N : 1.0f;
    float best_score = std::numeric_limits<float>::max();
    int best_m = 1, best_n = num_warps;

    for (int m = 1; m <= num_warps && m <= max_m; ++m) {
      if (num_warps % m != 0) {
        continue;
      }
      int n = num_warps / m;
      if (n > max_n) {
        continue;
      }
      float m_per_warp = static_cast<float>(M) / (m * kMPerWarp);
      float n_per_warp = static_cast<float>(N) / (n * kNPerWarp);
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
    ICHECK(0) << "Unknown GemmSPWarpPolicy";
  }

  ICHECK(m_warp * n_warp == num_warps)
      << "m_warp * n_warp must equal num_warps, m_warp: " << m_warp
      << ", n_warp: " << n_warp << ", num_warps: " << num_warps;
  policy.m_warp = m_warp;
  policy.n_warp = n_warp;
  return {m_warp, n_warp};
}

} // namespace

struct GemmSP {
  static String SelectInst(const GemmSPNode &op, int block_size,
                           Target target) {
    if (op.isWgmma_ || op.isTcgen05_ || !TargetIsQY2(target) ||
        !CheckMMASP(op)) {
      FatalUnavailable(op, target);
    }
    return kMusaMMASP;
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmSPWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    ICHECK(TargetIsQY2(target))
        << "MUSA sparse MMA is only enabled for QY2 targets.";
    ICHECK(gemm_inst == kMusaMMASP)
        << "Unknown MUSA sparse GEMM instruction: " << gemm_inst;
    constexpr int kSparseMMAWarpSize = 32;
    ICHECK_EQ(block_size % kSparseMMAWarpSize, 0)
        << "block_size must be divisible by warp size.";
    return ComputeDefaultWarpPartition(policy, M, N,
                                       block_size / kSparseMMAWarpSize);
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    ICHECK(gemm_inst == kMusaMMASP)
        << "Unknown MUSA sparse GEMM instruction: " << gemm_inst;
    return true;
  }

  static String InstructionKind(String gemm_inst) {
    if (gemm_inst == kMusaMMASP) {
      return "mma.sp";
    }
    return "unknown";
  }
};

} // namespace musa

namespace {

bool MatchMUSAGemmSPTarget(Target target) { return TargetIsMusa(target); }

bool RegisterMUSAGemmSP() {
  RegisterGemmSPImpl(GemmSPImpl{
      "musa.GemmSP",
      MatchMUSAGemmSPTarget,
      musa::GemmSP::SelectInst,
      musa::GemmSP::ComputeWarpPartition,
      musa::GemmSP::ReuseExistingSharedLayout,
      musa::GemmSP::InstructionKind,
  });
  return true;
}

const bool musa_gemm_sp_registered = RegisterMUSAGemmSP();

} // namespace

} // namespace tl
} // namespace tvm
