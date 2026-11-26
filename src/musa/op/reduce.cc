/*!
 * \file tl/musa/op/reduce.cc
 * \brief MUSA implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "backend/common/target_utils.h"
#include "musa/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;

namespace musa {

void CheckSyncThreadsScope(int reducing_threads, PrimExpr all_threads,
                           Target target) {
  const int warp_size = TargetMUSAGetWarpSize(target);
  if (reducing_threads <= warp_size) {
    return;
  }
  const int64_t *all_threads_value = as_const_int(all_threads);
  ICHECK(all_threads_value != nullptr)
      << "MUSA tl.reduce with SyncThreadsBarrier requires a compile-time "
         "constant thread extent.";
  ICHECK_GT(*all_threads_value, 0)
      << "MUSA tl.reduce requires positive thread extent.";
  ICHECK_EQ(*all_threads_value % reducing_threads, 0)
      << "MUSA tl.reduce with SyncThreadsBarrier requires all_threads ("
      << *all_threads_value << ") to be divisible by reducing_threads ("
      << reducing_threads << ").";
}

bool CanUseWarpFirstAllReduce(int reducing_threads, int scale,
                              PrimExpr thread_offset, Target target) {
  const int warp_size = TargetMUSAGetWarpSize(target);
  const int64_t *offset = as_const_int(thread_offset);
  if (offset == nullptr || *offset != 0) {
    return false;
  }
  if (reducing_threads <= warp_size || reducing_threads % warp_size != 0) {
    return false;
  }
  if (scale <= 0 || scale > warp_size || (scale & (scale - 1)) != 0) {
    return false;
  }
  const int num_warps = reducing_threads / warp_size;
  return num_warps <= 32 && (num_warps & (num_warps - 1)) == 0;
}

struct Reduce : backend::ReduceLowerer<Reduce> {
  static bool AllReduceNeedsWorkspace(int reducing_threads, int, Target target) {
    return reducing_threads > TargetMUSAGetWarpSize(target);
  }

  static bool SupportsFp16Bf16NanReduce(Target) { return false; }

  static int GetPreferredVectorizedSize(const ReduceOpNode &, Target) {
    return 1;
  }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    CheckSyncThreadsScope(reducing_threads, all_threads, target);
    std::stringstream ss;
    if (CanUseWarpFirstAllReduce(reducing_threads, scale, thread_offset,
                                 target)) {
      ss << "tl::WarpFirstAllReduce<" << reducer << ", " << reducing_threads
         << ", " << scale << ", " << thread_offset
         << ", tl::SyncThreadsBarrier, " << batch << ", " << workspace_stride
         << ">::run_batch";
    } else {
      ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
         << scale << ", " << thread_offset << ", tl::SyncThreadsBarrier, "
         << batch << ", " << workspace_stride << ">::run_batch";
    }
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    CheckSyncThreadsScope(reducing_threads, all_threads, target);
    std::stringstream ss;
    if (CanUseWarpFirstAllReduce(reducing_threads, scale, thread_offset,
                                 target)) {
      ss << "tl::WarpFirstAllReduce<" << reducer << ", " << reducing_threads
         << ", " << scale << ", " << thread_offset << ">::run";
    } else {
      ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
         << scale << ", " << thread_offset << ">::run";
    }
    return ss.str();
  }
};

} // namespace musa

namespace {

bool MatchMUSAReduceTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSAReduce() {
  RegisterReduceImpl(ReduceImpl{
      "musa.Reduce",
      MatchMUSAReduceTarget,
      musa::Reduce::Lower,
  });
  return true;
}

const bool musa_reduce_registered = RegisterMUSAReduce();

} // namespace

} // namespace tl
} // namespace tvm
