/*!
 * \file tl/backend/musa/op/finalize_reducer.cc
 * \brief MUSA implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "backend/common/op/finalize_reducer.h"

#include "target/utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace musa {

struct FinalizeReducer : backend::FinalizeReducerLowerer<FinalizeReducer> {
  static int WarpSize(Target) { return 32; }

  static bool UseSyncBarrier(Target target, int reducing_threads) {
    return TargetIsPH1(target) && reducing_threads >= 64;
  }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetHasSMVersionGE(target, 90)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
    } else {
      ss << ", tl::SyncThreadsBarrier";
    }
    ss << ", " << batch << ", " << workspace_stride << ">::run_batch";
    return ss.str();
  }

  static std::string MakeScalarAllReduce(std::string reducer,
                                         int reducing_threads, int scale,
                                         PrimExpr thread_offset,
                                         PrimExpr all_threads, Target target) {
    std::stringstream ss;
    ss << "tl::AllReduce<" << reducer << ", " << reducing_threads << ", "
       << scale << ", " << thread_offset;
    if (TargetHasSMVersionGE(target, 90)) {
      ss << ", tl::NamedBarrier<" << all_threads << ">";
    }
    ss << ">::run";
    return ss.str();
  }
};

} // namespace musa

namespace {

bool MatchMUSAFinalizeReducerTarget(Target target) {
  return TargetIsMusa(target);
}

bool RegisterMUSAFinalizeReducer() {
  RegisterFinalizeReducerImpl(FinalizeReducerImpl{
      "musa.FinalizeReducer",
      MatchMUSAFinalizeReducerTarget,
      musa::FinalizeReducer::Lower,
  });
  return true;
}

const bool musa_finalize_reducer_registered = RegisterMUSAFinalizeReducer();

} // namespace

} // namespace tl
} // namespace tvm
