/*!
 * \file tl/backend/musa/op/reduce.cc
 * \brief MUSA implementation for tl.reduce AllReduce lowering.
 */

#include "backend/common/op/reduce.h"

#include "support/check.h"
#include "backend/common/target_utils.h"

#include <sstream>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace musa {

struct Reduce : backend::ReduceLowerer<Reduce> {
  static Stmt Lower(const ReduceOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    if (!TargetIsPH1(lower_args.target)) {
      return backend::ReduceLowerer<Reduce>::Lower(op, lower_args, analyzer);
    }

    const int64_t *extent = as_const_int(lower_args.thread_bounds->extent);
    ICHECK(extent) << "PH1 AllReduce requires a constant participating extent.";
    const int participants = static_cast<int>(*extent);
    ICHECK_GT(participants, 0);
    LowerArgs args = lower_args;
    int reducing_threads = 0;
    if (lower_args.alloc_mbarrier) {
      args.alloc_mbarrier = [&](int group_threads,
                                std::optional<std::string> name) {
        reducing_threads = group_threads;
        // The barrier is shared by all concurrent reduction groups. Use
        // the current predicated thread range, not a single group's width
        // or the entire CTA (which may contain non-participating warps).
        return lower_args.alloc_mbarrier(participants, std::move(name));
      };
    }
    if (op.batch > 1 && lower_args.add_workspace) {
      args.add_workspace = [&](int size, DataType dtype) {
        // Each batch slot needs storage for every participating thread, just
        // like scalar AllReduce. The common lowerer requests a group-sized
        // slot; expand it to match the MUSA batch workspace stride below.
        ICHECK_GT(reducing_threads, 0);
        ICHECK_EQ(size % reducing_threads, 0);
        return lower_args.add_workspace(size / reducing_threads * participants,
                                        dtype);
      };
    }
    return backend::ReduceLowerer<Reduce>::Lower(op, args, analyzer);
  }

  static int GetPreferedVectorizedSize(DataType dtype, Target) {
    return backend::reduce::GetPreferedVectorizedSize(dtype);
  }

  static bool SupportsFp16Bf16NanReduce(Target target) {
    return TargetIsMusa(target);
  }

  static bool UseSyncBarrier(Target target, int reducing_threads) {
    return TargetIsPH1(target) && reducing_threads >= 64;
  }

  static int GetQY2ThreadYExtent(Target target, PrimExpr thread_y_extent,
                                 PrimExpr all_threads) {
    const int64_t *extent = as_const_int(thread_y_extent);
    if (!TargetIsQY2(target) || extent == nullptr || *extent <= 1) {
      return 1;
    }
    const int64_t *thread_extent = as_const_int(all_threads);
    ICHECK(thread_extent)
        << "QY2 threadIdx.y reduce requires constant threadIdx.x extent.";
    ICHECK_LE(*thread_extent, 128)
        << "QY2 threadIdx.y reduce currently supports at most one "
        << "128-thread hardware warp per threadIdx.y lane.";
    return static_cast<int>(*extent);
  }

  static int GetThreadYWorkspaceMultiplier(Target target,
                                           PrimExpr thread_y_extent,
                                           PrimExpr all_threads) {
    return GetQY2ThreadYExtent(target, thread_y_extent, all_threads);
  }

  static std::string MakeBatchAllReduce(std::string reducer,
                                        int reducing_threads, int scale,
                                        PrimExpr thread_offset,
                                        PrimExpr all_threads, int batch,
                                        int workspace_stride, Target target) {
    if (UseSyncBarrier(target, reducing_threads)) {
      const int64_t *extent = as_const_int(all_threads);
      ICHECK(extent);
      workspace_stride = static_cast<int>(*extent);
    }
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

  static std::string MakeBatchAllReduceWithThreadY(
      std::string reducer, int reducing_threads, int scale,
      PrimExpr thread_offset, PrimExpr all_threads, int batch,
      int workspace_stride, Target target, PrimExpr thread_y_extent) {
    if (GetQY2ThreadYExtent(target, thread_y_extent, all_threads) <= 1) {
      return MakeBatchAllReduce(std::move(reducer), reducing_threads, scale,
                                thread_offset, all_threads, batch,
                                workspace_stride, target);
    }
    std::stringstream ss;
    ss << "tl::AllReduceSubgroup<" << reducer << ", " << reducing_threads
       << ", " << scale << ", " << thread_offset << ", " << all_threads << ", "
       << batch << ", " << workspace_stride << ">::run_batch";
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

  static std::string
  MakeScalarAllReduceWithThreadY(std::string reducer, int reducing_threads,
                                 int scale, PrimExpr thread_offset,
                                 PrimExpr all_threads, Target target,
                                 PrimExpr thread_y_extent) {
    if (GetQY2ThreadYExtent(target, thread_y_extent, all_threads) <= 1) {
      return MakeScalarAllReduce(std::move(reducer), reducing_threads, scale,
                                 thread_offset, all_threads, target);
    }
    std::stringstream ss;
    ss << "tl::AllReduceSubgroup<" << reducer << ", " << reducing_threads
       << ", " << scale << ", " << thread_offset << ", " << all_threads
       << ">::run";
    return ss.str();
  }
};

} // namespace musa

namespace {

bool MatchMUSAReduceTarget(Target target) { return TargetIsMusa(target); }

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
