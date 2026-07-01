/*!
 * \file tl/backend/musa/op/finalize_reducer.cc
 * \brief MUSA implementation for tl.finalize_reducer AllReduce lowering.
 */

#include "op/finalize_reducer.h"

#include "target/utils.h"

#include <tvm/tir/builtin.h>

#include <array>
#include <sstream>

namespace tvm {
namespace tl {

using namespace tir;

namespace musa {

struct FinalizeReducer {
  static int WarpSize(Target) { return 32; }

  static Stmt Lower(const FinalizeReducerOpNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    auto buffer = T.buffer_remap[op.reducer];
    auto opt_layout = T.layout_map.Get(op.reducer);
    ICHECK(opt_layout);
    ICHECK(opt_layout->as<Fragment>());
    auto layout = opt_layout->as<Fragment>().value();
    Array<PrimExpr> indices_0;
    indices_0.reserve(layout->OutputDim());
    for (int i = 0; i < layout->OutputDim(); ++i) {
      indices_0.push_back(Var("__finred_" + std::to_string(i)));
    }

    const int64_t *p_extent = as_const_int(layout->ReplicateExtent());
    ICHECK(p_extent);
    int extent = *p_extent;
    ICHECK(extent == 1 || extent == *as_const_int(T.thread_bounds->extent))
        << "Illegal finalize_reducer: extent=" << extent
        << "; T.thread_bounds=" << T.thread_bounds;

    if (extent == 1) {
      return Evaluate(0);
    }

    std::array op_names{"tl::SumOp", "tl::MaxOp", "tl::MinOp"};
    auto op_str = op_names[static_cast<int>(op.op)];

    int reducing_threads = extent;
    auto thread_offset = T.thread_bounds->min;

    int64_t layout_batch_size = 1;
    for (int i = 0; i < layout->OutputDim(); ++i) {
      const int64_t *p = as_const_int(layout->OutputShape()[i]);
      if (p == nullptr) {
        layout_batch_size = -1;
        break;
      }
      layout_batch_size *= *p;
    }

    int64_t effective_batch = static_cast<int64_t>(op.batch);

    if (effective_batch > 1 && layout_batch_size > 0) {
      CHECK_LE(effective_batch, layout_batch_size)
          << "finalize_reducer: batch (" << effective_batch
          << ") exceeds total output elements (" << layout_batch_size << ")";
      CHECK_EQ(layout_batch_size % effective_batch, 0)
          << "finalize_reducer: batch (" << effective_batch
          << ") must evenly divide total output elements (" << layout_batch_size
          << ")";
    }

    bool use_batch =
        effective_batch > 1 && reducing_threads > WarpSize(T.target);
    bool use_musa_barrier = TargetIsPH1(T.target) && reducing_threads >= 64;
    Buffer reduce_sync_barrier;

    if (use_batch) {
      int workspace_stride =
          static_cast<int>(*as_const_int(T.thread_bounds->extent));
      std::string allreduce = MakeBatchAllReduce(
          op_str, reducing_threads, 1, thread_offset, T.thread_bounds->extent,
          static_cast<int>(effective_batch), workspace_stride, T.target);
      int ws_size = workspace_stride * static_cast<int>(effective_batch);
      PrimExpr workspace = T.AddWorkspace(ws_size, buffer->dtype);
      Array<PrimExpr> args = {StringImm(allreduce), buffer->data};
      if (use_musa_barrier) {
        auto all_threads = T.thread_bounds->extent;
        reduce_sync_barrier = T.AddBarrier(*as_const_int(all_threads));
        PrimExpr barrier_id =
            BufferLoad(reduce_sync_barrier, {IntImm(DataType::Int(32), 0)});
        args.push_back(barrier_id);
      }
      args.push_back(workspace);
      return Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
    }

    std::string allreduce =
        MakeScalarAllReduce(op_str, reducing_threads, 1, thread_offset,
                            T.thread_bounds->extent, T.target);
    Array<PrimExpr> thread_reduce_args = {StringImm(allreduce),
                                          BufferLoad(buffer, indices_0)};
    if (use_musa_barrier) {
      auto all_threads = T.thread_bounds->extent;
      reduce_sync_barrier = T.AddBarrier(*as_const_int(all_threads));
      PrimExpr barrier_id =
          BufferLoad(reduce_sync_barrier, {IntImm(DataType::Int(32), 0)});
      thread_reduce_args.push_back(barrier_id);
    }
    if (reducing_threads >= 32) {
      PrimExpr workspace =
          T.AddWorkspace(*as_const_int(T.thread_bounds->extent), buffer->dtype);
      thread_reduce_args.push_back(workspace);
    }
    auto call = Call(buffer->dtype, builtin::call_extern(), thread_reduce_args);
    Stmt body = BufferStore(buffer, call, indices_0);

    for (int i = layout->OutputDim() - 1; i >= 0; i--) {
      body = For(indices_0[i].as<Var>().value(), 0, layout->OutputShape()[i],
                 ForKind::kParallel, body);
    }

    return body;
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
