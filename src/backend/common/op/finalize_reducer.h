/*!
 * \file tl/backend/common/op/finalize_reducer.h
 * \brief Shared tl.finalize_reducer lowering for GPU backends.
 */

#ifndef TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
#define TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_

#include "op/finalize_reducer.h"
#include "support/check.h"

#include <tvm/tirx/builtin.h>

#include <array>
#include <cstdint>
#include <string>

namespace tvm {
namespace tl {
namespace backend {

using namespace tirx;
using namespace ffi;

template <typename Impl> struct FinalizeReducerLowerer {
  static Stmt Lower(const FinalizeReducerOpNode &op, const LowerArgs &T,
                    arith::Analyzer *) {
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
      ICHECK_LE(effective_batch, layout_batch_size)
          << "finalize_reducer: batch (" << effective_batch
          << ") exceeds total output elements (" << layout_batch_size << ")";
      ICHECK_EQ(layout_batch_size % effective_batch, 0)
          << "finalize_reducer: batch (" << effective_batch
          << ") must evenly divide total output elements (" << layout_batch_size
          << ")";
    }

    bool use_batch =
        effective_batch > 1 && reducing_threads > Impl::WarpSize(T.target);
    const bool use_sync_barrier =
        Impl::UseSyncBarrier(T.target, reducing_threads);
    Buffer sync_barrier;
    int sync_barrier_id = 0;
    if (use_sync_barrier) {
      auto all_threads = T.thread_bounds->extent;
      ICHECK(T.AllocMBarrier)
          << "finalize_reducer requires an mbarrier allocator for named "
             "barrier synchronization";
      ICHECK(T.mbarrier_buffer != nullptr);
      sync_barrier_id = T.AllocMBarrier(*as_const_int(all_threads));
      sync_barrier = T.mbarrier_buffer->value();
      ICHECK(sync_barrier.defined());
    }

    if (use_batch) {
      int workspace_stride =
          static_cast<int>(*as_const_int(T.thread_bounds->extent));
      std::string allreduce = Impl::MakeBatchAllReduce(
          op_str, reducing_threads, 1, thread_offset, T.thread_bounds->extent,
          static_cast<int>(effective_batch), workspace_stride, T.target);
      int ws_size = workspace_stride * static_cast<int>(effective_batch);
      PrimExpr workspace = T.AddWorkspace(ws_size, buffer->dtype);
      Array<PrimExpr> args = {StringImm(allreduce), buffer->data};
      if (use_sync_barrier) {
        PrimExpr barrier_id = BufferLoad(
            sync_barrier, {IntImm(DataType::Int(32), sync_barrier_id)});
        args.push_back(barrier_id);
      }
      args.push_back(workspace);
      return Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
    }

    std::string allreduce =
        Impl::MakeScalarAllReduce(op_str, reducing_threads, 1, thread_offset,
                                  T.thread_bounds->extent, T.target);
    Array<PrimExpr> thread_reduce_args = {StringImm(allreduce),
                                          BufferLoad(buffer, indices_0)};
    if (use_sync_barrier) {
      PrimExpr barrier_id = BufferLoad(
          sync_barrier, {IntImm(DataType::Int(32), sync_barrier_id)});
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
};

} // namespace backend
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_COMMON_OP_FINALIZE_REDUCER_H_
