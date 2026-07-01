/*!
 * \file tl/backend/musa/op/fill.cc
 * \brief MUSA implementation for tl.fill lowering.
 */

#include "op/fill.h"

#include "op/utils.h"
#include "target/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

namespace musa {

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer) {
    if (IsFragmentBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var,
                                       analyzer, par_op->GetLoopLayout());
      auto vectorized_loop = VectorizeLoop(thread_loop, analyzer, T.layout_map);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);

      if (par_op->GetPredicate(T.thread_var).defined()) {
        return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                          unrolled_loop);
      }
      return unrolled_loop;
    }

    if (IsLocalBuffer(op.dst) || IsLocalVarBuffer(op.dst)) {
      auto init_loop = op.MakeSIMTLoop(analyzer);
      auto vectorized_loop = VectorizeLoop(init_loop, analyzer, T.layout_map);
      return PragmaUnrollLoop(vectorized_loop);
    }

    if (IsSharedBuffer(op.dst) || IsGlobalBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({T.target,
                           T.thread_bounds,
                           T.layout_map,
                           analyzer,
                           false,
                           T.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop = PartitionLoop(par_op->GetRoot(), T.thread_var,
                                       analyzer, par_op->GetLoopLayout());
      auto vectorized_loop = VectorizeLoop(thread_loop, analyzer, T.layout_map);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);
      if (par_op->GetPredicate(T.thread_var).defined()) {
        return IfThenElse(par_op->GetPredicate(T.thread_var).value(),
                          unrolled_loop);
      }
      return unrolled_loop;
    }

    LOG(FATAL) << "Unsupported scope " << op.dst.scope();
    return Stmt();
  }
};

} // namespace musa

namespace {

bool MatchMUSAFillTarget(Target target) { return TargetIsMusa(target); }

bool RegisterMUSAFill() {
  RegisterFillImpl(FillImpl{
      "musa.Fill",
      MatchMUSAFillTarget,
      musa::Fill::Lower,
  });
  return true;
}

const bool musa_fill_registered = RegisterMUSAFill();

} // namespace

} // namespace tl
} // namespace tvm
