/*!
 * \file tl/metal/op/fill.cc
 * \brief Metal implementation for tl.fill lowering.
 */

#include "op/fill.h"
#include <tvm/runtime/logging.h>

#include "op/utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"

namespace tvm {
namespace tl {

namespace metal {

struct Fill {
  static Stmt Lower(const FillNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    if (IsFragmentBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           false,
                           lower_args.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop = PartitionLoop(par_op->GetRoot(), lower_args.thread_var,
                                       analyzer, par_op->GetLoopLayout());
      auto vectorized_loop =
          VectorizeLoop(thread_loop, analyzer, lower_args.layout_map);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);

      if (par_op->GetPredicate(lower_args.thread_var).defined()) {
        return IfThenElse(par_op->GetPredicate(lower_args.thread_var).value(),
                          unrolled_loop);
      }
      return unrolled_loop;
    }

    if (IsLocalBuffer(op.dst) || IsLocalVarBuffer(op.dst)) {
      auto init_loop = op.MakeSIMTLoop(analyzer);
      auto vectorized_loop =
          VectorizeLoop(init_loop, analyzer, lower_args.layout_map);
      return PragmaUnrollLoop(vectorized_loop);
    }

    if (IsSharedBuffer(op.dst) || IsGlobalBuffer(op.dst)) {
      auto par_op = ParallelOp(op.MakeSIMTLoop(analyzer));
      par_op->InferLayout({lower_args.target,
                           lower_args.thread_bounds,
                           lower_args.layout_map,
                           analyzer,
                           false,
                           lower_args.buffer_remap,
                           {}},
                          InferLevel::kFree);
      auto thread_loop = PartitionLoop(par_op->GetRoot(), lower_args.thread_var,
                                       analyzer, par_op->GetLoopLayout());
      auto vectorized_loop =
          VectorizeLoop(thread_loop, analyzer, lower_args.layout_map);
      auto unrolled_loop = PragmaUnrollLoop(vectorized_loop);
      if (par_op->GetPredicate(lower_args.thread_var).defined()) {
        return IfThenElse(par_op->GetPredicate(lower_args.thread_var).value(),
                          unrolled_loop);
      }
      return unrolled_loop;
    }

    LOG(FATAL) << "Unsupported scope " << op.dst.scope();
    return Stmt();
  }
};

} // namespace metal

namespace {

bool MatchMetalFillTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalFill() {
  RegisterFillImpl(FillImpl{
      "metal.Fill",
      MatchMetalFillTarget,
      metal::Fill::Lower,
  });
  return true;
}

const bool metal_fill_registered = RegisterMetalFill();

} // namespace

} // namespace tl
} // namespace tvm
