/*!
 * \file tl/musa/op/copy.cc
 * \brief MUSA implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "backend/common/target_utils.h"
#include "musa/transform/async_copy_injector.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace musa {

namespace {

bool IsExplicitAsyncCopy(const CopyNode &op) {
  if (auto value = op.annotations.Get("is_async_copy")) {
    if (const auto *flag = value.value().as<IntImmNode>()) {
      return flag->value != 0;
    }
  }
  return false;
}

Stmt LowerAsyncCopy(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
  ICHECK(TargetMUSAHasAsyncCopy(lower_args.target))
      << "T.async_copy requires a MUSA target with async-copy support "
         "(mp_21 or newer). Got target="
      << lower_args.target;
  ICHECK(IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst))
      << "T.async_copy only supports global-to-shared copies. Got src scope="
      << op.src.scope() << ", dst scope=" << op.dst.scope();
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "T.async_copy requires matching source and destination dtypes.";

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);
  for (auto level :
       {InferLevel::kCommon, InferLevel::kStrict, InferLevel::kFree}) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }

  auto loop_layout = par_op->GetLoopLayout();
  Stmt lowered_loop = LowerParallelLoop(
      par_op->GetRoot(), loop_layout, lower_args.thread_index, analyzer,
      lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
      /*parallel_loop=*/true, /*should_vectorize=*/true,
      par_op->LoopLayoutRequiresPaddingGuard());

  auto injected = InjectMUSAAsyncCopy(
      lowered_loop, /*async_without_async_commit_wait=*/true);
  ICHECK(injected.injected_ptx_async_copy)
      << "T.async_copy requires an eligible global-to-shared vectorized copy.";

  Stmt commit_group =
      Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
  return SeqStmt({injected.stmt, commit_group});
}

} // namespace

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    return op.InferSIMTLayout(layout_args, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    if (IsExplicitAsyncCopy(op)) {
      return LowerAsyncCopy(op, lower_args, analyzer);
    }
    return LowerNormalCopy(op, lower_args, analyzer);
  }
};

} // namespace musa

namespace {

bool MatchMUSACopyTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSACopy() {
  RegisterCopyImpl(CopyImpl{
      "musa.Copy",
      MatchMUSACopyTarget,
      100,
      musa::Copy::InferLayout,
      musa::Copy::Lower,
  });
  return true;
}

const bool musa_copy_registered = RegisterMUSACopy();

} // namespace

} // namespace tl
} // namespace tvm
