/*!
 * \file tl/metal/op/copy.cc
 * \brief Metal implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace metal {

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    return op.InferSIMTLayout(layout_args, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    return LowerNormalCopy(op, lower_args, analyzer);
  }
};

} // namespace metal

namespace {

bool MatchMetalCopyTarget(Target target) { return TargetIsMetal(target); }

bool RegisterMetalCopy() {
  RegisterCopyImpl(CopyImpl{
      "metal.Copy",
      MatchMetalCopyTarget,
      100,
      metal::Copy::InferLayout,
      metal::Copy::Lower,
  });
  return true;
}

const bool metal_copy_registered = RegisterMetalCopy();

} // namespace

} // namespace tl
} // namespace tvm
