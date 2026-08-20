/*!
 * \file tl/musa/op/copy.cc
 * \brief MUSA implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "backend/common/target_utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace musa {

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
