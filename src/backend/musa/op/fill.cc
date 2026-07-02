/*!
 * \file tl/backend/musa/op/fill.cc
 * \brief MUSA implementation for tl.fill lowering.
 */

#include "backend/common/op/fill.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMUSAFillTarget(Target target) { return TargetIsMusa(target); }

bool RegisterMUSAFill() {
  RegisterFillImpl(FillImpl{
      "musa.Fill",
      MatchMUSAFillTarget,
      backend::Fill::Lower,
  });
  return true;
}

const bool musa_fill_registered = RegisterMUSAFill();

} // namespace

} // namespace tl
} // namespace tvm
