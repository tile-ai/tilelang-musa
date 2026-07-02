/*!
 * \file tl/backend/musa/op/cumsum.cc
 * \brief MUSA implementation for tl.cumsum lowering.
 */

#include "backend/common/op/cumsum.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMUSACumSumTarget(Target target) { return TargetIsMusa(target); }

bool RegisterMUSACumSum() {
  RegisterCumSumImpl(CumSumImpl{
      "musa.CumSum",
      MatchMUSACumSumTarget,
      backend::CumSum::Lower,
  });
  return true;
}

const bool musa_cumsum_registered = RegisterMUSACumSum();

} // namespace

} // namespace tl
} // namespace tvm
