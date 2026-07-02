/*!
 * \file tl/backend/musa/op/atomic_reduce.cc
 * \brief MUSA implementation for tl.atomicmax/tl.atomicmin lowering.
 */

#include "backend/common/op/atomic_reduce.h"

#include "target/utils.h"

namespace tvm {
namespace tl {

namespace {

bool MatchMUSAAtomicReduceTarget(Target target) { return TargetIsMusa(target); }

bool RegisterMUSAAtomicReduce() {
  RegisterAtomicReduceImpl(AtomicReduceImpl{
      "musa.AtomicReduce",
      MatchMUSAAtomicReduceTarget,
      backend::AtomicReduce::InferLayout,
      backend::AtomicReduce::Lower,
  });
  return true;
}

const bool musa_atomic_reduce_registered = RegisterMUSAAtomicReduce();

} // namespace

} // namespace tl
} // namespace tvm
