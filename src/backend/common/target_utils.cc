/*!
 * \file tl/backend/common/target_utils.cc
 * \brief Common target helper dispatch.
 */

#include "backend/common/target_utils.h"

#include <tvm/ffi/reflection/registry.h>

namespace tvm {
namespace tl {

bool TargetHasAsyncCopy(Target target) {
  if (TargetIsCuda(target)) {
    return TargetCudaHasAsyncCopy(target);
  }
  if (TargetIsMusa(target)) {
    return TargetMusaHasAsyncCopy(target);
  }
  if (TargetIsRocm(target)) {
    return TargetRocmHasAsyncCopy(target);
  }
  return false;
}

int TargetGetWarpSize(Target target) {
  if (TargetIsCuda(target)) {
    return TargetCudaGetWarpSize(target);
  }
  if (TargetIsMusa(target)) {
    return TargetMusaGetWarpSize(target);
  }
  if (TargetIsRocm(target)) {
    return TargetRocmGetWarpSize(target);
  }
  if (TargetIsMetal(target)) {
    return TargetMetalGetWarpSize(target);
  }
  return 32;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetHasAsyncCopy",
           [](Target target) { return TargetHasAsyncCopy(target); })
      .def("tl.TargetGetWarpSize",
           [](Target target) { return TargetGetWarpSize(target); });
}

} // namespace tl
} // namespace tvm
