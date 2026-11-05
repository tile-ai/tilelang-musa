/*!
 * \file tl/backend/musa/target_utils.cc
 * \brief MUSA target attribute helpers.
 */

#include "backend/musa/target_utils.h"

#include <cstdlib>
#include <string>

#include <tvm/ffi/reflection/registry.h>

#include "dlpack/dlpack.h"
#include "support/check.h"

namespace tvm {
namespace tl {
namespace {

int GetMusaArchInt(Target target) {
  auto arch = target->GetAttr<ffi::String>("arch");
  ICHECK(arch.has_value());
  const std::string arch_str = arch.value();
  ICHECK(arch_str.size() >= 3);
  ICHECK_EQ(arch_str.compare(0, 3, "mp_"), 0)
      << "MUSA arch string must start with mp_";
  return std::stoi(arch_str.substr(3));
}

} // namespace

bool TargetIsMusa(Target target) {
  return target->GetTargetDeviceType() == kDLExtDev;
}

bool TargetIsQY2(Target target) {
  return TargetIsMusa(target) && GetMusaArchInt(target) == 22;
}

bool TargetIsPH1(Target target) {
  return TargetIsMusa(target) && GetMusaArchInt(target) == 31;
}

bool TargetMusaHasAsyncCopy(Target target) {
  return TargetIsMusa(target) && GetMusaArchInt(target) >= 21;
}

bool TargetMusaHasLdmatrix(Target target) {
  return TargetIsMusa(target) && GetMusaArchInt(target) == 22;
}

bool TargetMusaHasBulkCopy(Target target) {
  return TargetIsMusa(target) && GetMusaArchInt(target) >= 31;
}

int TargetMusaGetWarpSize(Target target) {
  if (TargetIsQY2(target)) {
    const char *mma_shape = std::getenv("TILELANG_MUSA_MP22_MMA_SHAPE");
    const bool use_wave32_shape =
        mma_shape != nullptr && (std::string(mma_shape) == "m16n16k16" ||
                                 std::string(mma_shape) == "m8n32k16");
    return use_wave32_shape ? 32 : 128;
  }
  return 32;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.TargetIsMusa",
           [](Target target) { return TargetIsMusa(target); })
      .def("tl.TargetIsQY2", [](Target target) { return TargetIsQY2(target); })
      .def("tl.TargetIsPH1", [](Target target) { return TargetIsPH1(target); });
}

} // namespace tl
} // namespace tvm
