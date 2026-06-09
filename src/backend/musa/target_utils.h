/*!
 * \file tl/backend/musa/target_utils.h
 * \brief MUSA target attribute helpers.
 */

#ifndef TVM_TL_BACKEND_MUSA_TARGET_UTILS_H_
#define TVM_TL_BACKEND_MUSA_TARGET_UTILS_H_

#include <tvm/target/target.h>

namespace tvm {
namespace tl {

bool TargetIsMusa(Target target);
bool TargetIsQY2(Target target);
bool TargetIsPH1(Target target);
bool TargetMusaHasAsyncCopy(Target target);
bool TargetMusaHasLdmatrix(Target target);
bool TargetMusaHasBulkCopy(Target target);
int TargetMusaGetWarpSize(Target target);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_MUSA_TARGET_UTILS_H_
