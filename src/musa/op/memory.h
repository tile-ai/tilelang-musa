/*! \file tl/musa/op/memory.h
 *  \brief MUSA-specific memory intrinsic Ops.
 */

#ifndef TVM_TL_MUSA_OP_MEMORY_H_
#define TVM_TL_MUSA_OP_MEMORY_H_

#include <tvm/ir/op.h>

namespace tvm {
namespace tl {
namespace musa {

TVM_DLL const Op &lsu_ld_cache_hint();
TVM_DLL const Op &lsu_ld_volatile_cache_hint();

} // namespace musa
} // namespace tl
} // namespace tvm

#endif // TVM_TL_MUSA_OP_MEMORY_H_
