/*! \file tl/musa/op/distributed.h
 *  \brief MUSA-specific peer-memory intrinsic Ops.
 */

#ifndef TVM_TL_MUSA_OP_DISTRIBUTED_H_
#define TVM_TL_MUSA_OP_DISTRIBUTED_H_

#include <tvm/ir/op.h>

namespace tvm {
namespace tl {
namespace musa {

TVM_DLL const Op &ldg128_peer();
TVM_DLL const Op &stg128_peer();
TVM_DLL const Op &peer_warp_reduce128();
TVM_DLL const Op &peer_release_fence();
TVM_DLL const Op &peer_signal_store();

} // namespace musa
} // namespace tl
} // namespace tvm

#endif // TVM_TL_MUSA_OP_DISTRIBUTED_H_
