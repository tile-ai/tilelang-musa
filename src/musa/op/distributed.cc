/*! \file tl/musa/op/distributed.cc
 *  \brief Registration of MUSA peer-memory intrinsic Ops.
 */

#include "musa/op/distributed.h"

#include <tvm/tirx/op.h>

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {
namespace musa {

using namespace tirx;

const Op &ldg128_peer() {
  static const Op &op = Op::Get("tl.musa.ldg128_peer");
  return op;
}
TVM_REGISTER_OP("tl.musa.ldg128_peer")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "ldg128_peer")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure));

const Op &stg128_peer() {
  static const Op &op = Op::Get("tl.musa.stg128_peer");
  return op;
}
TVM_REGISTER_OP("tl.musa.stg128_peer")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "stg128_peer")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

const Op &peer_warp_reduce128() {
  static const Op &op = Op::Get("tl.musa.peer_warp_reduce128");
  return op;
}
TVM_REGISTER_OP("tl.musa.peer_warp_reduce128")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "peer_warp_reduce128")
    .set_num_inputs(3)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

const Op &peer_release_fence() {
  static const Op &op = Op::Get("tl.musa.peer_release_fence");
  return op;
}
TVM_REGISTER_OP("tl.musa.peer_release_fence")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "peer_release_fence")
    .set_num_inputs(0)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

const Op &peer_signal_store() {
  static const Op &op = Op::Get("tl.musa.peer_signal_store");
  return op;
}
TVM_REGISTER_OP("tl.musa.peer_signal_store")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "peer_signal_store")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace musa
} // namespace tl
} // namespace tvm
