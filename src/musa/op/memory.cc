/*! \file tl/musa/op/memory.cc
 *  \brief Registration of MUSA-specific memory intrinsic Ops.
 */

#include "musa/op/memory.h"

#include <tvm/tirx/op.h>

#include "op/builtin_registry.h"

namespace tvm {
namespace tl {
namespace musa {

using namespace tirx;

const Op &lsu_ld_cache_hint() {
  static const Op &op = Op::Get("tl.musa.lsu_ld_cache_hint");
  return op;
}
TVM_REGISTER_OP("tl.musa.lsu_ld_cache_hint")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "lsu_ld_cache_hint")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kReadState));

const Op &lsu_ld_volatile_cache_hint() {
  static const Op &op = Op::Get("tl.musa.lsu_ld_volatile_cache_hint");
  return op;
}
TVM_REGISTER_OP("tl.musa.lsu_ld_volatile_cache_hint")
    .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                  "lsu_ld_volatile_cache_hint")
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

} // namespace musa
} // namespace tl
} // namespace tvm
