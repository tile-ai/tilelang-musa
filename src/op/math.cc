/*!
 * \file tl/op/math.cc
 * \brief Math operations.
 *
 */

#include "support/check.h"
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {
using namespace tirx;

PrimExpr pow_of_int_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  const ffi::Array<PrimExpr> &arg = call->args;
  ICHECK_EQ(arg.size(), 2);
  PrimExpr base = arg[0];
  PrimExpr exp = arg[1];
  ffi::String pow_of_int_name =
      "tl::pow_of_int<" + std::to_string(exp.as<IntImmNode>()->value) + ">";
  return tirx::Call(base.dtype(), tirx::builtin::call_extern(),
                    {StringImm(pow_of_int_name), base});
}

TVM_REGISTER_OP("tl.pow_of_int")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "pow_of_int")
    .set_attr<FLowerIntrinsic>("musa.FLowerIntrinsic", pow_of_int_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", pow_of_int_op)
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", pow_of_int_op);

PrimExpr infinity_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  const DataType &dtype = call->dtype;
  ICHECK_EQ(dtype.lanes(), 1);

  // NOTE(wt): Codegen for PrintConst:Inf will handle this based on dtype
  if (dtype.is_float()) {
    if (dtype.bits() == 64 || dtype.bits() == 32 || dtype.bits() == 16) {
      return FloatImm(dtype, std::numeric_limits<float>::infinity(),
                      call->span);
    }
  } else if (dtype.is_bfloat16()) {
    return FloatImm(dtype, std::numeric_limits<float>::infinity(), call->span);
  } else if (dtype.is_tfloat32()) {
    return FloatImm(dtype, std::numeric_limits<float>::infinity(), call->span);
  }
  LOG(FATAL) << "Cannot decide infinity for type " << dtype;
  throw; // Unreachable, keeps compiler happy
}

TVM_REGISTER_OP("tl.infinity")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "infinity")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic", infinity_op)
    .set_attr<FLowerIntrinsic>("musa.FLowerIntrinsic", infinity_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic", infinity_op);

PrimExpr round_ties_away_from_zero_op(PrimExpr args) {
  const CallNode *call = args.as<CallNode>();
  ICHECK(call != nullptr);
  ICHECK_EQ(call->args.size(), 1);
  const DataType &dtype = call->dtype;
  if (dtype.is_int() || dtype.is_uint() || dtype.is_bool()) {
    return call->args[0];
  }
  ffi::String func_name =
      dtype.is_float() && dtype.bits() == 64 ? "round" : "roundf";
  return tirx::Call(dtype, tirx::builtin::call_pure_extern(),
                    {StringImm(func_name), call->args[0]}, call->annotations);
}

TVM_REGISTER_OP("tl.round_ties_away_from_zero")
    .set_num_inputs(1)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TVectorizable>("TVectorizable", true)
    .set_attr<TScriptPrinterName>("TScriptPrinterName",
                                  "round_ties_away_from_zero")
    .set_attr<FLowerIntrinsic>("cuda.FLowerIntrinsic",
                               round_ties_away_from_zero_op)
    .set_attr<FLowerIntrinsic>("hip.FLowerIntrinsic",
                               round_ties_away_from_zero_op);

const Op &fast_div() {
  static const Op &op = Op::Get("tl.fast_div");
  return op;
}

const Op &fast_mod() {
  static const Op &op = Op::Get("tl.fast_mod");
  return op;
}

const Op &musa_fast_div() {
  static const Op &op = Op::Get("tl.musa_fast_div");
  return op;
}

const Op &musa_fast_mod() {
  static const Op &op = Op::Get("tl.musa_fast_mod");
  return op;
}

const Op &musa_fast_div_device() {
  static const Op &op = Op::Get("tl.musa_fast_div_device");
  return op;
}

const Op &musa_fast_mod_device() {
  static const Op &op = Op::Get("tl.musa_fast_mod_device");
  return op;
}

TVM_REGISTER_OP("tl.fast_div")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "fast_div");

TVM_REGISTER_OP("tl.fast_mod")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "fast_mod");

TVM_REGISTER_OP("tl.musa_fast_div")
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "musa_fast_div");

TVM_REGISTER_OP("tl.musa_fast_mod")
    .set_num_inputs(4)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "musa_fast_mod");

TVM_REGISTER_OP("tl.musa_fast_div_device")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "musa_fast_div_device");

TVM_REGISTER_OP("tl.musa_fast_mod_device")
    .set_num_inputs(2)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kPure))
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "musa_fast_mod_device");

} // namespace tl
} // namespace tvm
