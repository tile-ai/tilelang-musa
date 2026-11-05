/*!
 * \file lower_musa_accelerated_ops.cc
 * \brief Lower MUSA-friendly compute patterns into explicit TL ops.
 */

#include "support/check.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/target/target.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

const CastNode *AsFloat16ToFloat32VectorCast(const PrimExpr &expr) {
  const auto *cast = expr.as<CastNode>();
  if (!cast) {
    return nullptr;
  }

  DataType from_ty = cast->value.dtype();
  DataType to_ty = cast->dtype;
  if (!from_ty.is_float16() || !to_ty.is_float() || to_ty.bits() != 32) {
    return nullptr;
  }
  if (from_ty.lanes() != to_ty.lanes()) {
    return nullptr;
  }
  return cast;
}

const BroadcastNode *AsFloat32ScalarBroadcast(const PrimExpr &expr) {
  const auto *broadcast = expr.as<BroadcastNode>();
  if (!broadcast) {
    return nullptr;
  }

  DataType broadcast_ty = broadcast->dtype;
  DataType value_ty = broadcast->value.dtype();
  if (!broadcast_ty.is_float() || broadcast_ty.bits() != 32 ||
      !value_ty.is_float() || value_ty.bits() != 32 || !value_ty.is_scalar()) {
    return nullptr;
  }
  return broadcast;
}

bool MatchHalfFloatMulToBFloat16(DataType target_ty, const PrimExpr &expr,
                                 PrimExpr *half_src, PrimExpr *float_src) {
  if (!target_ty.is_bfloat16()) {
    return false;
  }

  int lanes = target_ty.lanes();
  if (lanes < 4 || lanes > 16 || lanes % 4 != 0) {
    return false;
  }

  const auto *mul = expr.as<MulNode>();
  if (!mul) {
    return false;
  }

  auto match_side = [&](const PrimExpr &half_cast_expr,
                        const PrimExpr &float_expr) {
    const auto *half_cast = AsFloat16ToFloat32VectorCast(half_cast_expr);
    if (!half_cast) {
      return false;
    }

    if (half_cast->dtype.lanes() != lanes) {
      return false;
    }

    DataType float_ty = float_expr.dtype();
    if (!float_ty.is_float() || float_ty.bits() != 32 ||
        float_ty.lanes() != lanes) {
      return false;
    }

    *half_src = half_cast->value;
    if (const auto *broadcast = AsFloat32ScalarBroadcast(float_expr)) {
      *float_src = broadcast->value;
    } else {
      if (lanes > 8) {
        return false;
      }
      *float_src = float_expr;
    }
    return true;
  };

  return match_side(mul->a, mul->b) || match_side(mul->b, mul->a);
}

class LowerMUSAAcceleratedOpsRewriter : public StmtExprMutator {
public:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(tl::mul_half_float_to_bfloat16_x4())) {
      return StmtExprMutator::VisitExpr_(op);
    }

    ICHECK_EQ(op->args.size(), 2);
    PrimExpr half_src = this->VisitExpr(op->args[0]);
    PrimExpr float_src = this->VisitExpr(op->args[1]);
    return MakeExplicitMulHalfFloatToBFloat16X4Call(op->dtype, half_src,
                                                    float_src);
  }

  PrimExpr VisitExpr_(const CastNode *op) final {
    PrimExpr value = this->VisitExpr(op->value);

    PrimExpr half_src;
    PrimExpr float_src;
    if (MatchHalfFloatMulToBFloat16(op->dtype, value, &half_src, &float_src)) {
      return MakeChunkedMulHalfFloatToBFloat16X4Call(op->dtype, half_src,
                                                     float_src);
    }

    if (value.same_as(op->value)) {
      return ffi::GetRef<PrimExpr>(op);
    }
    return Cast(op->dtype, value);
  }

private:
  PrimExpr MakeExplicitMulHalfFloatToBFloat16X4Call(DataType result_ty,
                                                    PrimExpr half_src,
                                                    PrimExpr float_src) {
    ICHECK(result_ty.is_bfloat16() && result_ty.lanes() == 4)
        << "T.mul_half_float_to_bfloat16_x4 expects a bfloat16x4 result, "
           "but got "
        << result_ty;

    DataType half_ty = half_src.dtype();
    ICHECK(half_ty.is_float16() && half_ty.lanes() == 4)
        << "T.mul_half_float_to_bfloat16_x4 expects lhs to be float16x4, "
           "but got "
        << half_ty;

    DataType float_ty = float_src.dtype();
    ICHECK(float_ty.is_float() && float_ty.bits() == 32 &&
           float_ty.lanes() == 4)
        << "T.mul_half_float_to_bfloat16_x4 expects rhs to be float32x4, "
           "but got "
        << float_ty;

    return Call(result_ty, tl::mul_half_float_to_bfloat16_x4(),
                {half_src, float_src});
  }

  PrimExpr MakeChunkedMulHalfFloatToBFloat16X4Call(DataType result_ty,
                                                   PrimExpr half_src,
                                                   PrimExpr float_src) {
    int lanes = result_ty.lanes();
    ICHECK(result_ty.is_bfloat16() && lanes >= 4 && lanes <= 16 &&
           lanes % 4 == 0)
        << "T.mul_half_float_to_bfloat16_x4 expects a bfloat16 vector result "
           "with 4, 8, or 16 lanes, but got "
        << result_ty;

    DataType half_ty = half_src.dtype();
    ICHECK(half_ty.is_float16() && half_ty.lanes() == lanes)
        << "T.mul_half_float_to_bfloat16_x4 expects lhs to be float16x" << lanes
        << ", but got " << half_ty;

    DataType float_ty = float_src.dtype();
    ICHECK(float_ty.is_float() && float_ty.bits() == 32)
        << "T.mul_half_float_to_bfloat16_x4 expects rhs to be float32 scalar "
           "or float32x"
        << lanes << ", but got " << float_ty;

    if (float_ty.is_scalar()) {
      return Call(result_ty, tl::mul_half_float_to_bfloat16_x4(),
                  {half_src, float_src});
    }

    ICHECK_EQ(float_ty.lanes(), lanes)
        << "T.mul_half_float_to_bfloat16_x4 expects rhs lanes to match lhs "
           "lanes, but got "
        << float_ty << " vs " << half_ty;

    if (const auto *broadcast = AsFloat32ScalarBroadcast(float_src)) {
      return Call(result_ty, tl::mul_half_float_to_bfloat16_x4(),
                  {half_src, broadcast->value});
    }

    ICHECK_LE(lanes, 8)
        << "T.mul_half_float_to_bfloat16_x4 only supports non-broadcast "
           "float32 vector rhs up to 8 lanes.";
    return Call(result_ty, tl::mul_half_float_to_bfloat16_x4(),
                {half_src, float_src});
  }
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass LowerMUSAAcceleratedOps() {
  auto pass_func = [](PrimFunc f, const IRModule &m, PassContext ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || target.value()->kind->name != "musa") {
      return f;
    }

    auto *n = f.CopyOnWrite();
    n->body = LowerMUSAAcceleratedOpsRewriter()(std::move(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerMUSAAcceleratedOps", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerMUSAAcceleratedOps",
                        LowerMUSAAcceleratedOps);
}

} // namespace tl
} // namespace tvm
