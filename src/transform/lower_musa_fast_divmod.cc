/*!
 * \file lower_musa_fast_divmod.cc
 * \brief Prepare explicit MUSA fast-divmod on the host or device.
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <cstdint>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

class VarCollector : public ExprVisitor {
public:
  std::vector<Var> Collect(const PrimExpr &expr) {
    VisitExpr(expr);
    return vars_;
  }

private:
  void VisitExpr_(const VarNode *op) final {
    if (seen_.insert(op).second) {
      vars_.push_back(ffi::GetRef<Var>(op));
    }
  }

  std::unordered_set<const VarNode *> seen_;
  std::vector<Var> vars_;
};

class HostEvaluableExprChecker : public ExprVisitor {
public:
  explicit HostEvaluableExprChecker(
      const std::unordered_set<const VarNode *> &host_vars)
      : host_vars_(host_vars) {}

  bool Check(const PrimExpr &expr) {
    VisitExpr(expr);
    return host_evaluable_;
  }

private:
  void VisitExpr_(const VarNode *op) final {
    if (!host_vars_.count(op)) {
      host_evaluable_ = false;
    }
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    // A device buffer load cannot be used to form a host-side magic divisor.
    host_evaluable_ = false;
  }

  void VisitExpr_(const CallNode *op) final {
    // Calls may refer to device-only intrinsics or extern functions even when
    // they contain no thread-local Vars.  Keep them on the device unless a
    // future implementation explicitly proves a particular call host-safe.
    host_evaluable_ = false;
  }

  const std::unordered_set<const VarNode *> &host_vars_;
  bool host_evaluable_{true};
};

struct MagicNumbers {
  uint32_t multiplier;
  uint32_t shift_right;
};

MagicNumbers ComputeMagicNumbers(int32_t divisor) {
  ICHECK_GT(divisor, 0)
      << "T.fast_div, T.fast_mod, and T.fast_divmod require a strictly "
         "positive divisor";
  if (divisor == 1) {
    return {0u, 0u};
  }

  uint32_t denom = static_cast<uint32_t>(divisor);
  uint32_t ceil_log2 = 0;
  for (uint32_t x = denom - 1; x != 0; x >>= 1) {
    ++ceil_log2;
  }
  uint32_t p = 31 + ceil_log2;
  uint64_t multiplier = ((uint64_t{1} << p) + denom - 1) / denom;
  return {static_cast<uint32_t>(multiplier), p - 32};
}

class MUSAFastDivmodLowerer : public StmtExprMutator {
public:
  static PrimFunc Lower(PrimFunc f) {
    std::unordered_set<const VarNode *> host_vars;
    auto collect_expr_vars = [&host_vars](const PrimExpr &expr) {
      for (const Var &var : VarCollector().Collect(expr)) {
        host_vars.insert(var.get());
      }
    };
    for (const Var &param : f->params) {
      host_vars.insert(param.get());
    }
    for (const auto &[param, buffer] : f->buffer_map) {
      host_vars.insert(param.get());
      host_vars.insert(buffer->data.get());
      for (const PrimExpr &shape : buffer->shape) {
        collect_expr_vars(shape);
      }
      for (const PrimExpr &stride : buffer->strides) {
        collect_expr_vars(stride);
      }
      collect_expr_vars(buffer->elem_offset);
    }

    MUSAFastDivmodLowerer lowerer(std::move(host_vars));
    Stmt body = lowerer.VisitStmt(f->body);
    body = lowerer.WrapHostSetup(std::move(body));

    auto *n = f.CopyOnWrite();
    n->body = std::move(body);

    return f;
  }

private:
  struct DivisorInfo {
    PrimExpr divisor;
    PrimExpr multiplier;
    PrimExpr shift_right;
    bool needs_host_setup{false};
    bool needs_device_setup{false};
  };

  explicit MUSAFastDivmodLowerer(std::unordered_set<const VarNode *> host_vars)
      : host_vars_(std::move(host_vars)) {}

  static void CheckOperandTypes(const PrimExpr &dividend,
                                const PrimExpr &divisor) {
    ICHECK_EQ(dividend->dtype, DataType::Int(32))
        << "T.fast_div, T.fast_mod, and T.fast_divmod require an int32 "
           "dividend, but got "
        << dividend->dtype;
    ICHECK_EQ(divisor->dtype, DataType::Int(32))
        << "T.fast_div, T.fast_mod, and T.fast_divmod require an int32 "
           "divisor, but got "
        << divisor->dtype;
  }

  DivisorInfo &GetOrCreateDivisorInfo(const PrimExpr &divisor) {
    auto it = divisor_info_map_.find(divisor);
    if (it != divisor_info_map_.end()) {
      return divisors_[it->second];
    }

    DivisorInfo info;
    info.divisor = divisor;
    if (const auto *imm = divisor.as<IntImmNode>()) {
      MagicNumbers magic =
          ComputeMagicNumbers(static_cast<int32_t>(imm->value));
      info.multiplier = IntImm(DataType::UInt(32), magic.multiplier);
      info.shift_right = IntImm(DataType::UInt(32), magic.shift_right);
    } else {
      // Match mutlass::FastDivmod semantics: a divisor that is only available
      // in device code prepares its own multiplier and shift there.
      info.needs_device_setup = true;
    }

    size_t index = divisors_.size();
    divisor_info_map_.emplace(divisor, index);
    divisors_.push_back(std::move(info));
    return divisors_.back();
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    bool is_fast_div = op->op.same_as(fast_div());
    bool is_fast_mod = op->op.same_as(fast_mod());
    if (!is_fast_div && !is_fast_mod) {
      return StmtExprMutator::VisitExpr_(op);
    }

    ICHECK_EQ(op->args.size(), 2U)
        << (is_fast_div ? "tl.fast_div" : "tl.fast_mod")
        << " expects dividend and divisor";
    PrimExpr dividend = VisitExpr(op->args[0]);
    PrimExpr divisor = VisitExpr(op->args[1]);
    CheckOperandTypes(dividend, divisor);
    DivisorInfo &info = GetOrCreateDivisorInfo(divisor);
    if (info.needs_device_setup) {
      return Call(op->dtype,
                  is_fast_div ? musa_fast_div_device() : musa_fast_mod_device(),
                  {dividend, divisor}, op->annotations, op->span);
    }
    return Call(op->dtype, is_fast_div ? musa_fast_div() : musa_fast_mod(),
                {dividend, divisor, info.multiplier, info.shift_right},
                op->annotations, op->span);
  }

  static PrimExpr MakeCeilLog2(const PrimExpr &divisor) {
    DataType u32 = DataType::UInt(32);
    PrimExpr divisor_u32 = Cast(u32, divisor);
    // clz(0) is target-dependent.  Keep its argument positive and select the
    // mathematically correct zero result for divisor == 1 afterwards.
    PrimExpr divisor_minus_one =
        max(divisor_u32 - make_const(u32, 1), make_const(u32, 1));
    PrimExpr leading_zeros = Call(u32, Op::Get("tir.clz"), {divisor_minus_one});
    PrimExpr ceil_log2 = make_const(u32, 32) - leading_zeros;
    return Select(divisor == make_const(divisor.dtype(), 1), make_const(u32, 0),
                  ceil_log2);
  }

  static bool SameFastDivArgs(const CallNode &lhs, const CallNode &rhs) {
    if (lhs.args.size() != rhs.args.size() ||
        (lhs.args.size() != 2U && lhs.args.size() != 4U)) {
      return false;
    }
    ExprDeepEqual equal;
    for (size_t i = 0; i < lhs.args.size(); ++i) {
      if (!equal(lhs.args[i], rhs.args[i])) {
        return false;
      }
    }
    return true;
  }

  static PrimExpr MakeMultiplier(const PrimExpr &divisor) {
    DataType u32 = DataType::UInt(32);
    DataType i64 = DataType::Int(64);
    PrimExpr p = make_const(u32, 31) + MakeCeilLog2(divisor);
    PrimExpr numerator = make_const(i64, 1) << Cast(i64, p);
    PrimExpr divisor_i64 = Cast(i64, divisor);
    PrimExpr multiplier =
        (numerator + divisor_i64 - make_const(i64, 1)) / divisor_i64;
    return Select(divisor == make_const(divisor.dtype(), 1), make_const(u32, 0),
                  Cast(u32, multiplier));
  }

  static PrimExpr MakeShiftRight(const PrimExpr &divisor) {
    DataType u32 = DataType::UInt(32);
    PrimExpr p = make_const(u32, 31) + MakeCeilLog2(divisor);
    PrimExpr shift_right = p - make_const(u32, 32);
    return Select(divisor == make_const(divisor.dtype(), 1), make_const(u32, 0),
                  shift_right);
  }

  Stmt WrapHostSetup(Stmt body) const { return body; }

  std::unordered_set<const VarNode *> host_vars_;
  std::unordered_map<PrimExpr, size_t, StructuralHash, ExprDeepEqual>
      divisor_info_map_;
  std::vector<DivisorInfo> divisors_;
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass LowerMUSAFastDivmod() {
  auto pass_func = [](PrimFunc f, const IRModule &m, PassContext ctx) {
    return MUSAFastDivmodLowerer::Lower(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerMUSAFastDivmod", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerMUSAFastDivmod",
                        LowerMUSAFastDivmod);
}

} // namespace tl
} // namespace tvm
