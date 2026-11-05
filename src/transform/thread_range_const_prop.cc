/*!
 * \file thread_range_const_prop.cc
 * \brief Limited constant propagation from threadIdx branch ranges.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <string>
#include <unordered_map>
#include <unordered_set>

#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

struct IntRange {
  int64_t min{0};
  int64_t max{-1};

  bool valid() const { return min <= max; }
};

struct AffineThreadExpr {
  const VarNode *var{nullptr};
  int64_t offset{0};
};

enum class CompareKind {
  kLT,
  kLE,
  kGT,
  kGE,
  kEQ,
};

std::optional<int64_t> GetConstInt(const PrimExpr &expr) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return imm->value;
  }
  return std::nullopt;
}

std::optional<int64_t> CheckedAdd(int64_t a, int64_t b) {
  __int128 value = static_cast<__int128>(a) + static_cast<__int128>(b);
  if (value < std::numeric_limits<int64_t>::min() ||
      value > std::numeric_limits<int64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<int64_t>(value);
}

std::optional<int64_t> CheckedSub(int64_t a, int64_t b) {
  __int128 value = static_cast<__int128>(a) - static_cast<__int128>(b);
  if (value < std::numeric_limits<int64_t>::min() ||
      value > std::numeric_limits<int64_t>::max()) {
    return std::nullopt;
  }
  return static_cast<int64_t>(value);
}

int64_t FloorDivInt(int64_t a, int64_t b) {
  ICHECK_GT(b, 0);
  if (a >= 0) {
    return a / b;
  }
  return -(((-a) + b - 1) / b);
}

PrimExpr MakeIntImmLike(const PrimExpr &expr, int64_t value) {
  return make_const(expr.dtype(), value, expr->span);
}

bool IsTargetThreadIdxTag(const std::string &tag) {
  return tag == "threadIdx.x";
}

class ThreadRangeConstPropRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt stmt) {
    ThreadRangeConstPropRewriter rewriter;
    return rewriter(std::move(stmt));
  }

private:
  using RangeMap = std::unordered_map<const VarNode *, IntRange>;
  using ThreadVarSet = std::unordered_set<const VarNode *>;

  class ScopedState {
  public:
    explicit ScopedState(ThreadRangeConstPropRewriter *rewriter)
        : rewriter_(rewriter), saved_ranges_(rewriter->ranges_),
          saved_thread_vars_(rewriter->thread_vars_) {}

    ~ScopedState() {
      rewriter_->ranges_ = std::move(saved_ranges_);
      rewriter_->thread_vars_ = std::move(saved_thread_vars_);
    }

  private:
    ThreadRangeConstPropRewriter *rewriter_;
    RangeMap saved_ranges_;
    ThreadVarSet saved_thread_vars_;
  };

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (IsTargetThreadIdxTag(iv->thread_tag)) {
        std::optional<int64_t> extent = GetConstInt(op->value);
        if (extent.has_value() && extent.value() > 0) {
          ScopedState scope(this);
          const VarNode *var = iv->var.get();
          thread_vars_.insert(var);
          ranges_[var] = IntRange{0, extent.value() - 1};
          Stmt body = VisitStmt(op->body);
          return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    PrimExpr condition = VisitExpr(op->condition);

    Stmt then_case;
    {
      ScopedState scope(this);
      ApplyCondition(op->condition, true);
      then_case = VisitStmt(op->then_case);
    }

    Stmt else_case;
    if (op->else_case.defined()) {
      ScopedState scope(this);
      ApplyCondition(op->condition, false);
      else_case = VisitStmt(op->else_case.value());
    }

    return IfThenElse(condition, then_case, else_case, op->span);
  }

  PrimExpr VisitExpr_(const FloorDivNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *node = expr.as<FloorDivNode>();
    if (node == nullptr) {
      return expr;
    }

    std::optional<int64_t> divisor = GetConstInt(node->b);
    std::optional<IntRange> range = EvalThreadExprRange(node->a);
    if (!divisor.has_value() || divisor.value() <= 0 || !range.has_value() ||
        !range.value().valid()) {
      return expr;
    }

    int64_t lo = FloorDivInt(range.value().min, divisor.value());
    int64_t hi = FloorDivInt(range.value().max, divisor.value());
    if (lo == hi) {
      return MakeIntImmLike(expr, lo);
    }
    return expr;
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *node = expr.as<CallNode>();
    if (node == nullptr || node->args.size() != 2) {
      return expr;
    }

    if (node->op.same_as(builtin::shift_right())) {
      return TryFoldShiftRight(expr, node);
    }
    return expr;
  }

  PrimExpr TryFoldShiftRight(const PrimExpr &expr, const CallNode *op) const {
    std::optional<IntRange> range = EvalThreadExprRange(op->args[0]);
    std::optional<int64_t> shift = GetConstInt(op->args[1]);
    DataType value_dtype = op->args[0].dtype();
    if (!range.has_value() || !range.value().valid() || !shift.has_value() ||
        !value_dtype.is_scalar() ||
        !(value_dtype.is_int() || value_dtype.is_uint()) || shift.value() < 0 ||
        shift.value() >= value_dtype.bits() || range.value().min < 0) {
      return expr;
    }

    int64_t lo = range.value().min >> shift.value();
    int64_t hi = range.value().max >> shift.value();
    if (lo == hi) {
      return MakeIntImmLike(expr, lo);
    }
    return expr;
  }

  std::optional<AffineThreadExpr>
  MatchAffineThreadExpr(const PrimExpr &expr) const {
    if (const auto *var = expr.as<VarNode>()) {
      if (thread_vars_.count(var)) {
        return AffineThreadExpr{var, 0};
      }
      return std::nullopt;
    }

    if (const auto *add = expr.as<AddNode>()) {
      if (auto lhs = MatchAffineThreadExpr(add->a)) {
        if (auto rhs = GetConstInt(add->b)) {
          if (auto offset = CheckedAdd(lhs->offset, rhs.value())) {
            return AffineThreadExpr{lhs->var, offset.value()};
          }
        }
      }
      if (auto rhs = MatchAffineThreadExpr(add->b)) {
        if (auto lhs_const = GetConstInt(add->a)) {
          if (auto offset = CheckedAdd(rhs->offset, lhs_const.value())) {
            return AffineThreadExpr{rhs->var, offset.value()};
          }
        }
      }
      return std::nullopt;
    }

    if (const auto *sub = expr.as<SubNode>()) {
      if (auto lhs = MatchAffineThreadExpr(sub->a)) {
        if (auto rhs = GetConstInt(sub->b)) {
          if (auto offset = CheckedSub(lhs->offset, rhs.value())) {
            return AffineThreadExpr{lhs->var, offset.value()};
          }
        }
      }
    }

    return std::nullopt;
  }

  std::optional<IntRange> EvalThreadExprRange(const PrimExpr &expr) const {
    std::optional<AffineThreadExpr> affine = MatchAffineThreadExpr(expr);
    if (!affine.has_value()) {
      return std::nullopt;
    }

    auto it = ranges_.find(affine->var);
    if (it == ranges_.end() || !it->second.valid()) {
      return std::nullopt;
    }

    std::optional<int64_t> min = CheckedAdd(it->second.min, affine->offset);
    std::optional<int64_t> max = CheckedAdd(it->second.max, affine->offset);
    if (!min.has_value() || !max.has_value()) {
      return std::nullopt;
    }
    return IntRange{min.value(), max.value()};
  }

  void ApplyCondition(const PrimExpr &condition, bool truth_value) {
    if (const auto *and_op = condition.as<AndNode>()) {
      if (truth_value) {
        ApplyCondition(and_op->a, true);
        ApplyCondition(and_op->b, true);
      }
      return;
    }
    if (const auto *or_op = condition.as<OrNode>()) {
      if (!truth_value) {
        ApplyCondition(or_op->a, false);
        ApplyCondition(or_op->b, false);
      }
      return;
    }
    if (const auto *not_op = condition.as<NotNode>()) {
      ApplyCondition(not_op->a, !truth_value);
      return;
    }

    if (const auto *op = condition.as<LTNode>()) {
      ApplyComparison(op->a, op->b, CompareKind::kLT, truth_value);
    } else if (const auto *op = condition.as<LENode>()) {
      ApplyComparison(op->a, op->b, CompareKind::kLE, truth_value);
    } else if (const auto *op = condition.as<GTNode>()) {
      ApplyComparison(op->a, op->b, CompareKind::kGT, truth_value);
    } else if (const auto *op = condition.as<GENode>()) {
      ApplyComparison(op->a, op->b, CompareKind::kGE, truth_value);
    } else if (const auto *op = condition.as<EQNode>()) {
      ApplyComparison(op->a, op->b, CompareKind::kEQ, truth_value);
    }
  }

  void ApplyComparison(const PrimExpr &lhs, const PrimExpr &rhs,
                       CompareKind kind, bool truth_value) {
    if (auto lhs_affine = MatchAffineThreadExpr(lhs)) {
      if (auto rhs_const = GetConstInt(rhs)) {
        auto threshold = CheckedSub(rhs_const.value(), lhs_affine->offset);
        if (threshold.has_value()) {
          ApplyVarComparison(lhs_affine->var, kind, threshold.value(),
                             truth_value);
        }
        return;
      }
    }

    if (auto rhs_affine = MatchAffineThreadExpr(rhs)) {
      if (auto lhs_const = GetConstInt(lhs)) {
        auto threshold = CheckedSub(lhs_const.value(), rhs_affine->offset);
        if (threshold.has_value()) {
          ApplyVarComparison(rhs_affine->var, Reverse(kind), threshold.value(),
                             truth_value);
        }
      }
    }
  }

  CompareKind Reverse(CompareKind kind) const {
    switch (kind) {
    case CompareKind::kLT:
      return CompareKind::kGT;
    case CompareKind::kLE:
      return CompareKind::kGE;
    case CompareKind::kGT:
      return CompareKind::kLT;
    case CompareKind::kGE:
      return CompareKind::kLE;
    case CompareKind::kEQ:
      return CompareKind::kEQ;
    }
    LOG(FATAL) << "Unreachable";
    return CompareKind::kEQ;
  }

  void ApplyVarComparison(const VarNode *var, CompareKind kind, int64_t value,
                          bool truth_value) {
    if (!truth_value && kind == CompareKind::kEQ) {
      return;
    }

    if (!truth_value) {
      switch (kind) {
      case CompareKind::kLT:
        kind = CompareKind::kGE;
        break;
      case CompareKind::kLE:
        kind = CompareKind::kGT;
        break;
      case CompareKind::kGT:
        kind = CompareKind::kLE;
        break;
      case CompareKind::kGE:
        kind = CompareKind::kLT;
        break;
      case CompareKind::kEQ:
        return;
      }
    }

    auto it = ranges_.find(var);
    if (it == ranges_.end()) {
      return;
    }

    IntRange range = it->second;
    switch (kind) {
    case CompareKind::kLT:
      if (value == std::numeric_limits<int64_t>::min()) {
        range.max = std::numeric_limits<int64_t>::min();
        range.min = 1;
      } else {
        range.max = std::min(range.max, value - 1);
      }
      break;
    case CompareKind::kLE:
      range.max = std::min(range.max, value);
      break;
    case CompareKind::kGT:
      if (value == std::numeric_limits<int64_t>::max()) {
        range.max = std::numeric_limits<int64_t>::min();
        range.min = 1;
      } else {
        range.min = std::max(range.min, value + 1);
      }
      break;
    case CompareKind::kGE:
      range.min = std::max(range.min, value);
      break;
    case CompareKind::kEQ:
      range.min = std::max(range.min, value);
      range.max = std::min(range.max, value);
      break;
    }
    ranges_[var] = range;
  }

  RangeMap ranges_;
  ThreadVarSet thread_vars_;
};

} // namespace

PrimFunc ThreadRangeConstProp(PrimFunc f) {
  auto *n = f.CopyOnWrite();
  n->body = ThreadRangeConstPropRewriter::Rewrite(std::move(n->body));
  return f;
}

namespace transform {

using namespace tirx::transform;

tvm::transform::Pass ThreadRangeConstProp() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return tl::ThreadRangeConstProp(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.ThreadRangeConstProp", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.ThreadRangeConstProp",
                        ThreadRangeConstProp);
}

} // namespace transform
} // namespace tl
} // namespace tvm
