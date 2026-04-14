/*!
 * \brief Merge narrow async-copy loops into a wider async-copy call.
 * \file merge_async_copy.cc
 */

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cstdint>
#include <optional>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

class AsyncCopyMerger : public StmtMutator {
public:
  Stmt VisitStmt_(const ForNode *op) final {
    Stmt stmt = StmtMutator::VisitStmt_(op);
    const auto *loop = stmt.as<ForNode>();
    if (loop == nullptr) {
      return stmt;
    }
    Optional<Stmt> merged = TryMergeLoop(loop);
    return merged.defined() ? merged.value() : stmt;
  }

private:
  struct AccessPtrInfo {
    BufferLoad base_load;
    PrimExpr extent;
    PrimExpr rw_mask;
  };

  struct AsyncCopyInfo {
    DataType dtype;
    Op op;
    AccessPtrInfo dst;
    AccessPtrInfo src;
    IntImm bytes;
    Optional<PrimExpr> predicate;
    Optional<PrimExpr> robust_base;
    Optional<PrimExpr> robust_size;
  };

  struct AttrInfo {
    ffi::Any node;
    ffi::String attr_key;
    PrimExpr value;
  };

  struct WrappedAsyncCopy {
    AsyncCopyInfo copy;
    std::vector<AttrInfo> attrs;
  };

  static bool IsAsyncCopyOp(const Op &op) {
    return op.same_as(builtin::ptx_cp_async()) ||
           op.same_as(tl::ptx_cp_async()) ||
           op.same_as(tl::musa_cp_async_robust());
  }

  static std::optional<AccessPtrInfo> MatchAccessPtr(const PrimExpr &expr) {
    const auto *call = expr.as<CallNode>();
    if (call == nullptr || !call->op.same_as(tl::access_ptr()) ||
        call->args.size() != 3U) {
      return std::nullopt;
    }
    const auto *load = call->args[0].as<BufferLoadNode>();
    if (load == nullptr || load->indices.size() != 1U) {
      return std::nullopt;
    }
    return AccessPtrInfo{Downcast<BufferLoad>(call->args[0]), call->args[1],
                         call->args[2]};
  }

  static std::optional<AsyncCopyInfo> MatchAsyncCopy(const CallNode *call) {
    if (call == nullptr) {
      return std::nullopt;
    }

    Optional<Op> op = call->op.as<Op>();
    if (!op.defined() || !IsAsyncCopyOp(op.value())) {
      return std::nullopt;
    }

    std::optional<AccessPtrInfo> dst = MatchAccessPtr(call->args[0]);
    std::optional<AccessPtrInfo> src = MatchAccessPtr(call->args[1]);
    const auto *bytes = call->args[2].as<IntImmNode>();
    if (!dst.has_value() || !src.has_value() || bytes == nullptr) {
      return std::nullopt;
    }

    AsyncCopyInfo info{call->dtype,
                       op.value(),
                       dst.value(),
                       src.value(),
                       Downcast<IntImm>(call->args[2]),
                       Optional<PrimExpr>(),
                       Optional<PrimExpr>(),
                       Optional<PrimExpr>()};

    if (op.value().same_as(tl::musa_cp_async_robust())) {
      if (call->args.size() != 5U && call->args.size() != 6U) {
        return std::nullopt;
      }
      info.robust_base = call->args[3];
      info.robust_size = call->args[4];
      if (call->args.size() == 6U) {
        info.predicate = call->args[5];
      }
    } else {
      if (call->args.size() != 3U && call->args.size() != 4U) {
        return std::nullopt;
      }
      if (call->args.size() == 4U) {
        info.predicate = call->args[3];
      }
    }
    return info;
  }

  static std::optional<WrappedAsyncCopy>
  MatchWrappedAsyncCopy(const Stmt &stmt) {
    std::vector<AttrInfo> attrs;
    Stmt current = stmt;
    while (const auto *attr = current.as<AttrStmtNode>()) {
      attrs.push_back(AttrInfo{attr->node, attr->attr_key, attr->value});
      current = attr->body;
    }

    const auto *eval = current.as<EvaluateNode>();
    if (eval == nullptr) {
      return std::nullopt;
    }
    const auto *call = eval->value.as<CallNode>();
    std::optional<AsyncCopyInfo> info = MatchAsyncCopy(call);
    if (!info.has_value()) {
      return std::nullopt;
    }
    return WrappedAsyncCopy{info.value(), attrs};
  }

  static Stmt Rewrap(const std::vector<AttrInfo> &attrs, Stmt body) {
    for (auto it = attrs.rbegin(); it != attrs.rend(); ++it) {
      body = AttrStmt((*it).node, (*it).attr_key, (*it).value, body);
    }
    return body;
  }

  Optional<Stmt> TryMergeLoop(const ForNode *loop) {
    int64_t min = 0;
    int64_t extent = 0;
    if (!TryGetConstInt64(loop->min, &min) ||
        !TryGetConstInt64(loop->extent, &extent) || extent <= 1) {
      return Optional<Stmt>();
    }

    std::optional<WrappedAsyncCopy> wrapped = MatchWrappedAsyncCopy(loop->body);
    if (!wrapped.has_value()) {
      return Optional<Stmt>();
    }

    const AsyncCopyInfo &copy = wrapped.value().copy;
    int64_t total_bytes = static_cast<int64_t>(copy.bytes->value) * extent;
    if (!IsValidCPAsyncTransferBytes(static_cast<int>(total_bytes))) {
      return Optional<Stmt>();
    }
    if (!AttrsInvariant(wrapped.value().attrs, loop->loop_var)) {
      return Optional<Stmt>();
    }
    if (!AccessPtrHasConstantStep(
            copy.dst, loop->loop_var, min, extent,
            StepInElements(copy.dst.base_load, copy.bytes)) ||
        !AccessPtrHasConstantStep(
            copy.src, loop->loop_var, min, extent,
            StepInElements(copy.src.base_load, copy.bytes))) {
      return Optional<Stmt>();
    }
    if (!AsyncCopyInvariant(copy, loop->loop_var)) {
      return Optional<Stmt>();
    }

    AsyncCopyInfo merged = copy;
    merged.dst = MakeMergedAccessPtr(copy.dst, loop->loop_var, min, extent);
    merged.src = MakeMergedAccessPtr(copy.src, loop->loop_var, min, extent);
    merged.bytes = IntImm(copy.bytes->dtype, total_bytes);
    return Rewrap(wrapped.value().attrs, MakeAsyncCopyStmt(merged));
  }

  Stmt MakeAsyncCopyStmt(const AsyncCopyInfo &copy) {
    Array<PrimExpr> args{
        MakeAccessPtr(copy.dst),
        MakeAccessPtr(copy.src),
        copy.bytes,
    };
    if (copy.robust_base.defined()) {
      args.push_back(copy.robust_base.value());
      args.push_back(copy.robust_size.value());
    }
    if (copy.predicate.defined()) {
      args.push_back(copy.predicate.value());
    }
    return Evaluate(Call(copy.dtype, copy.op, args));
  }

  PrimExpr MakeAccessPtr(const AccessPtrInfo &ptr) const {
    return Call(DataType::Handle(), tl::access_ptr(),
                {ptr.base_load, ptr.extent, ptr.rw_mask});
  }

  AccessPtrInfo MakeMergedAccessPtr(const AccessPtrInfo &ptr,
                                    const Var &loop_var, int64_t min,
                                    int64_t extent) {
    PrimExpr base = SubstituteValue(ptr.base_load->indices[0], loop_var,
                                    IntImm(loop_var->dtype, min));
    BufferLoad base_load = BufferLoad(ptr.base_load->buffer, {base});
    PrimExpr merged_extent =
        analyzer_.Simplify(ptr.extent * IntImm(ptr.extent.dtype(), extent));
    return AccessPtrInfo{base_load, merged_extent, ptr.rw_mask};
  }

  bool AttrsInvariant(const std::vector<AttrInfo> &attrs, const Var &loop_var) {
    for (const AttrInfo &attr : attrs) {
      if (UsesVar(attr.value, [v = loop_var.get()](const VarNode *var) {
            return var == v;
          })) {
        return false;
      }
    }
    return true;
  }

  bool AsyncCopyInvariant(const AsyncCopyInfo &copy, const Var &loop_var) {
    if (!ExprInvariant(copy.dst.extent, loop_var) ||
        !ExprInvariant(copy.src.extent, loop_var) ||
        !ExprInvariant(copy.dst.rw_mask, loop_var) ||
        !ExprInvariant(copy.src.rw_mask, loop_var)) {
      return false;
    }
    if (copy.predicate.defined() &&
        !ExprInvariant(copy.predicate.value(), loop_var)) {
      return false;
    }
    if (copy.robust_base.defined() &&
        !ExprInvariant(copy.robust_base.value(), loop_var)) {
      return false;
    }
    if (copy.robust_size.defined() &&
        !ExprInvariant(copy.robust_size.value(), loop_var)) {
      return false;
    }
    return true;
  }

  bool ExprInvariant(const PrimExpr &expr, const Var &loop_var) {
    return !UsesVar(
        expr, [v = loop_var.get()](const VarNode *var) { return var == v; });
  }

  bool AccessPtrHasConstantStep(const AccessPtrInfo &ptr, const Var &loop_var,
                                int64_t min, int64_t extent,
                                int64_t expected_step) {
    if (expected_step <= 0) {
      return false;
    }
    PrimExpr prev = SubstituteValue(ptr.base_load->indices[0], loop_var,
                                    IntImm(loop_var->dtype, min));
    for (int64_t i = 1; i < extent; ++i) {
      PrimExpr curr = SubstituteValue(ptr.base_load->indices[0], loop_var,
                                      IntImm(loop_var->dtype, min + i));
      PrimExpr delta = analyzer_.Simplify(curr - prev);
      int64_t delta_value = 0;
      if (!TryGetConstInt64(delta, &delta_value) ||
          delta_value != expected_step) {
        return false;
      }
      prev = curr;
    }
    return true;
  }

  static bool TryGetConstInt64(const PrimExpr &expr, int64_t *value) {
    if (const auto *imm = expr.as<IntImmNode>()) {
      *value = imm->value;
      return true;
    }
    return false;
  }

  static int64_t StepInElements(const BufferLoad &load, const IntImm &bytes) {
    int elem_bytes = load->buffer->dtype.bytes();
    if (elem_bytes <= 0 || bytes->value % elem_bytes != 0) {
      return 0;
    }
    return bytes->value / elem_bytes;
  }

  PrimExpr SubstituteValue(const PrimExpr &expr, const Var &var,
                           const PrimExpr &value) {
    return analyzer_.Simplify(Substitute(expr, {{var, value}}));
  }

  arith::Analyzer analyzer_;
};

} // namespace

namespace transform {

tvm::transform::Pass MergeAsyncCopy() {
  auto pass_func = [](PrimFunc f, const IRModule &m,
                      const tvm::transform::PassContext &ctx) {
    if (!f.defined() || !f->body.defined()) {
      return f;
    }
    AsyncCopyMerger merger;
    auto *n = f.CopyOnWrite();
    n->body = merger(std::move(n->body));
    return f;
  };
  return tvm::tir::transform::CreatePrimFuncPass(pass_func, 0,
                                                 "tl.MergeAsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.MergeAsyncCopy", MergeAsyncCopy);
}

} // namespace transform

} // namespace tl
} // namespace tvm
