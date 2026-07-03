/*!
 * \file unified_barrier.cc
 * \brief Rewrite partial shared sync into mbarrier arrive/wait for MUSA.
 */
#include "../op/builtin.h"
#include "support/check.h"
#include "tvm/runtime/logging.h"
#include "tvm/tirx/buffer.h"
#include "tvm/tirx/builtin.h"
#include "tvm/tirx/expr.h"
#include "tvm/tirx/op.h"
#include "tvm/tirx/stmt_functor.h"
#include "tvm/tirx/transform.h"
#include <tvm/ffi/reflection/registry.h>

#include <optional>
#include <string>
#include <unordered_map>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

// Collect barrier metadata and rewrite partial sync to internal placeholders.
class UnifiedBarrierPrepass : public StmtExprMutator {
public:
  const std::vector<PrimExpr> &partial_sync_thread_counts() const {
    return partial_sync_thread_counts_;
  }
  int barrier_count() const { return barrier_count_; }
  int sync_count() const {
    return static_cast<int>(partial_sync_thread_counts_.size());
  }
  int placeholder_count() const { return placeholder_count_; }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::tvm_storage_sync())) {
        // Rewrite partial thread sync IR from `tvm_storage_sync("shared.dyn",
        // barrier_id, count)` to `partial_barrier_sync(offset)`.
        if (auto rewritten = RewriteStorageSync(call)) {
          return rewritten.value();
        }
      } else if (call->op.same_as(builtin::create_barriers())) {
        // collect barrier count from `T.create_barriers(barrier_count)`
        HandleCreateBarriers(call);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  static bool IsSharedSyncScope(const StringImmNode *scope) {
    return scope && (scope->value == "shared" || scope->value == "shared.dyn");
  }

  PrimExpr VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(barrier_id_placeholder())) {
      placeholder_count_++;
    }
    return StmtExprMutator::VisitExpr_(call);
  }

  std::optional<Stmt> RewriteStorageSync(const CallNode *call) {
    if (call->args.size() != 3) {
      return std::nullopt;
    }
    const auto *scope = call->args[0].as<StringImmNode>();
    if (!IsSharedSyncScope(scope)) {
      return std::nullopt;
    }
    int sync_offset = sync_count();
    PrimExpr thread_count = VisitExpr(call->args[2]);
    Array<PrimExpr> args = {IntImm(DataType::Int(32), sync_offset)};
    partial_sync_thread_counts_.push_back(thread_count);
    auto new_call = Call(call->dtype, partial_barrier_sync(), args);
    return Evaluate(new_call);
  }

  void HandleCreateBarriers(const CallNode *call) {
    if (call->args.size() != 1)
      return;
    if (const auto *n = call->args[0].as<IntImmNode>()) {
      barrier_count_ += static_cast<int>(n->value);
    }
  }

  std::vector<PrimExpr> partial_sync_thread_counts_;
  int barrier_count_{0};
  int placeholder_count_{0};
};

class MbarrierSyncRewriter : public StmtExprMutator {
public:
  MbarrierSyncRewriter(int base_count,
                       std::vector<std::pair<int, PrimExpr>> barrier_inits)
      : base_count_(base_count), barrier_inits_(std::move(barrier_inits)) {
    ICHECK(!barrier_inits_.empty());
  }

  // Make statements for barrier inits
  Array<Stmt> MakeInitStmts() {
    Array<Stmt> stmts;
    for (const auto &[id, thread_count] : barrier_inits_) {
      PrimExpr barrier = IntImm(DataType::Int(32), id);
      auto count = VisitExpr(thread_count);
      auto init =
          Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
               {barrier, count});
      stmts.push_back(Evaluate(init));
    }
    return stmts;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(tl::partial_barrier_sync())) {
        if (auto rewritten = RewritePartialBarrierSync(call)) {
          return rewritten.value();
        }
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // Rewrite partial_barrier_sync(offset) to partial_barrier_sync(id).
  std::optional<Stmt> RewritePartialBarrierSync(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 1);
    const auto *offset_imm = call->args[0].as<IntImmNode>();
    ICHECK(offset_imm) << "partial_barrier_sync offset must be IntImm";
    auto offset = offset_imm->value;
    int new_id = base_count_ + offset + 1;
    Array<PrimExpr> args = {IntImm(DataType::Int(32), new_id)};
    auto new_call = Call(call->dtype, partial_barrier_sync(), args);
    return Evaluate(new_call);
  }

  std::vector<std::pair<int, PrimExpr>> barrier_inits_;
  int base_count_{0};
};

class BarrierIdPlaceholderRewriter : public StmtExprMutator {
public:
  BarrierIdPlaceholderRewriter(int placeholder_start)
      : placeholder_next_id_(placeholder_start) {}

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(barrier_id_placeholder())) {
      int current_id = placeholder_next_id_ + 1;
      placeholder_next_id_++;
      return IntImm(DataType::Int(32), current_id);
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  int placeholder_next_id_{0};
};

class AllReduceRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body) {
    return AllReduceRewriter()(std::move(body));
  }

private:
  Stmt VisitStmt_(const BindNode *op) final {
    PrimExpr value = this->VisitExpr(op->value);

    if (const auto *imm = value.as<IntImmNode>()) {
      let_var_to_const_int_[op->var.get()] = imm->value;
    } else {
      let_var_to_const_int_.erase(op->var.get());
    }
    return Bind(op->var, value, op->span);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    PrimExpr expr = StmtExprMutator::VisitExpr_(op);
    const auto *call = expr.as<CallNode>();
    if (!call || !call->op.same_as(builtin::call_extern()) ||
        call->args.size() < 3) {
      return expr;
    }

    const auto *name_imm = call->args[0].as<StringImmNode>();
    if (!name_imm) {
      return expr;
    }
    const std::string &name = name_imm->value;
    if (name.rfind("tl::AllReduce<", 0) != 0 ||
        name.find("NamedBarrier<") != std::string::npos) {
      return expr;
    }
    std::string suffix;
    if (name.size() >= 12 && name.substr(name.size() - 12) == ">::run_batch") {
      suffix = ">::run_batch";
    } else if (name.size() >= 6 && name.substr(name.size() - 6) == ">::run") {
      suffix = ">::run";
    } else {
      return expr;
    }

    int64_t barrier_id = 0;
    if (const auto *imm = call->args[2].as<IntImmNode>()) {
      barrier_id = imm->value;
    } else if (const auto *var = call->args[2].as<VarNode>()) {
      auto it = let_var_to_const_int_.find(var);
      if (it == let_var_to_const_int_.end()) {
        return expr;
      }
      barrier_id = it->second;
    } else {
      return expr;
    }

    std::string new_name;
    std::string barrier_name =
        "tl::NamedBarrier<" + std::to_string(barrier_id) + ">";
    size_t sync_barrier_pos = name.find("tl::SyncThreadsBarrier");
    if (sync_barrier_pos != std::string::npos) {
      new_name = name;
      new_name.replace(sync_barrier_pos,
                       std::string("tl::SyncThreadsBarrier").size(),
                       barrier_name);
    } else {
      size_t run_pos = name.size() - suffix.size();
      new_name =
          name.substr(0, run_pos) + ", " + barrier_name + name.substr(run_pos);
    }
    Array<PrimExpr> new_args;
    new_args.reserve(call->args.size() - 1);
    new_args.push_back(StringImm(new_name));
    new_args.push_back(call->args[1]);
    for (size_t i = 3; i < call->args.size(); ++i) {
      new_args.push_back(call->args[i]);
    }
    return Call(call->dtype, call->op, new_args, call->annotations, call->span);
  }

  std::unordered_map<const VarNode *, int64_t> let_var_to_const_int_;
};

class CreateBarrierRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body, int new_barrier_count) {
    CreateBarrierRewriter rewriter;
    body = rewriter(std::move(body));

    // Insert a fresh create_barriers with the new count.
    Array<Stmt> stmts;
    auto create =
        Evaluate(Call(DataType::Handle(), builtin::create_barriers(),
                      {IntImm(DataType::Int(32), new_barrier_count)}));
    stmts.push_back(create);
    stmts.push_back(body);
    return SeqStmt(stmts);
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(builtin::create_barriers())) {
        // Drop existing create_barriers; we'll emit a single unified one later.
        return Stmt();
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }
};

class PartialBarrierSyncPhaseRewriter : public StmtExprMutator {
public:
  Stmt Rewrite(Stmt body) {
    body = this->VisitStmt(std::move(body));
    for (auto it = phase_vars_.rbegin(); it != phase_vars_.rend(); ++it) {
      const Buffer &buf = *it;
      Map<String, ffi::Any> annotations;
      annotations.Set(tl::attr::kLocalVarInit, IntImm(DataType::Int(32), 0));
      body = SeqStmt({AllocBuffer(buf, annotations), body});
    }
    return body;
  }

private:
  Stmt VisitStmt_(const EvaluateNode *op) final {
    if (const auto *call = op->value.as<CallNode>()) {
      if (call->op.same_as(tl::partial_barrier_sync())) {
        return RewritePartialBarrierSync(call);
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt RewritePartialBarrierSync(const CallNode *call) {
    ICHECK_EQ(call->args.size(), 1);
    Buffer phase_buf = MakePhaseVar();
    phase_vars_.push_back(phase_buf);
    Array<PrimExpr> indices = {IntImm(DataType::Int(32), 0)};
    PrimExpr phase = BufferLoad(phase_buf, indices);
    PrimExpr barrier = VisitExpr(call->args[0]);
    Stmt arrive = Evaluate(
        Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {barrier}));
    Stmt wait = Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(), {barrier, phase}));
    PrimExpr next_phase = bitwise_xor(phase, IntImm(DataType::Int(32), 1));
    Stmt store = BufferStore(phase_buf, next_phase, indices);
    return SeqStmt::Flatten(arrive, wait, store);
  }

  Buffer MakePhaseVar() {
    std::string name =
        "__partial_barrier_sync_phase_" + std::to_string(phase_vars_.size());
    Array<PrimExpr> shape = {IntImm(DataType::Int(32), 1)};
    return decl_buffer(shape, DataType::Int(32), name, "local.var");
  }

  std::vector<Buffer> phase_vars_;
};

PrimFunc RewriteUnifiedBarrier(PrimFunc f) {
  auto *n = f.CopyOnWrite();

  UnifiedBarrierPrepass prepass;
  // Run prepass on a copy of the body so we can still early-return the
  // untouched PrimFunc when there is no barrier to rewrite.
  Stmt body = prepass(n->body);
  int base_count = prepass.barrier_count();
  Array<Stmt> prefix;

  if (prepass.placeholder_count() != 0) {
    int placeholder_start = base_count;
    base_count += prepass.placeholder_count();
    BarrierIdPlaceholderRewriter rewriter(placeholder_start);
    body = rewriter(std::move(body));
  }

  body = AllReduceRewriter::Rewrite(std::move(body));

  if (prepass.sync_count() != 0) {
    const auto &sync_thread_counts = prepass.partial_sync_thread_counts();
    std::vector<std::pair<int, PrimExpr>> barrier_inits;
    barrier_inits.reserve(sync_thread_counts.size());
    for (size_t offset = 0; offset < sync_thread_counts.size(); ++offset) {
      barrier_inits.push_back({base_count + static_cast<int>(offset) + 1,
                               sync_thread_counts[offset]});
    }

    MbarrierSyncRewriter rewriter(base_count, std::move(barrier_inits));
    body = rewriter(std::move(body));
    auto cond = Call(DataType::Bool(), tl_shuffle_elect(),
                     {IntImm(DataType::Int(32), 0)});
    auto init_stmts = rewriter.MakeInitStmts();
    auto seq = init_stmts.size() == 1 ? init_stmts[0] : SeqStmt(init_stmts);
    prefix.push_back(IfThenElse(cond, seq));
    Stmt mem_sync =
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")}));
    prefix.push_back(mem_sync);
    prefix.push_back(body);
    body = SeqStmt(prefix);
  }

  body = CreateBarrierRewriter::Rewrite(std::move(body),
                                        base_count + prepass.sync_count());
  body = PartialBarrierSyncPhaseRewriter().Rewrite(std::move(body));

  n->body = std::move(body);
  return f;
}

namespace transform {

tvm::transform::Pass UnifiedBarrier() {
  auto pass_func = [](PrimFunc f, IRModule m,
                      const tvm::transform::PassContext &ctx) {
    if (!f->HasNonzeroAttr(tirx::attr::kIsGlobalFunc)) {
      return f;
    }
    return RewriteUnifiedBarrier(std::move(f));
  };
  return tirx::transform::CreatePrimFuncPass(pass_func, 0, "tl.UnifiedBarrier",
                                             {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.UnifiedBarrier", UnifiedBarrier);
}

} // namespace transform
} // namespace tl
} // namespace tvm
