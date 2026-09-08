/*!
 * \file tl/musa/transform/lower_partial_thread_sync.cc
 * \brief Lower partial shared-memory thread synchronization to MP31 barriers.
 */

#include "musa/op/builtin.h"

#include "op/utils.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <map>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

class PartialSyncCollector : public StmtExprVisitor {
public:
  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::create_barriers())) {
      ICHECK_EQ(op->args.size(), 1U);
      base_count_ = Downcast<IntImm>(op->args[0])->value;
    } else if (op->op.same_as(builtin::tvm_storage_sync()) &&
               op->args.size() == 3U) {
      const auto *barrier_id = op->args[1].as<IntImmNode>();
      ICHECK(barrier_id)
          << "MUSA partial storage sync requires a constant barrier id";
      partial_counts_.emplace(static_cast<int>(barrier_id->value), op->args[2]);
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  int base_count() const { return base_count_; }
  const std::map<int, PrimExpr> &partial_counts() const {
    return partial_counts_;
  }

private:
  int base_count_{-1};
  std::map<int, PrimExpr> partial_counts_;
};

class PartialSyncLowerer : public StmtExprMutator {
public:
  PartialSyncLowerer(int base_count, std::map<int, PrimExpr> partial_counts)
      : base_count_(base_count), partial_counts_(std::move(partial_counts)) {}

  Stmt Rewrite(Stmt body) {
    body = VisitStmt(std::move(body));
    ffi::Map<ffi::String, ffi::Any> annotations = {
        {attr::kLocalVarInit, Integer(0)},
    };
    for (auto it = phase_vars_.rbegin(); it != phase_vars_.rend(); ++it) {
      body = SeqStmt(ffi::Array<Stmt>{AllocBuffer(*it, annotations), body});
    }
    return body;
  }

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::create_barriers())) {
      int max_partial_id =
          partial_counts_.empty() ? 0 : partial_counts_.rbegin()->first;
      return Call(op->dtype, op->op, {Integer(base_count_ + max_partial_id)});
    }
    return StmtExprMutator::VisitExpr_(op);
  }

  Stmt VisitStmt_(const EvaluateNode *op) final {
    const auto *call = op->value.as<CallNode>();
    if (call == nullptr) {
      return StmtExprMutator::VisitStmt_(op);
    }

    if (call->op.same_as(builtin::create_barriers())) {
      Stmt create = Evaluate(VisitExpr(op->value));
      ffi::Array<Stmt> seq{create};
      for (const auto &[partial_id, thread_count] : partial_counts_) {
        PrimExpr barrier = Integer(base_count_ + partial_id);
        seq.push_back(Evaluate(Call(DataType::Handle(),
                                    builtin::ptx_init_barrier_thread_count(),
                                    {barrier, thread_count})));
      }
      return SeqStmt(seq);
    }

    if (!call->op.same_as(builtin::tvm_storage_sync()) ||
        call->args.size() != 3U) {
      return StmtExprMutator::VisitStmt_(op);
    }

    int partial_id = Downcast<IntImm>(call->args[1])->value;
    PrimExpr barrier = Integer(base_count_ + partial_id);
    Buffer phase = GetOrCreatePhaseVar(partial_id);
    ffi::Array<PrimExpr> index{Integer(0)};
    PrimExpr parity = BufferLoad(phase, index);
    Stmt fence = Evaluate(Call(DataType::Int(32), builtin::call_extern(),
                               {StringImm("__threadfence_block")}));
    Stmt arrive = Evaluate(
        Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {barrier}));
    Stmt wait = Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(), {barrier, parity}));
    Stmt toggle = BufferStore(phase, bitwise_xor(parity, Integer(1)), index);
    return SeqStmt::Flatten(fence, arrive, wait, toggle);
  }

  Buffer GetOrCreatePhaseVar(int partial_id) {
    auto it = phase_vars_by_id_.find(partial_id);
    if (it != phase_vars_by_id_.end()) {
      return it->second;
    }
    Buffer buffer = decl_buffer({Integer(1)}, DataType::Int(32),
                                "__musa_partial_barrier_phase_" +
                                    std::to_string(phase_vars_.size()),
                                "local.var");
    phase_vars_.push_back(buffer);
    phase_vars_by_id_.emplace(partial_id, buffer);
    return buffer;
  }

  int base_count_;
  std::map<int, PrimExpr> partial_counts_;
  std::map<int, Buffer> phase_vars_by_id_;
  std::vector<Buffer> phase_vars_;
};

PrimFunc LowerPartialThreadSync(PrimFunc func) {
  PartialSyncCollector collector;
  collector(func->body);
  if (collector.partial_counts().empty()) {
    return func;
  }
  ICHECK_GE(collector.base_count(), 0)
      << "MUSA partial thread sync requires an existing TME barrier pool";
  auto *node = func.CopyOnWrite();
  node->body =
      PartialSyncLowerer(collector.base_count(), collector.partial_counts())
          .Rewrite(std::move(node->body));
  return func;
}

namespace transform {

tvm::transform::Pass LowerPartialThreadSync() {
  auto pass_func = [](PrimFunc func, IRModule mod,
                      const tvm::transform::PassContext &ctx) {
    if (!func->HasNonzeroAttr(tirx::attr::kIsGlobalFunc)) {
      return func;
    }
    return tl::LowerPartialThreadSync(std::move(func));
  };
  return tirx::transform::CreatePrimFuncPass(
      pass_func, 0, "tl.musa.LowerPartialThreadSync", {});
}

} // namespace transform

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.musa.transform.LowerPartialThreadSync",
                        transform::LowerPartialThreadSync);
}

} // namespace tl
} // namespace tvm
