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
#include <set>
#include <string>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

struct ParsedAllReduceCall {
  std::string reduce_type;
  bool is_batch{false};
  bool valid{false};
};

// Detect multi-warp tl::AllReduce / tl::WarpFirstAllReduce extern calls that
// take a shared-memory workspace (args.size() >= 3) and therefore invoke a
// barrier internally. Inside a warp-specialized consumer branch `if (tx < tc)`,
// using SyncThreadsBarrier (__syncthreads) deadlocks because producer warps
// never enter the consumer branch.
ParsedAllReduceCall ParseMultiWarpAllReduceCall(const CallNode *op) {
  ParsedAllReduceCall info;
  if (!op->op.same_as(builtin::call_extern()) || op->args.size() < 3U) {
    return info;
  }
  const auto *name = op->args[0].as<StringImmNode>();
  if (name == nullptr) {
    return info;
  }
  const std::string &sym = name->value;
  if (sym.rfind("tl::AllReduce<", 0) != 0 &&
      sym.rfind("tl::WarpFirstAllReduce<", 0) != 0) {
    return info;
  }
  constexpr const char *kRunSuffix = ">::run";
  constexpr const char *kRunBatchSuffix = ">::run_batch";
  const size_t run_len = std::char_traits<char>::length(kRunSuffix);
  const size_t batch_len = std::char_traits<char>::length(kRunBatchSuffix);
  if (sym.size() > batch_len &&
      sym.compare(sym.size() - batch_len, batch_len, kRunBatchSuffix) == 0) {
    info.reduce_type = sym.substr(0, sym.size() - batch_len + 1);
    info.is_batch = true;
    info.valid = true;
  } else if (sym.size() > run_len &&
             sym.compare(sym.size() - run_len, run_len, kRunSuffix) == 0) {
    info.reduce_type = sym.substr(0, sym.size() - run_len + 1);
    info.is_batch = false;
    info.valid = true;
  }
  return info;
}

} // namespace

class PartialSyncCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      if (const auto *iv = op->node.as<IterVarNode>()) {
        if (iv->thread_tag == "threadIdx.x") {
          tx_var_ = iv->var;
        }
      }
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const IfThenElseNode *op) final {
    const auto *lt = op->condition.as<LTNode>();
    bool matched_consumer = false;
    if (lt && tx_var_.defined() && lt->a.same_as(tx_var_)) {
      if (const auto *bound = lt->b.as<IntImmNode>()) {
        consumer_stack_.push_back(static_cast<int>(bound->value));
        matched_consumer = true;
      }
    }
    VisitStmt(op->then_case);
    if (matched_consumer) {
      consumer_stack_.pop_back();
    }
    if (op->else_case.defined()) {
      VisitStmt(op->else_case.value());
    }
  }

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
    } else if (!consumer_stack_.empty()) {
      ParsedAllReduceCall red = ParseMultiWarpAllReduceCall(op);
      if (red.valid) {
        reduce_consumer_threads_.insert(consumer_stack_.back());
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  int base_count() const { return base_count_; }
  const Var &tx_var() const { return tx_var_; }
  const std::map<int, PrimExpr> &partial_counts() const {
    return partial_counts_;
  }
  const std::set<int> &reduce_consumer_threads() const {
    return reduce_consumer_threads_;
  }

private:
  int base_count_{-1};
  Var tx_var_;
  std::vector<int> consumer_stack_;
  std::map<int, PrimExpr> partial_counts_;
  std::set<int> reduce_consumer_threads_;
};

class PartialSyncLowerer : public StmtExprMutator {
public:
  PartialSyncLowerer(int base_count, int total_barriers, Var tx_var,
                     std::map<int, PrimExpr> partial_counts,
                     std::map<int, int> reduce_barriers)
      : base_count_(base_count),
        total_barriers_(total_barriers),
        tx_var_(std::move(tx_var)),
        partial_counts_(std::move(partial_counts)),
        reduce_barriers_(std::move(reduce_barriers)) {}

  Stmt Rewrite(Stmt body) {
    body = VisitStmt(std::move(body));
    for (auto it = phase_vars_.rbegin(); it != phase_vars_.rend(); ++it) {
      Stmt init = BufferStore(*it, Integer(0), {Integer(0)});
      body = SeqStmt(ffi::Array<Stmt>{AllocBuffer(*it), init, body});
    }
    return body;
  }

private:
  Stmt VisitStmt_(const IfThenElseNode *op) final {
    const auto *lt = op->condition.as<LTNode>();
    bool matched_consumer = false;
    if (lt && tx_var_.defined() && lt->a.same_as(tx_var_)) {
      if (const auto *bound = lt->b.as<IntImmNode>()) {
        consumer_stack_.push_back(static_cast<int>(bound->value));
        matched_consumer = true;
      }
    }
    PrimExpr cond = VisitExpr(op->condition);
    Stmt then_case = VisitStmt(op->then_case);
    if (matched_consumer) {
      consumer_stack_.pop_back();
    }
    ffi::Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      else_case = VisitStmt(op->else_case.value());
    }
    if (cond.same_as(op->condition) && then_case.same_as(op->then_case) &&
        else_case.same_as(op->else_case)) {
      return ffi::GetRef<Stmt>(op);
    }
    return IfThenElse(cond, then_case, else_case);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::create_barriers())) {
      return Call(op->dtype, op->op, {Integer(total_barriers_)});
    }
    if (!consumer_stack_.empty()) {
      ParsedAllReduceCall red = ParseMultiWarpAllReduceCall(op);
      if (red.valid) {
        auto it = reduce_barriers_.find(consumer_stack_.back());
        if (it != reduce_barriers_.end()) {
          int bar_id = it->second;
          Buffer phase = GetOrCreateReducePhaseVar(consumer_stack_.back());
          PrimExpr phase_ptr = Call(DataType::Handle(), builtin::address_of(),
                                    {BufferLoad(phase, {Integer(0)})});
          std::string fn =
              (red.is_batch ? "tl::mp31_allreduce_batch_partial<"
                            : "tl::mp31_allreduce_partial<") +
              red.reduce_type + " >";
          ffi::Array<PrimExpr> new_args;
          new_args.push_back(StringImm(fn));
          for (size_t i = 1; i < op->args.size(); ++i) {
            new_args.push_back(VisitExpr(op->args[i]));
          }
          new_args.push_back(Integer(bar_id));
          new_args.push_back(phase_ptr);
          return Call(op->dtype, op->op, new_args);
        }
      }
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
      for (const auto &[tc, bar_id] : reduce_barriers_) {
        seq.push_back(Evaluate(Call(DataType::Handle(),
                                    builtin::ptx_init_barrier_thread_count(),
                                    {Integer(bar_id), Integer(tc)})));
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

  Buffer GetOrCreateReducePhaseVar(int consumer_threads) {
    auto it = reduce_phase_vars_by_tc_.find(consumer_threads);
    if (it != reduce_phase_vars_by_tc_.end()) {
      return it->second;
    }
    Buffer buffer = decl_buffer({Integer(1)}, DataType::Int(32),
                                "__musa_partial_barrier_phase_" +
                                    std::to_string(phase_vars_.size()),
                                "local.var");
    phase_vars_.push_back(buffer);
    reduce_phase_vars_by_tc_.emplace(consumer_threads, buffer);
    return buffer;
  }

  int base_count_;
  int total_barriers_;
  Var tx_var_;
  std::map<int, PrimExpr> partial_counts_;
  std::map<int, int> reduce_barriers_;
  std::vector<int> consumer_stack_;
  std::map<int, Buffer> phase_vars_by_id_;
  std::map<int, Buffer> reduce_phase_vars_by_tc_;
  std::vector<Buffer> phase_vars_;
};

PrimFunc LowerPartialThreadSync(PrimFunc func) {
  PartialSyncCollector collector;
  collector(func->body);
  const bool has_partial_sync = !collector.partial_counts().empty();
  const bool has_ws_reduce =
      collector.base_count() >= 0 && !collector.reduce_consumer_threads().empty();
  if (!has_partial_sync && !has_ws_reduce) {
    return func;
  }
  ICHECK_GE(collector.base_count(), 0)
      << "MUSA partial thread sync requires an existing TME barrier pool";
  int max_partial_id = collector.partial_counts().empty()
                           ? 0
                           : collector.partial_counts().rbegin()->first;
  int next_barrier = collector.base_count() + max_partial_id + 1;
  std::map<int, int> reduce_barriers;
  for (int tc : collector.reduce_consumer_threads()) {
    reduce_barriers.emplace(tc, next_barrier++);
  }
  int total_barriers = next_barrier - 1;
  auto *node = func.CopyOnWrite();
  node->body = PartialSyncLowerer(collector.base_count(), total_barriers,
                                  collector.tx_var(), collector.partial_counts(),
                                  std::move(reduce_barriers))
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
