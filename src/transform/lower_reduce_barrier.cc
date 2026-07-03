/*!
 *  \file lower_reduce_barrier.cc
 *  \brief Lower shared.reduce_barrier buffers after warp specialization.
 */
#include "../op/builtin.h"
#include "support/check.h"
#include "tvm/ir/type.h"
#include "tvm/tirx/expr.h"
#include "tvm/tirx/stmt.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <utility>

namespace tvm {
namespace tl {

using namespace tirx;

class ReduceBarrierRewriter : public StmtExprMutator {
public:
  static Stmt Rewrite(Stmt body, bool disable_shuffle_elect = false) {
    ReduceBarrierRewriter rewriter(disable_shuffle_elect);
    return rewriter(std::move(body));
  }

private:
  struct BarrierVarPair {
    Var front;
    Var back;
  };

  explicit ReduceBarrierRewriter(bool disable_shuffle_elect)
      : disable_shuffle_elect_(disable_shuffle_elect) {}

  static bool IsReduceBarrierBuffer(const Buffer &buffer) {
    const auto *ptr_type = buffer->data->type_annotation.as<PointerTypeNode>();
    if (!ptr_type) {
      return false;
    }
    return ptr_type->storage_scope == "shared.reduce_barrier";
  }

  BarrierVarPair CreateBarrierVarPair(const Buffer &buffer) {
    int pair_idx = num_barrier_pairs_++;
    Var front_var(buffer->name + "_0_" + std::to_string(pair_idx),
                  DataType::Int(32));
    Var back_var(buffer->name + "_1_" + std::to_string(pair_idx),
                 DataType::Int(32));
    return BarrierVarPair{front_var, back_var};
  }

  void AddInitCalls(Array<Stmt> *init_calls, const BarrierVarPair &pair,
                    PrimExpr arrive_count) {
    auto front_call =
        Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
             {pair.front, arrive_count});
    auto back_call =
        Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
             {pair.back, arrive_count});
    init_calls->push_back(Evaluate(front_call));
    init_calls->push_back(Evaluate(back_call));
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    SBlock block = tvm::ffi::GetRef<SBlock>(op);
    Array<Buffer> alloc_buffers = op->alloc_buffers;

    // Rewrite reduce barrier buffers allocated in this block.
    Map<Buffer, PrimExpr> local_expr_remap;
    Array<Var> barrier_id_vars;
    Array<Stmt> init_calls;
    Array<Buffer> filtered;
    filtered.reserve(alloc_buffers.size());
    for (auto buf : alloc_buffers) {
      if (!IsReduceBarrierBuffer(buf)) {
        filtered.push_back(buf);
        continue;
      }
      BarrierVarPair pair = CreateBarrierVarPair(buf);
      local_expr_remap.Set(buf, pair.front);
      // Keep front/back adjacent so placeholder rewrite assigns consecutive
      // ids.
      barrier_id_vars.push_back(pair.front);
      barrier_id_vars.push_back(pair.back);
      AddInitCalls(&init_calls, pair, buf->shape[0]);
    }

    // No reduce barriers in this block.
    if (local_expr_remap.size() == 0) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // Remove alloc_buffer("shared.reduce_barrier").
    if (!filtered.same_as(op->alloc_buffers)) {
      block.CopyOnWrite()->alloc_buffers = filtered;
    }

    PrimExpr condition;
    if (!disable_shuffle_elect_) {
      condition = Call(DataType::Bool(), tl_shuffle_elect(), {0});
    } else {
      ICHECK(thread_var_.defined()) << "thread_var_ is not defined";
      condition = EQ(thread_var_->var, 0);
    }
    Stmt init_stmt =
        init_calls.size() == 1 ? init_calls[0] : SeqStmt(init_calls);

    Array<Stmt> new_body;
    new_body.push_back(IfThenElse(condition, init_stmt, Stmt()));
    new_body.push_back(
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")})));
    new_body.push_back(block->body);
    Array<Stmt> block_body;
    block_body.reserve(barrier_id_vars.size() + 1);
    for (int i = 0; i < static_cast<int>(barrier_id_vars.size()); ++i) {
      PrimExpr placeholder =
          Call(DataType::Int(32), barrier_id_placeholder(), {});
      block_body.push_back(Bind(barrier_id_vars[i], placeholder));
    }
    block_body.push_back(SeqStmt(new_body));
    Stmt new_block_body = SeqStmt(block_body);

    block.CopyOnWrite()->body = new_block_body;

    // Update this block's BufferLoad/BufferStore and visit nested statements.
    buffer_expr_remap_stack_.push_back(local_expr_remap);
    Stmt updated = StmtExprMutator::VisitStmt_(block.get());
    buffer_expr_remap_stack_.pop_back();
    return updated;
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tirx::attr::thread_extent) {
      IterVar iv = Downcast<IterVar>(op->node);
      if (iv->thread_tag == "threadIdx.x") {
        thread_var_ = iv;
      }
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  // find reduce_barrier[] and rewrite to reduce_barrier
  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto load = Downcast<BufferLoad>(StmtExprMutator::VisitExpr_(op));
    auto buffer = load->buffer;
    for (auto it = buffer_expr_remap_stack_.rbegin();
         it != buffer_expr_remap_stack_.rend(); ++it) {
      if (it->count(buffer)) {
        return it->at(buffer);
      }
    }
    return load;
  }

  // find reduce_barrier[] stats and assert
  Stmt VisitStmt_(const BufferStoreNode *op) final {
    auto store = Downcast<BufferStore>(StmtExprMutator::VisitStmt_(op));
    auto buffer = store->buffer;
    for (auto it = buffer_expr_remap_stack_.rbegin();
         it != buffer_expr_remap_stack_.rend(); ++it) {
      if (it->count(buffer)) {
        ICHECK(false) << "Storing to a reduce barrier var is not supported.";
      }
    }
    return store;
  }

  // Scope-local remaps for BufferLoad/BufferStore rewriting.
  // Push/pop per block to avoid cross-block leakage.
  std::vector<Map<Buffer, PrimExpr>> buffer_expr_remap_stack_;

  // Used when disable_shuffle_elect_ is true.
  IterVar thread_var_;
  bool disable_shuffle_elect_;
  int num_barrier_pairs_{0};
};

PrimFunc LowerReduceBarrier(PrimFunc f, bool disable_shuffle_elect) {
  f.CopyOnWrite()->body =
      ReduceBarrierRewriter::Rewrite(std::move(f->body), disable_shuffle_elect);
  return f;
}

namespace transform {

using namespace tirx::transform;

tvm::transform::Pass LowerReduceBarrier() {
  auto pass_func = [](PrimFunc f, IRModule m,
                      const tvm::transform::PassContext &ctx) {
    bool disable_shuffle_elect =
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value();
    return tl::LowerReduceBarrier(std::move(f), disable_shuffle_elect);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerReduceBarrier", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerReduceBarrier", LowerReduceBarrier);
}

} // namespace transform
} // namespace tl
} // namespace tvm
