/*!
 * \file lower PH intrin.cc
 * \brief Lower PH intrinsics Mthread GPU(mp31+)
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <vector>

#include "../op/builtin.h"
#include "../runtime/runtime.h"

namespace tvm {
namespace tl {

using namespace tirx;

class LowerPHIntrin : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc &f, bool disable_shuffle_elect,
                             bool enable_tma_desc_prefetch) {
    PrimFuncNode *fptr = f.CopyOnWrite();
    LowerPHIntrin substituter(disable_shuffle_elect, enable_tma_desc_prefetch);
    fptr->body = substituter.VisitStmt(f->body);
    // Collect prologue/epilogue statements for host-side setup/teardown
    Array<Stmt> prologue_stmts;
    Array<Stmt> epilogue_stmts;
    for (const auto &desc_init : substituter.desc_inits_) {
      if (!desc_init.emitted) {
        prologue_stmts.push_back(desc_init.stmt);
      }
    }
    f = WithAttr(std::move(f), "tma_descriptor_args",
                 substituter.init_desc_arg_map_);

    // Additionally, if L2 persistent cache annotations were lowered earlier,
    // materialize TVM FFI calls to set the stream access policy window.
    if (f->attrs.defined() && f->attrs->dict.count("l2_persistent_map")) {
      auto l2_map =
          f->GetAttr<Map<String, Array<PrimExpr>>>("l2_persistent_map");
      if (l2_map.defined()) {
        // Build a lookup from buffer name to Buffer object
        std::unordered_map<std::string, Buffer> name2buf;
        for (const auto &kv : f->buffer_map) {
          name2buf.emplace(kv.second->name, kv.second);
        }
        for (const auto &kv : l2_map.value()) {
          const std::string buf_name = kv.first;
          const Array<PrimExpr> &args = kv.second;
          if (name2buf.count(buf_name) == 0) {
            continue;
          }
          const Buffer &buf = name2buf.at(buf_name);
          PrimExpr base_ptr = buf->data;
          if (buf->elem_offset.defined() && !is_zero(buf->elem_offset)) {
            PrimExpr byte_offset =
                buf->elem_offset *
                IntImm(buf->elem_offset.dtype(), buf->dtype.bytes());
            base_ptr =
                Call(DataType::Handle(), builtin::handle_add_byte_offset(),
                     {base_ptr, byte_offset});
          }
          Array<PrimExpr> packed_args;
          packed_args.push_back(
              StringImm(tvm_musa_stream_set_access_policy_window));
          packed_args.push_back(base_ptr);
          ICHECK_GE(args.size(), 2);
          packed_args.push_back(args[1]);
          packed_args.push_back(args[0]);
          prologue_stmts.push_back(Evaluate(Call(
              DataType::Int(32), builtin::tvm_call_packed(), packed_args)));
        }
        Array<PrimExpr> reset_args;
        reset_args.push_back(
            StringImm(tvm_musa_stream_reset_access_policy_window));
        epilogue_stmts.push_back(Evaluate(
            Call(DataType::Int(32), builtin::tvm_call_packed(), reset_args)));
      }
    }

    // Stitch prologue statements before the original body
    if (!prologue_stmts.empty()) {
      // Chain the Let/Evaluate statements sequentially
      Stmt seq = prologue_stmts.size() == 1 ? prologue_stmts[0]
                                            : SeqStmt(prologue_stmts);
      fptr->body = SeqStmt({seq, fptr->body});
    }
    if (!epilogue_stmts.empty()) {
      Stmt seq_end = epilogue_stmts.size() == 1 ? epilogue_stmts[0]
                                                : SeqStmt(epilogue_stmts);
      fptr->body = SeqStmt({fptr->body, seq_end});
    }
    return f;
  }

  Stmt VisitStmt_(const AllocBufferNode *op) final {
    Stmt stmt = StmtExprMutator::VisitStmt_(op);
    Array<Stmt> init_stmts;
    for (auto &desc_init : desc_inits_) {
      if (!desc_init.emitted && desc_init.base_var == op->buffer->data.get()) {
        init_stmts.push_back(desc_init.stmt);
        desc_init.emitted = true;
      }
    }
    if (init_stmts.empty()) {
      return stmt;
    }

    Array<Stmt> seq;
    seq.push_back(stmt);
    for (const auto &init_stmt : init_stmts) {
      seq.push_back(init_stmt);
    }
    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key != tirx::attr::thread_extent) {
      return StmtExprMutator::VisitStmt_(op);
    }

    IterVar iv = Downcast<IterVar>(op->node);
    if (iv->thread_tag != "threadIdx.x") {
      return StmtExprMutator::VisitStmt_(op);
    }

    Stmt body = StmtExprMutator::VisitStmt(op->body);
    if (prefetch_calls_.empty()) {
      return AttrStmt(op->node, op->attr_key, op->value, body);
    }

    Array<Stmt> prefetch_calls = prefetch_calls_;
    PrimExpr condition;
    if (disable_shuffle_elect_) {
      condition = EQ(iv->var, 0);
    } else {
      condition = Call(DataType::Bool(), tl_shuffle_elect(), {0});
    }
    Stmt prefetch_stmt = IfThenElse(condition, prefetch_calls.size() > 1
                                                   ? SeqStmt(prefetch_calls)
                                                   : prefetch_calls[0]);
    prefetch_calls_.clear();
    return AttrStmt(op->node, op->attr_key, op->value,
                    SeqStmt({prefetch_stmt, body}));
  }

  PrimExpr VisitExpr_(const CallNode *call) final {
    bool is_tma_descriptor = call->op.same_as(create_tma_descriptor());
    bool is_tma_im2col_descriptor =
        call->op.same_as(create_tma_im2col_descriptor());
    if (!is_tma_descriptor && !is_tma_im2col_descriptor) {
      return StmtExprMutator::VisitExpr_(call);
    }

    Call call_ref = tvm::ffi::GetRef<Call>(call);
    auto iter = desc_map_.find(call_ref);
    if (iter != desc_map_.end()) {
      return iter->second;
    }

    String name = call->args[2].as<Var>().value()->name_hint;
    int desc_index = static_cast<int>(desc_map_.size());
    Var var = Var(name + "_desc_" + std::to_string(desc_index),
                  PointerType(PrimType(cuTensorMapType()), "grid_constant"));
    desc_map_[call_ref] = var;
    Array<PrimExpr> init_desc_args = MakeInitDescArgs(call_ref, var);
    init_desc_arg_map_.Set(var, init_desc_args);
    desc_inits_.push_back({call->args[2].as<Var>().value().get(),
                           MakeInitDescStmt(var, init_desc_args), false});
    if (enable_tma_desc_prefetch_ && is_tma_descriptor) {
      prefetch_calls_.push_back(
          Evaluate(Call(DataType::Handle(), builtin::call_extern(),
                        {StringImm("tl::prefetch_tma_descriptor"), var})));
    }
    return var;
  }

private:
  struct DescInit {
    const VarNode *base_var;
    Stmt stmt;
    bool emitted;
  };

  static Array<PrimExpr> MakeInitDescArgs(const Call &call, const Var &var) {
    Array<PrimExpr> init_desc_args;
    if (call->op.same_as(create_tma_descriptor())) {
      init_desc_args.push_back(StringImm(tvm_tensormap_create_tiled));
    } else if (call->op.same_as(create_tma_im2col_descriptor())) {
      init_desc_args.push_back(StringImm(tvm_tensormap_create_im2col));
    } else {
      ICHECK(0) << call->op;
    }
    init_desc_args.push_back(var);
    init_desc_args.insert(init_desc_args.end(), call->args.begin(),
                          call->args.end());
    return init_desc_args;
  }

  static Stmt MakeInitDescStmt(const Var &var,
                               const Array<PrimExpr> &init_desc_args) {
    // Should allocate 128 bytes for TensorMap on stack.
    Call alloc_desc = Call(DataType::Handle(), builtin::tvm_stack_alloca(),
                           {StringImm("tvm_ffi_any"), 16});
    Call init_desc =
        Call(DataType::Handle(), builtin::tvm_call_packed(), init_desc_args);
    return SeqStmt({Bind(var, alloc_desc), Evaluate(init_desc)});
  }

  Array<Stmt> prefetch_calls_;
  std::unordered_map<Call, Var, StructuralHash, ExprDeepEqual> desc_map_;
  std::vector<DescInit> desc_inits_;
  Map<Var, Array<PrimExpr>> init_desc_arg_map_;
  LowerPHIntrin(bool disable_shuffle_elect, bool enable_tma_desc_prefetch)
      : disable_shuffle_elect_(disable_shuffle_elect),
        enable_tma_desc_prefetch_(enable_tma_desc_prefetch) {}
  bool disable_shuffle_elect_;
  bool enable_tma_desc_prefetch_;
};

using namespace tirx::transform;

tvm::transform::Pass LowerPHIntrin() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool disable_shuffle_elect =
        ctx->GetConfig<Bool>(kDisableShuffleElect, Bool(false)).value();
    bool enable_tma_desc_prefetch =
        ctx->GetConfig<Bool>(kEnableMusaTmaPrefetch, Bool(false)).value();
    return LowerPHIntrin::Substitute(f, disable_shuffle_elect,
                                     enable_tma_desc_prefetch);
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerPHIntrin", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerPHIntrin", LowerPHIntrin);
}

} // namespace tl
} // namespace tvm
