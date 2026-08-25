/* Lower MP31 TME descriptors and barrier buffers to device/host ABI values. */

#include <tvm/ffi/extra/structural_equal.h>
#include <tvm/ffi/extra/structural_hash.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <unordered_map>

#include "musa/runtime.h"
#include "musa/target_utils.h"
#include "musa/op/builtin.h"
#include "op/builtin.h"
#include "support/check.h"

namespace tvm {
namespace tl {
namespace musa {

using namespace tirx;
using namespace ffi;

class LowerTMEIntrinPass : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    LowerTMEIntrinPass pass;
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = pass.VisitStmt(f->body);

    if (!pass.desc_init_args_.empty()) {
      f = WithAttr(std::move(f), "tma_descriptor_args", pass.desc_init_args_);
      Array<Stmt> prologue;
      for (const auto &entry : pass.desc_inits_) {
        Call alloc = Call(DataType::Handle(), builtin::tvm_stack_alloca(),
                          {StringImm("tvm_ffi_any"), 16});
        Call init =
            Call(DataType::Handle(), builtin::tvm_call_packed(), entry.second);
        prologue.push_back(tirx::Bind(entry.first, alloc));
        prologue.push_back(Evaluate(init));
      }
      fptr = f.CopyOnWrite();
      prologue.push_back(fptr->body);
      fptr->body = SeqStmt(prologue);
    }
    return f;
  }

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(create_tma_descriptor())) {
      return StmtExprMutator::VisitExpr_(op);
    }

    Call call = GetRef<Call>(op);
    auto it = descriptor_map_.find(call);
    if (it != descriptor_map_.end()) {
      return it->second;
    }

    ICHECK_GE(op->args.size(), 3U);
    const auto *base = op->args[2].as<VarNode>();
    ICHECK(base) << "MP31 TME descriptor global address must be a buffer var";
    Var descriptor(base->name_hint + "_desc",
                   PointerType(TensorMapType(), "grid_constant"));
    descriptor_map_.emplace(call, descriptor);

    Array<PrimExpr> init_args;
    init_args.push_back(StringImm(tvm_musa_tensordesc_create_tiled));
    init_args.push_back(descriptor);
    init_args.insert(init_args.end(), op->args.begin(), op->args.end());
    desc_init_args_.Set(descriptor, init_args);
    desc_inits_.push_back({descriptor, init_args});
    return descriptor;
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    Array<Buffer> barrier_buffers;
    for (auto buffer : op->alloc_buffers) {
      if (buffer.scope() == "shared.barrier" ||
          buffer.scope() == "shared.cluster_barrier") {
        barrier_buffers.push_back(buffer);
      }
    }
    if (barrier_buffers.empty()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    // MTCC MP31 reserves async barrier id 0.
    int base = 1;
    for (auto buffer : barrier_buffers) {
      ICHECK(buffer->shape.size() == 1);
      auto extent = buffer->shape[0].as<IntImmNode>();
      ICHECK(extent) << "MP31 TME barriers require a static allocation size";
      barrier_base_.emplace(buffer->data.get(), base);
      base += static_cast<int>(extent->value);
    }

    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    auto block_ptr = block.CopyOnWrite();
    Array<Buffer> kept;
    for (auto buffer : block->alloc_buffers) {
      if (barrier_base_.count(buffer->data.get()) == 0) {
        kept.push_back(buffer);
      }
    }
    block_ptr->alloc_buffers = kept;

    Array<Stmt> init;
    init.push_back(Evaluate(
        Call(DataType::Handle(), builtin::create_barriers(), {base - 1})));
    for (auto buffer : barrier_buffers) {
      auto it = barrier_base_.find(buffer->data.get());
      auto ann = block->annotations.Get("barrier_init");
      if (!ann) {
        continue;
      }
      auto init_map = ann->as<Map<Var, Array<PrimExpr>>>();
      if (!init_map || !init_map.value().count(buffer->data)) {
        continue;
      }
      auto counts = init_map.value().at(buffer->data);
      for (size_t i = 0; i < counts.size(); ++i) {
        init.push_back(Evaluate(
            Call(DataType::Handle(), builtin::ptx_init_barrier_thread_count(),
                 {IntImm(DataType::Int(32), it->second + static_cast<int>(i)),
                  counts[i]})));
      }
    }
    // Match CUDA LowerSharedBarrier: no TME transaction may observe the
    // barrier before its leader-thread initialization is visible CTA-wide.
    init.push_back(
        Evaluate(Call(DataType::Handle(), builtin::tvm_storage_sync(),
                      {StringImm("shared")})));
    block_ptr->body = SeqStmt({SeqStmt(init), block->body});
    return block;
  }

  PrimExpr VisitExpr_(const BufferLoadNode *op) final {
    auto it = barrier_base_.find(op->buffer->data.get());
    if (it == barrier_base_.end()) {
      return StmtExprMutator::VisitExpr_(op);
    }
    ICHECK_EQ(op->indices.size(), 1U);
    PrimExpr index = op->indices[0];
    if (it->second != 0) {
      index += it->second;
    }
    return index;
  }

  std::unordered_map<Call, Var, StructuralHash, tirx::ExprDeepEqual>
      descriptor_map_;
  Map<Var, Array<PrimExpr>> desc_init_args_;
  std::vector<std::pair<Var, Array<PrimExpr>>> desc_inits_;
  std::unordered_map<const VarNode *, int> barrier_base_;
};

namespace transform {
using namespace tirx::transform;

tvm::transform::Pass LowerTMEIntrin() {
  auto pass_func = [=](PrimFunc f, const IRModule &, PassContext) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsMP31(target.value())) {
      return f;
    }
    return LowerTMEIntrinPass::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.musa.LowerTMEIntrin", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.musa.transform.LowerTMEIntrin", LowerTMEIntrin);
}

} // namespace transform
} // namespace musa
} // namespace tl
} // namespace tvm
