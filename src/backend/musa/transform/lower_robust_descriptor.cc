/*!
 * \file backend/musa/transform/lower_robust_descriptor.cc
 * \brief Hoist MUSA robust descriptors to host-created kernel parameters.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "op/builtin.h"
#include "runtime/runtime.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

class DescriptorBaseVarFinder : public ExprVisitor {
public:
  Optional<Var> Find(const PrimExpr &expr) {
    VisitExpr(expr);
    return result_;
  }

private:
  void VisitExpr_(const BufferLoadNode *op) final {
    if (!result_.defined()) {
      result_ = op->buffer->data;
    }
    ExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const VarNode *op) final {
    if (!result_.defined() && op->dtype.is_handle()) {
      result_ = ffi::GetRef<Var>(op);
    }
  }

  Optional<Var> result_;
};

class DescriptorVarCollector : public ExprVisitor {
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

class HostEvaluableDescriptorExprChecker : public ExprVisitor {
public:
  explicit HostEvaluableDescriptorExprChecker(
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
    // Reading a buffer value belongs to the device computation even when the
    // buffer itself is a kernel parameter.  Hoisting such a load would make
    // host code dereference device memory while constructing the descriptor.
    host_evaluable_ = false;
  }

  void VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::address_of())) {
      if (op->args.size() != 1U) {
        host_evaluable_ = false;
        return;
      }
      const auto *load = op->args[0].as<BufferLoadNode>();
      if (load == nullptr || !host_vars_.count(load->buffer->data.get())) {
        host_evaluable_ = false;
        return;
      }
      // address_of forms a pointer without reading the buffer value.  Its
      // indices must still be computable from host-side parameters.
      for (const PrimExpr &index : load->indices) {
        VisitExpr(index);
        if (!host_evaluable_) {
          return;
        }
      }
      return;
    }
    ExprVisitor::VisitExpr_(op);
  }

  const std::unordered_set<const VarNode *> &host_vars_;
  bool host_evaluable_{true};
};

PrimExpr MakeHostDescriptorBasePointer(const PrimExpr &base) {
  const auto *call = base.as<CallNode>();
  if (call == nullptr || !call->op.same_as(builtin::address_of()) ||
      call->args.size() != 1U) {
    return base;
  }

  const auto *load = call->args[0].as<BufferLoadNode>();
  if (load == nullptr) {
    return base;
  }

  // Keep device element types out of the generated C host stub.  Printing
  // address_of(BufferLoad) directly would introduce typed pointer casts such
  // as `half*`, while descriptor construction only needs an opaque address.
  Array<PrimExpr> physical_offsets = load->buffer.OffsetOf(load->indices);
  Buffer flattened_buffer = load->buffer.GetFlattenedBuffer();
  ICHECK_EQ(physical_offsets.size(), flattened_buffer->shape.size());
  ICHECK(!physical_offsets.empty());

  PrimExpr linear_offset = physical_offsets[0];
  for (size_t i = 1; i < physical_offsets.size(); ++i) {
    linear_offset =
        linear_offset * flattened_buffer->shape[i] + physical_offsets[i];
  }
  PrimExpr byte_offset = linear_offset * IntImm(linear_offset.dtype(),
                                                load->buffer->dtype.bytes());
  if (is_zero(byte_offset)) {
    return load->buffer->data;
  }
  return Call(DataType::Handle(), builtin::handle_add_byte_offset(),
              {load->buffer->data, byte_offset});
}

class MUSARobustDescriptorHoister : public StmtExprMutator {
public:
  static PrimFunc Substitute(PrimFunc f) {
    std::unordered_set<const VarNode *> host_vars;
    auto add_host_expr_vars = [&host_vars](const PrimExpr &expr) {
      for (const Var &var : DescriptorVarCollector().Collect(expr)) {
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
        add_host_expr_vars(shape);
      }
      for (const PrimExpr &stride : buffer->strides) {
        add_host_expr_vars(stride);
      }
      add_host_expr_vars(buffer->elem_offset);
    }

    MUSARobustDescriptorHoister hoister(std::move(host_vars));
    Stmt body = hoister.VisitStmt(f->body);

    Map<Var, Array<PrimExpr>> init_arg_map;
    Array<Stmt> prologue_stmts;
    for (const auto &[call, var] : hoister.descriptors_) {

      // One TVMFFIAny slot is 16 bytes and receives 64-byte stack alignment
      // from CodeGenCHost, sufficient for the 128-bit descriptor value.
      Call alloc_desc = Call(DataType::Handle(), builtin::tvm_stack_alloca(),
                             {StringImm("tvm_ffi_any"), 1});
      Array<PrimExpr> init_args{
          StringImm(tvm_musa_robust_descriptor_create), var,
          MakeHostDescriptorBasePointer(call->args[0]), call->args[1]};
      Call init_desc =
          Call(DataType::Handle(), builtin::tvm_call_packed(), init_args);
      prologue_stmts.push_back(
          SeqStmt({Bind(var, alloc_desc), Evaluate(init_desc)}));
      init_arg_map.Set(var, init_args);
    }
    if (!prologue_stmts.empty()) {
      prologue_stmts.push_back(body);
      body = SeqStmt(prologue_stmts);
    }

    auto *n = f.CopyOnWrite();
    n->body = std::move(body);
    return WithAttr(std::move(f), "robust_descriptor_args", init_arg_map);
  }

  PrimExpr VisitExpr_(const CallNode *op) final {
    if (!op->op.same_as(make_robust_desc())) {
      return StmtExprMutator::VisitExpr_(op);
    }

    ICHECK_EQ(op->args.size(), 2)
        << "tl.make_robust_desc expects (base_ptr, size_bytes)";
    Call call = ffi::GetRef<Call>(op);
    bool host_constructible =
        HostEvaluableDescriptorExprChecker(host_vars_).Check(call->args[0]) &&
        HostEvaluableDescriptorExprChecker(host_vars_).Check(call->args[1]);
    if (!host_constructible) {
      // Keep device-local descriptors as calls. MUSA codegen will use the
      // compatibility path that constructs the descriptor in the kernel.
      return call;
    }
    auto it = descriptor_map_.find(call);
    if (it != descriptor_map_.end()) {
      return it->second;
    }

    std::string prefix = "robust";
    if (auto base_var = DescriptorBaseVarFinder().Find(call->args[0])) {
      prefix = base_var.value()->name_hint;
    }
    Var var(prefix + "_robust_desc_" + std::to_string(descriptors_.size()),
            PointerType(PrimType(musaRobustDescriptorType()), "grid_constant"));
    descriptor_map_.emplace(call, var);
    descriptors_.emplace_back(call, var);
    return var;
  }

private:
  explicit MUSARobustDescriptorHoister(
      std::unordered_set<const VarNode *> host_vars)
      : host_vars_(std::move(host_vars)) {}

  std::unordered_set<const VarNode *> host_vars_;
  std::unordered_map<Call, Var, ffi::StructuralHash, ExprDeepEqual>
      descriptor_map_;
  std::vector<std::pair<Call, Var>> descriptors_;
};

} // namespace

using namespace tirx::transform;

tvm::transform::Pass LowerMUSARobustDescriptor() {
  auto pass_func = [](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return MUSARobustDescriptorHoister::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerMUSARobustDescriptor", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerMUSARobustDescriptor",
                        LowerMUSARobustDescriptor);
}

} // namespace tl
} // namespace tvm
