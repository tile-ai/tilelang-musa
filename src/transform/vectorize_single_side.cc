/*!
 * \file vectorize_single_side.cc
 * \brief Form Ramp-based vector IR for single-sided global memory accesses.
 *
 * This pass canonicalizes strict scalar copy loops into vector TIR without
 * generating target intrinsics.  LowerLDGSTG consumes the resulting Ramp-based
 * global accesses and lowers them to ldg/stg intrinsics.
 */

#include <array>
#include <vector>

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/expr.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "../op/utils.h"
#include "../target/utils.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

constexpr std::array<int, 4> kSupportedVectorBits = {256, 128, 64, 32};

bool IsZero(const PrimExpr &expr) {
  const int64_t *value = as_const_int(expr);
  return value != nullptr && *value == 0;
}

bool IsSupportedScalarDType(DataType dtype) {
  if (!dtype.is_scalar()) {
    return false;
  }
  if (dtype.is_bool() || dtype.bits() == 1) {
    return false;
  }
  return true;
}

std::vector<int> GetCandidateLanes(DataType dtype, int64_t extent) {
  std::vector<int> lanes;
  for (int total_bits : kSupportedVectorBits) {
    if (total_bits % dtype.bits() != 0) {
      continue;
    }
    int candidate = total_bits / dtype.bits();
    if (candidate <= 1 || extent % candidate != 0) {
      continue;
    }
    lanes.push_back(candidate);
  }
  return lanes;
}

PrimExpr MakeLaneValue(const Var &loop_var, const Var &group_var, int lanes,
                       int lane) {
  PrimExpr base = group_var * make_const(loop_var.dtype(), lanes);
  if (lane == 0) {
    return base;
  }
  return base + make_const(loop_var.dtype(), lane);
}

PrimExpr SubstituteLoopVar(const PrimExpr &expr, const Var &loop_var,
                           const Var &group_var, int lanes, int lane,
                           arith::Analyzer *analyzer) {
  PrimExpr value = Substitute(
      expr, {{loop_var, MakeLaneValue(loop_var, group_var, lanes, lane)}});
  return analyzer->Simplify(value);
}

PrimExpr MakeContiguousRamp(const PrimExpr &base, int lanes) {
  return Ramp(base, make_const(base.dtype(), 1), lanes);
}

struct AccessPattern {
  bool contiguous{false};
  bool aligned{false};
  PrimExpr base;
};

} // namespace

class VectorizeSingleSideRewriter : public StmtExprMutator {
public:
  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kSourceRobustDesc) {
      bool previous_in_source_robust_desc = in_source_robust_desc_;
      in_source_robust_desc_ = true;
      Stmt body = this->VisitStmt(op->body);
      in_source_robust_desc_ = previous_in_source_robust_desc;
      return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
    }
    if (op->attr_key == tl::attr::kForceAsyncCopy) {
      bool previous_in_force_async_copy = in_force_async_copy_;
      in_force_async_copy_ = true;
      Stmt body = this->VisitStmt(op->body);
      in_force_async_copy_ = previous_in_force_async_copy;
      return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
    }
    if (op->attr_key == tirx::attr::tilelang_assume) {
      PrimExpr constraint = Downcast<PrimExpr>(op->node);
      auto recovery = analyzer_.EnterConstraint(constraint, /*is_assume=*/true);
      Stmt body = this->VisitStmt(op->body);
      recovery();
      return AttrStmt(op->node, op->attr_key, op->value, body, op->span);
    }
    return StmtExprMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    if (ShouldSkipVectorize()) {
      return StmtExprMutator::VisitStmt_(op);
    }

    Optional<Stmt> vectorized =
        TryVectorizeScalarCopyLoop(ffi::GetRef<For>(op));
    if (vectorized.defined()) {
      return vectorized.value();
    }

    return StmtExprMutator::VisitStmt_(op);
  }

private:
  bool in_force_async_copy_{false};
  bool in_source_robust_desc_{false};
  arith::Analyzer analyzer_;

  bool ShouldSkipVectorize() const {
    return in_force_async_copy_ || in_source_robust_desc_;
  }

  AccessPattern AnalyzeAccess(const PrimExpr &index, const Var &loop_var,
                              const Var &group_var, int lanes) {
    AccessPattern result;
    result.base =
        SubstituteLoopVar(index, loop_var, group_var, lanes, 0, &analyzer_);
    PrimExpr lanes_expr = make_const(result.base.dtype(), lanes);
    PrimExpr zero = make_const(result.base.dtype(), 0);
    result.aligned =
        analyzer_.CanProveEqual(FloorMod(result.base, lanes_expr), zero);
    result.contiguous = true;
    for (int lane = 1; lane < lanes; ++lane) {
      PrimExpr lane_index = SubstituteLoopVar(index, loop_var, group_var, lanes,
                                              lane, &analyzer_);
      PrimExpr expected = analyzer_.Simplify(
          result.base + make_const(result.base.dtype(), lane));
      if (!analyzer_.CanProveEqual(lane_index, expected)) {
        result.contiguous = false;
        break;
      }
    }
    return result;
  }

  PrimExpr MakeVectorFromScalarLoads(const BufferLoadNode *load,
                                     const Var &loop_var, const Var &group_var,
                                     int lanes) {
    ffi::Array<PrimExpr> values;
    for (int lane = 0; lane < lanes; ++lane) {
      PrimExpr src_index = SubstituteLoopVar(
          load->indices[0], loop_var, group_var, lanes, lane, &analyzer_);
      values.push_back(BufferLoad(load->buffer, {src_index}));
    }
    return Shuffle::Concat(values);
  }

  Stmt MakeScalarStoresFromVector(const BufferStoreNode *store,
                                  const PrimExpr &vector_value,
                                  const Var &loop_var, const Var &group_var,
                                  int lanes) {
    ffi::Array<Stmt> stores;
    for (int lane = 0; lane < lanes; ++lane) {
      PrimExpr dst_index = SubstituteLoopVar(
          store->indices[0], loop_var, group_var, lanes, lane, &analyzer_);
      PrimExpr lane_value = Shuffle::ExtractElement(vector_value, lane);
      stores.push_back(BufferStore(store->buffer, lane_value, {dst_index}));
    }
    return SeqStmt(stores);
  }

  Optional<Stmt> TryVectorizeScalarCopyLoop(const For &loop) {
    const ForNode *op = loop.get();
    if (op->thread_binding.defined()) {
      return std::nullopt;
    }
    if (!IsZero(op->min)) {
      return std::nullopt;
    }

    const int64_t *extent = as_const_int(analyzer_.Simplify(op->extent));
    if (extent == nullptr) {
      return std::nullopt;
    }

    const auto *store = op->body.as<BufferStoreNode>();
    if (store == nullptr || store->predicate.defined() ||
        store->indices.size() != 1 || !store->value.dtype().is_scalar()) {
      return std::nullopt;
    }

    const auto *load = store->value.as<BufferLoadNode>();
    if (load == nullptr || load->predicate.defined() ||
        load->indices.size() != 1 || !load->dtype.is_scalar()) {
      return std::nullopt;
    }

    if (store->buffer->dtype != load->buffer->dtype ||
        store->buffer->dtype != store->value.dtype() ||
        !IsSupportedScalarDType(store->buffer->dtype)) {
      return std::nullopt;
    }

    const bool dst_is_global = IsGlobalBuffer(store->buffer);
    const bool src_is_global = IsGlobalBuffer(load->buffer);
    if (dst_is_global == src_is_global) {
      return std::nullopt;
    }

    std::vector<int> candidate_lanes =
        GetCandidateLanes(store->buffer->dtype, *extent);
    if (candidate_lanes.empty()) {
      return std::nullopt;
    }

    auto try_vectorize_with_lanes = [&](int lanes) -> Optional<Stmt> {
      Var group_var(op->loop_var->name_hint + "_vec", op->loop_var.dtype());
      AccessPattern dst =
          AnalyzeAccess(store->indices[0], op->loop_var, group_var, lanes);
      AccessPattern src =
          AnalyzeAccess(load->indices[0], op->loop_var, group_var, lanes);
      const bool dst_vectorizable =
          dst_is_global && dst.contiguous && dst.aligned;
      const bool src_vectorizable =
          src_is_global && src.contiguous && src.aligned;
      if (!dst_vectorizable && !src_vectorizable) {
        return std::nullopt;
      }

      PrimExpr dst_ramp = MakeContiguousRamp(dst.base, lanes);
      PrimExpr src_ramp = MakeContiguousRamp(src.base, lanes);
      Stmt new_body;
      if (src_vectorizable) {
        PrimExpr vector_load = BufferLoad(load->buffer, {src_ramp});
        Var vec(op->loop_var->name_hint + "_single_side",
                load->dtype.with_lanes(lanes));
        Stmt scalar_stores = MakeScalarStoresFromVector(
            store, vec, op->loop_var, group_var, lanes);
        new_body = SeqStmt({Bind(vec, vector_load), scalar_stores});
      } else {
        PrimExpr vector_value =
            MakeVectorFromScalarLoads(load, op->loop_var, group_var, lanes);
        new_body = BufferStore(store->buffer, vector_value, {dst_ramp});
      }

      PrimExpr group_extent = make_const(op->extent.dtype(), *extent / lanes);
      return For(group_var, IntImm(op->loop_var.dtype(), 0), group_extent,
                 op->kind, new_body, op->thread_binding, op->annotations,
                 std::nullopt, op->span);
    };

    for (int lanes : candidate_lanes) {
      Optional<Stmt> vectorized = try_vectorize_with_lanes(lanes);
      if (vectorized.defined()) {
        return vectorized;
      }
    }
    return std::nullopt;
  }
};

using namespace tirx::transform;

tvm::transform::Pass VectorizeSingleSide() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target_opt.defined()) {
      return f;
    }
    Target target = target_opt.value();
    if (!TargetIsMusa(target)) {
      return f;
    }

    bool enable_lower_ldgstg =
        ctx->GetConfig<Bool>(kEnableLowerLDGSTG, Bool(false)).value();
    bool disable_vectorize_single_side =
        ctx->GetConfig<Bool>(kDisableVectorizeSingleSide, Bool(false)).value();
    bool disable_vectorize =
        ctx->GetConfig<Bool>("tir.disable_vectorize", Bool(false)).value();
    if (!enable_lower_ldgstg || disable_vectorize_single_side ||
        disable_vectorize) {
      return f;
    }

    auto *n = f.CopyOnWrite();
    n->body = VectorizeSingleSideRewriter()(n->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.VectorizeSingleSide", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.VectorizeSingleSide",
                        VectorizeSingleSide);
}

} // namespace tl
} // namespace tvm
