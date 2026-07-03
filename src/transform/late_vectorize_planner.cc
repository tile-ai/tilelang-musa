/*!
 * \file late_vectorize_planner.cc
 * \brief Late vectorization planner for MUSA SIMD ops.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../target/utils.h"
#include "arith/ir_mutator_with_analyzer.h"
#include "loop_vectorize.h"

namespace tvm {
namespace tl {

using namespace tirx;
using arith::IRMutatorWithAnalyzer;

class LateVectorizePlanner : public IRMutatorWithAnalyzer {
public:
  static PrimFunc Substitute(PrimFunc f) {
    arith::Analyzer analyzer;
    LateVectorizePlanner substituter(&analyzer);
    auto *fptr = f.CopyOnWrite();
    fptr->body = substituter.VisitStmt(f->body);
    return f;
  }

private:
  explicit LateVectorizePlanner(arith::Analyzer *analyzer)
      : IRMutatorWithAnalyzer(analyzer) {}

  static bool IsMusaFP8Type(DataType t) {
    return t.is_float8_e4m3() || t.is_float8_e5m2();
  }

  // Keep this list aligned with vectorized cast paths in codegen_musa.cc.
  // Supported vector intrinsics currently include:
  //   1) fp16 <-> fp32
  //   2) bf16 <-> fp32
  //   3) fp32 <-> fp8 (e4m3/e5m2)
  //   4) fp16 <-> fp8 (e4m3/e5m2)
  static bool IsMusaSIMDCastCandidate(DataType from_ty, DataType to_ty) {
    if (from_ty.lanes() != 1 || to_ty.lanes() != 1 || from_ty == to_ty) {
      return false;
    }
    if ((from_ty.is_float16() && to_ty.is_float() && to_ty.bits() == 32) ||
        (from_ty.is_float() && from_ty.bits() == 32 && to_ty.is_float16())) {
      return true;
    }
    if ((from_ty.is_bfloat16() && to_ty.is_float() && to_ty.bits() == 32) ||
        (from_ty.is_float() && from_ty.bits() == 32 && to_ty.is_bfloat16())) {
      return true;
    }
    if (from_ty.is_float() && from_ty.bits() == 32 && IsMusaFP8Type(to_ty)) {
      return true;
    }
    if (IsMusaFP8Type(from_ty) && to_ty.is_float() && to_ty.bits() == 32) {
      return true;
    }
    if ((from_ty.is_float16() && IsMusaFP8Type(to_ty)) ||
        (IsMusaFP8Type(from_ty) && to_ty.is_float16())) {
      return true;
    }
    return false;
  }

  PrimExpr GetElementOffset(const Buffer &buffer,
                            const tvm::ffi::Array<PrimExpr> &indices) const {
    PrimExpr elem_offset = 0;
    if (buffer->strides.empty()) {
      PrimExpr stride = 1;
      for (int i = static_cast<int>(indices.size()) - 1; i >= 0; --i) {
        elem_offset += indices[i] * stride;
        stride = stride * buffer->shape[i];
      }
    } else {
      for (int i = 0; i < static_cast<int>(indices.size()); ++i) {
        elem_offset += indices[i] * buffer->strides[i];
      }
    }
    return analyzer_->Simplify(elem_offset);
  }

  const BufferStoreNode *GetSingleStore(const Stmt &body) const {
    const BufferStoreNode *store = nullptr;
    bool is_single_store_loop = true;
    PostOrderVisit(body, [&](const ObjectRef &obj) {
      if (!is_single_store_loop) {
        return;
      }
      if (obj.as<ForNode>() || obj.as<IfThenElseNode>()) {
        is_single_store_loop = false;
        return;
      }
      const auto *candidate = obj.as<BufferStoreNode>();
      if (!candidate) {
        return;
      }
      if (store != nullptr) {
        is_single_store_loop = false;
        return;
      }
      store = candidate;
    });
    return is_single_store_loop ? store : nullptr;
  }

  bool HasElementwiseAccessPattern(const For &loop,
                                   const BufferStoreNode *store) const {
    int vectorize_size = GetVectorizeSize(loop);
    if (vectorize_size <= 1) {
      return false;
    }

    PrimExpr store_offset = GetElementOffset(store->buffer, store->indices);
    if (CanProveIndependent(store_offset, loop->loop_var, analyzer_)) {
      return false;
    }
    if (!IndicesCanVectorize(store_offset, loop->loop_var, loop->extent,
                             vectorize_size, analyzer_)) {
      return false;
    }

    bool accesses_match = true;
    PostOrderVisit(store->value, [&](const ObjectRef &obj) {
      if (!accesses_match) {
        return;
      }
      const auto *load = obj.as<BufferLoadNode>();
      if (!load) {
        return;
      }
      PrimExpr load_offset = GetElementOffset(load->buffer, load->indices);
      if (CanProveIndependent(load_offset, loop->loop_var, analyzer_)) {
        return;
      }
      accesses_match = IndicesCanVectorize(
          load_offset, loop->loop_var, loop->extent, vectorize_size, analyzer_);
    });
    return accesses_match;
  }

  bool MatchesExp2VecPattern(const For &loop) const {
    if (loop->kind == ForKind::kVectorized) {
      return false;
    }
    const BufferStoreNode *store = GetSingleStore(loop->body);
    if (store == nullptr) {
      return false;
    }
    const auto *call = store->value.as<CallNode>();
    if (call == nullptr || !call->op.same_as(Op::Get("tir.exp2"))) {
      return false;
    }
    const DataType &t = call->dtype;
    if (!(t.is_float() && t.bits() == 32 && t.lanes() == 1)) {
      return false;
    }
    return HasElementwiseAccessPattern(loop, store);
  }

  bool MatchesCvtVecPattern(const For &loop) const {
    if (loop->kind == ForKind::kVectorized) {
      return false;
    }
    const BufferStoreNode *store = GetSingleStore(loop->body);
    if (store == nullptr) {
      return false;
    }
    const auto *cast = store->value.as<CastNode>();
    if (cast == nullptr ||
        !IsMusaSIMDCastCandidate(cast->value.dtype(), cast->dtype)) {
      return false;
    }
    return HasElementwiseAccessPattern(loop, store);
  }

  bool ContainsMusaSIMDOpportunity(const For &loop) const {
    return MatchesExp2VecPattern(loop) || MatchesCvtVecPattern(loop);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    For for_node = Downcast<For>(IRMutatorWithAnalyzer::VisitStmt_(op));
    if (ContainsMusaSIMDOpportunity(for_node)) {
      return VectorizeLoop(for_node);
    }
    return for_node;
  }
};

tvm::transform::Pass LateVectorizePlanner() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    return LateVectorizePlanner::Substitute(std::move(f));
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LateVectorizePlanner", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LateVectorizePlanner",
                        LateVectorizePlanner);
}

} // namespace tl
} // namespace tvm
