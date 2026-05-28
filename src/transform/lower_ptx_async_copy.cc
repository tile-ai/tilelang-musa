/*!
 * \brief Lower eligible global->shared copies into PTX cp.async
 * \file lower_ptx_async_copy.cc
 */
#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <algorithm>
#include <cstdint>
#include <limits>
#include <optional>
#include <vector>

#include "../op/builtin.h"
#include "../op/utils.h"
#include "../target/utils.h"
#include "ptx_async_copy_injector.h"
#include "tir/ir/buffer_common.h"
#include "tvm/tir/stmt.h"

namespace tvm {
namespace tl {

using namespace tir;

class PTXAsyncCopyInjector : public StmtMutator {
public:
  explicit PTXAsyncCopyInjector(bool enable_auto_async_copy,
                                bool async_without_async_commit_wait,
                                bool disable_force_async_wait = false,
                                bool sync_inside_conditionals = false)
      : enable_auto_async_copy_(enable_auto_async_copy),
        async_without_async_commit_wait_(async_without_async_commit_wait),
        disable_force_async_wait_(disable_force_async_wait),
        sync_inside_conditionals_(sync_inside_conditionals),
        disable_safe_robust_copy_predication_(
            tvm::transform::PassContext::Current()
                ->GetConfig<Bool>(kDisableSafeRobustCopyPredication,
                                  Bool(false))
                .value()) {}

  bool InjectedPTXAsyncCopy() const { return injected_ptx_async_copy_; }

  Stmt Finalize(Stmt body) {
    if (!pending_sync_copies_ || UseExplicitAsyncSemantics()) {
      pending_sync_copies_ = false;
      uncommitted_sync_copies_ = false;
      return body;
    }

    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(body);
    AppendSyncVisibility(&seq, uncommitted_sync_copies_);
    pending_sync_copies_ = false;
    uncommitted_sync_copies_ = false;
    return SeqStmt(seq);
  }

  Stmt VisitStmt_(const AttrStmtNode *op) final {
    if (op->attr_key == tl::attr::kSourceRobustDesc) {
      PrimExpr previous_src_robust_desc = current_src_robust_desc_;
      current_src_robust_desc_ = op->value;
      Stmt body = this->VisitStmt(op->body);
      current_src_robust_desc_ = previous_src_robust_desc;
      return AttrStmt(op->node, op->attr_key, op->value, body);
    }
    if (op->attr_key == tl::attr::kForceAsyncCopy) {
      bool previous_force_async_copy = in_force_async_copy_;
      in_force_async_copy_ = true;
      Stmt body = this->VisitStmt(op->body);
      in_force_async_copy_ = previous_force_async_copy;
      return AttrStmt(op->node, op->attr_key, op->value, body);
    }
    if (op->attr_key == tir::attr::async_scope) {
      ++explicit_async_scope_depth_;
      Stmt body = this->VisitStmt(op->body);
      --explicit_async_scope_depth_;
      // `async_scope` is a lowering-only marker for cp.async semantics.
      return body;
    }
    return StmtMutator::VisitStmt_(op);
  }

  Stmt VisitStmt_(const ForNode *op) final {
    // Track nested vectorized loop extents so we can rewrite element-wise
    // copies (e.g. float16 stores) into `tir.ptx_cp_async` with element bytes,
    // relying on the later `tl.VectorizeLoop` pass to widen:
    //   for v in T.vectorized(k): ptx_cp_async(dst, src, elem_bytes)
    // => ptx_cp_async(dst_base, src_base, elem_bytes * k)
    //
    // This mirrors the logic in `CPAsyncStoreRewriter` used by `T.copy`
    // lowering, and avoids duplicating vectorize-loop collapse here.
    int previous_vectorized_lanes = current_vectorized_lanes_;
    bool pushed_vectorized_loop = false;
    if (op->kind == ForKind::kVectorized) {
      const auto *extent_imm = op->extent.as<IntImmNode>();
      ICHECK(extent_imm)
          << "Vectorized loops must have constant extent, but got "
          << op->extent;
      int lanes = static_cast<int>(extent_imm->value);
      if (lanes > 1 && current_vectorized_lanes_ <=
                           std::numeric_limits<int>::max() / lanes) {
        current_vectorized_lanes_ *= lanes;
        active_vectorized_loops_.push_back({op->loop_var, lanes});
        pushed_vectorized_loop = true;
      }
    }
    Stmt stmt = StmtMutator::VisitStmt_(op);
    if (pushed_vectorized_loop) {
      active_vectorized_loops_.pop_back();
    }
    current_vectorized_lanes_ = previous_vectorized_lanes;
    return stmt;
  }

  Optional<Stmt> TryInjectPTX(const BufferLoadNode *load,
                              const BufferStoreNode *store,
                              bool predicated = false,
                              const PrimExpr &predicate_value = PrimExpr()) {
    // Pipeline:
    // 1) Analyze source/destination indices and transfer width eligibility.
    // 2) Validate pointer type metadata for access_ptr construction.
    // 3) Build cp.async with scalar/vectorized offsets if representable.
    std::optional<CopyIndexInfo> index_info = PrepareCopyIndexInfo(load, store);
    if (!index_info.has_value()) {
      return Optional<Stmt>();
    }

    std::optional<PointerTypeInfo> ptr_info =
        PreparePointerTypeInfo(load, store);
    if (!ptr_info.has_value()) {
      // Be conservative: if pointer metadata is missing, skip injection.
      return Optional<Stmt>();
    }

    if (index_info->index_lanes == 1) {
      if (current_vectorized_lanes_ > 1 &&
          !HasContiguousVectorizedOffsets(index_info->src_index,
                                          index_info->dst_index)) {
        return Optional<Stmt>();
      }
      return MakeCPAsyncStmtFromLoads(
          store, ptr_info.value(),
          /*dst_base_load=*/BufferLoad(store->buffer, store->indices),
          /*src_base_load=*/BufferLoad(load->buffer, load->indices),
          /*bytes=*/index_info->transfer_bytes, predicated, predicate_value,
          current_src_robust_desc_);
    }

    Optional<Array<PrimExpr>> src_base_indices =
        ExtractVectorBaseIndices(load->indices);
    Optional<Array<PrimExpr>> dst_base_indices =
        ExtractVectorBaseIndices(store->indices);
    if (!src_base_indices.defined() || !dst_base_indices.defined()) {
      // If we can't extract base indices from vectorized accesses, fall back.
      if (predicated) {
        LOG(WARNING)
            << "Cannot extract base indices from vectorized accesses for "
               "predicated cp.async; falling back to regular buffer store/load";
      }
      return Optional<Stmt>();
    }
    return MakeCPAsyncStmtFromLoads(
        store, ptr_info.value(),
        /*dst_base_load=*/BufferLoad(store->buffer, dst_base_indices.value()),
        /*src_base_load=*/BufferLoad(load->buffer, src_base_indices.value()),
        /*bytes=*/index_info->transfer_bytes, predicated, predicate_value,
        current_src_robust_desc_);
  }

  Stmt VisitStmt_(const SeqStmtNode *op) final {
    if (UseExplicitAsyncSemantics()) {
      return StmtMutator::VisitStmt_(op);
    }

    // Insert commit+wait at statement boundaries to preserve synchronous
    // semantics for normal global->shared BufferStore copies.
    //
    // Important: avoid flushing inside inner loop bodies just because there
    // are trailing no-op statements (e.g., Evaluate(0)) after the injected
    // cp.async. Instead, treat "pure copy region" statements as part of the
    // copy run and only flush right before the next non-copy statement.
    Array<Stmt> out;
    out.reserve(op->seq.size() + 2);

    CopySyncState sync_state{pending_sync_copies_, uncommitted_sync_copies_};
    pending_sync_copies_ = false;
    uncommitted_sync_copies_ = false;

    for (const Stmt &stmt : op->seq) {
      VisitedStmtInfo visited_info = VisitAndAnalyzeStmt(stmt);
      bool stmt_is_pure_copy_region = visited_info.analysis.is_pure_copy_region;

      // Before we execute a non-copy statement, we must preserve synchronous
      // semantics for injected cp.async stores by making the data visible.
      if (sync_state.open_copy_region && !stmt_is_pure_copy_region) {
        AppendSyncVisibility(&out, sync_state.uncommitted_transfers);
        sync_state.open_copy_region = false;
        sync_state.uncommitted_transfers = false;
      }

      // If we are carrying uncommitted injected cp.async into an explicit wait,
      // ensure they are committed so the wait actually covers them.
      if (sync_state.open_copy_region && sync_state.uncommitted_transfers &&
          visited_info.analysis.wait > 0) {
        out.push_back(MakeCommitGroupStmt());
        sync_state.uncommitted_transfers = false;
      }

      out.push_back(visited_info.visited);

      if (visited_info.opens_copy_region) {
        sync_state.open_copy_region = true;
        sync_state.uncommitted_transfers =
            sync_state.uncommitted_transfers ||
            visited_info.has_uncommitted_transfers;
      }

      if (visited_info.analysis.commit > 0) {
        // A commit closes the currently open group, so there are no longer any
        // uncommitted injected cp.async transfers.
        sync_state.uncommitted_transfers = false;
      }

      if (visited_info.analysis.wait > 0) {
        // Any explicit wait serves as a synchronization boundary for injected
        // synchronous copies.
        sync_state.open_copy_region = false;
        sync_state.uncommitted_transfers = false;
      }
    }

    pending_sync_copies_ = sync_state.open_copy_region;
    uncommitted_sync_copies_ = sync_state.uncommitted_transfers;

    if (out.empty()) {
      return Evaluate(0);
    }
    if (out.size() == 1) {
      return out[0];
    }
    return SeqStmt(out);
  }

  Stmt VisitStmt_(const IfThenElseNode *op) final {
    if (UseExplicitAsyncSemantics()) {
      return StmtMutator::VisitStmt_(op);
    }

    // Treat branches as separate control flow paths. We propagate pending
    // synchronous copies into both branches (they occur before the branch),
    // but do not let mutations in one branch affect the other.
    bool pending_before = pending_sync_copies_;
    bool uncommitted_before = uncommitted_sync_copies_;

    // MUSA's compiler is sensitive to commit_group on control-flow paths that
    // did not execute a cp.async, so keep branch-local copies branch-local.
    if (sync_inside_conditionals_ && !pending_before && !uncommitted_before) {
      pending_sync_copies_ = false;
      uncommitted_sync_copies_ = false;
      Stmt then_case = this->VisitStmt(op->then_case);
      bool pending_then = pending_sync_copies_;
      bool uncommitted_then = uncommitted_sync_copies_;
      if (pending_then) {
        then_case =
            AppendSyncVisibilityToStmt(std::move(then_case), uncommitted_then);
      }

      Optional<Stmt> else_case;
      if (op->else_case.defined()) {
        pending_sync_copies_ = false;
        uncommitted_sync_copies_ = false;
        else_case = this->VisitStmt(op->else_case.value());
        bool pending_else = pending_sync_copies_;
        bool uncommitted_else = uncommitted_sync_copies_;
        if (pending_else) {
          else_case = AppendSyncVisibilityToStmt(std::move(else_case.value()),
                                                 uncommitted_else);
        }
      }

      pending_sync_copies_ = false;
      uncommitted_sync_copies_ = false;
      if (then_case.same_as(op->then_case) &&
          (!else_case.defined() || else_case.same_as(op->else_case))) {
        return tvm::ffi::GetRef<Stmt>(op);
      }
      return IfThenElse(op->condition, then_case, else_case);
    }

    pending_sync_copies_ = pending_before;
    uncommitted_sync_copies_ = uncommitted_before;
    Stmt then_case = this->VisitStmt(op->then_case);
    bool pending_then = pending_sync_copies_;
    bool uncommitted_then = uncommitted_sync_copies_;

    bool pending_else = pending_before;
    bool uncommitted_else = uncommitted_before;
    Optional<Stmt> else_case;
    if (op->else_case.defined()) {
      pending_sync_copies_ = pending_before;
      uncommitted_sync_copies_ = uncommitted_before;
      else_case = this->VisitStmt(op->else_case.value());
      pending_else = pending_sync_copies_;
      uncommitted_else = uncommitted_sync_copies_;
    }

    pending_sync_copies_ = pending_then || pending_else;
    uncommitted_sync_copies_ = uncommitted_then || uncommitted_else;

    if (then_case.same_as(op->then_case) &&
        (!else_case.defined() || else_case.same_as(op->else_case))) {
      return tvm::ffi::GetRef<Stmt>(op);
    }
    return IfThenElse(op->condition, then_case, else_case);
  }

  Stmt VisitStmt_(const BufferStoreNode *store) final {
    if (!IsSharedBuffer(store->buffer)) {
      return StmtMutator::VisitStmt_(store);
    }
    // Only lower copies in regions where async-copy rewrite is enabled.
    if (!enable_auto_async_copy_ && !in_force_async_copy_) {
      return StmtMutator::VisitStmt_(store);
    }

    Optional<PrimExpr> predicate = std::nullopt;
    const BufferLoadNode *load =
        MatchZeroFillBufferLoad(store->value, &predicate);
    if (load) {
      Optional<Stmt> injected =
          TryInjectPTX(load, store, predicate.defined(),
                       predicate.defined() ? predicate.value() : PrimExpr());
      if (injected.defined()) {
        injected_ptx_async_copy_ = true;
        if (!UseExplicitAsyncSemantics() &&
            !(in_force_async_copy_ && disable_force_async_wait_)) {
          pending_sync_copies_ = true;
          uncommitted_sync_copies_ = true;
        }
        return injected.value();
      }
    }

    return StmtMutator::VisitStmt_(store);
  }

private:
  bool UseExplicitAsyncSemantics() const {
    return async_without_async_commit_wait_ || explicit_async_scope_depth_ > 0;
  }

  // A copy candidate represented after flattening source/destination indexing.
  struct CopyIndexInfo {
    PrimExpr src_index;
    PrimExpr dst_index;
    int index_lanes{1};
    int transfer_bytes{0};
  };

  // Pointer element type metadata extracted from buffer handle annotations.
  struct PointerTypeInfo {
    DataType dst_elem_type;
    DataType src_elem_type;
  };

  // Synchronization state for injected cp.async runs carried across statements.
  struct CopySyncState {
    bool open_copy_region{false};
    bool uncommitted_transfers{false};
  };

  struct ActiveVectorizedLoop {
    Var loop_var;
    int extent;
  };

  // ---- Copy candidate analysis helpers ----
  static bool IsZeroValue(const PrimExpr &expr) {
    if (const auto *broadcast = expr.as<BroadcastNode>()) {
      return IsZeroValue(broadcast->value);
    }
    if (const auto *float_imm = expr.as<FloatImmNode>()) {
      return float_imm->value == 0.0f;
    }
    if (const auto *int_imm = expr.as<IntImmNode>()) {
      return int_imm->value == 0;
    }
    return false;
  }

  static const BufferLoadNode *
  MatchZeroFillBufferLoad(const PrimExpr &value,
                          Optional<PrimExpr> *predicate) {
    if (const auto *load = value.as<BufferLoadNode>()) {
      return load;
    }

    const auto *call = value.as<CallNode>();
    if (!call || !call->op.same_as(builtin::if_then_else()) ||
        !IsZeroValue(call->args[2])) {
      return nullptr;
    }

    const BufferLoadNode *load =
        MatchZeroFillBufferLoad(call->args[1], predicate);
    if (load == nullptr) {
      return nullptr;
    }

    *predicate =
        predicate->defined()
            ? Optional<PrimExpr>(And(call->args[0], predicate->value()))
            : Optional<PrimExpr>(call->args[0]);
    return load;
  }

  static Optional<PrimExpr>
  FlattenToLinearOffset(const Buffer &buf,
                        const ffi::Array<PrimExpr> &indices) {
    // Convert N-D indices (potentially with axis_separators) into a single
    // row-major linear element offset.
    ffi::Array<PrimExpr> physical = buf.OffsetOf(indices);
    Buffer flattened_buf = buf.GetFlattenedBuffer();
    if (physical.size() != flattened_buf->shape.size() || physical.empty()) {
      return Optional<PrimExpr>();
    }

    PrimExpr linear = physical[0];
    for (size_t i = 1; i < physical.size(); ++i) {
      linear = linear * flattened_buf->shape[i] + physical[i];
    }
    return linear;
  }

  std::optional<CopyIndexInfo>
  PrepareCopyIndexInfo(const BufferLoadNode *load,
                       const BufferStoreNode *store) {
    if (!IsGlobalBuffer(load->buffer)) {
      return std::nullopt;
    }

    Optional<PrimExpr> src_index_opt =
        FlattenToLinearOffset(load->buffer, load->indices);
    Optional<PrimExpr> dst_index_opt =
        FlattenToLinearOffset(store->buffer, store->indices);
    if (!src_index_opt.defined() || !dst_index_opt.defined()) {
      return std::nullopt;
    }

    PrimExpr src_index = src_index_opt.value();
    PrimExpr dst_index = dst_index_opt.value();
    if (src_index->dtype.lanes() != dst_index->dtype.lanes()) {
      // Not a straightforward vectorized copy; skip.
      return std::nullopt;
    }

    const int index_lanes = src_index->dtype.lanes();
    const int value_lanes = load->dtype.lanes();
    if (value_lanes > 1 && index_lanes > 1 && value_lanes != index_lanes) {
      // Mismatched vector lane representations; be conservative.
      return std::nullopt;
    }

    const int effective_lanes = std::max(value_lanes, index_lanes);
    const int elem_bytes = effective_lanes * load->dtype.bytes();
    const int total_bytes = static_cast<int>(elem_bytes) *
                            static_cast<int>(current_vectorized_lanes_);
    if (!IsValidCPAsyncTransferBytes(total_bytes)) {
      return std::nullopt;
    }

    CopyIndexInfo info;
    info.src_index = src_index;
    info.dst_index = dst_index;
    info.index_lanes = index_lanes;
    info.transfer_bytes = elem_bytes;
    return info;
  }

  static std::optional<PointerTypeInfo>
  PreparePointerTypeInfo(const BufferLoadNode *load,
                         const BufferStoreNode *store) {
    auto dst_elem_type = GetPointerType(store->buffer->data->type_annotation);
    auto src_elem_type = GetPointerType(load->buffer->data->type_annotation);
    if (!dst_elem_type.has_value() || !src_elem_type.has_value()) {
      return std::nullopt;
    }
    return PointerTypeInfo{dst_elem_type.value(), src_elem_type.value()};
  }

  // Common pattern after flattening a vectorized N-D buffer access:
  //   (broadcast(base_offset) + ramp(vec_base, 1, lanes))
  // or its commuted form:
  //   (ramp(vec_base, 1, lanes) + broadcast(base_offset))
  static PrimExpr ExtractRampBroadcastAddBase(const PrimExpr &index) {
    const auto *add = index.as<AddNode>();
    if (!add) {
      return PrimExpr();
    }
    const PrimExpr &lhs = add->a;
    const PrimExpr &rhs = add->b;
    if (const auto *lhs_ramp = lhs.as<RampNode>()) {
      if (!is_one(lhs_ramp->stride)) {
        return PrimExpr();
      }
      if (const auto *rhs_broadcast = rhs.as<BroadcastNode>()) {
        return tir::Add(lhs_ramp->base, rhs_broadcast->value);
      }
    }
    if (const auto *rhs_ramp = rhs.as<RampNode>()) {
      if (!is_one(rhs_ramp->stride)) {
        return PrimExpr();
      }
      if (const auto *lhs_broadcast = lhs.as<BroadcastNode>()) {
        return tir::Add(rhs_ramp->base, lhs_broadcast->value);
      }
    }
    return PrimExpr();
  }

  static bool IsLaneWiseScalarizableCall(const CallNode *call) {
    return call->op.same_as(builtin::bitwise_and()) ||
           call->op.same_as(builtin::bitwise_or()) ||
           call->op.same_as(builtin::bitwise_xor()) ||
           call->op.same_as(builtin::bitwise_not()) ||
           call->op.same_as(builtin::shift_left()) ||
           call->op.same_as(builtin::shift_right());
  }

  // Input: a scalar PrimExpr, or a vector PrimExpr whose lanes may all carry
  // the same value.
  //
  // Output: the scalar expression shared by all lanes. If `expr` is already
  // scalar, return it directly. If `expr` is a lane-invariant vector expression
  // such as Broadcast(x, lanes) or a whitelisted lane-wise call over
  // lane-invariant arguments, return the scalarized value. Otherwise return an
  // undefined PrimExpr().
  static PrimExpr ExtractLaneInvariantScalar(const PrimExpr &expr) {
    if (expr.dtype().lanes() == 1) {
      return expr;
    }
    if (const auto *broadcast = expr.as<BroadcastNode>()) {
      return broadcast->value;
    }
    if (const auto *cast = expr.as<CastNode>()) {
      PrimExpr value = ExtractLaneInvariantScalar(cast->value);
      if (!value.defined() || value.dtype().lanes() != 1) {
        return PrimExpr();
      }
      return tir::Cast(cast->dtype.with_lanes(1), value);
    }
    if (const auto *add = expr.as<AddNode>()) {
      PrimExpr a = ExtractLaneInvariantScalar(add->a);
      PrimExpr b = ExtractLaneInvariantScalar(add->b);
      return a.defined() && b.defined() ? tir::Add(a, b) : PrimExpr();
    }
    if (const auto *sub = expr.as<SubNode>()) {
      PrimExpr a = ExtractLaneInvariantScalar(sub->a);
      PrimExpr b = ExtractLaneInvariantScalar(sub->b);
      return a.defined() && b.defined() ? tir::Sub(a, b) : PrimExpr();
    }
    if (const auto *mul = expr.as<MulNode>()) {
      PrimExpr a = ExtractLaneInvariantScalar(mul->a);
      PrimExpr b = ExtractLaneInvariantScalar(mul->b);
      return a.defined() && b.defined() ? tir::Mul(a, b) : PrimExpr();
    }
    if (const auto *div = expr.as<FloorDivNode>()) {
      PrimExpr a = ExtractLaneInvariantScalar(div->a);
      PrimExpr b = ExtractLaneInvariantScalar(div->b);
      return a.defined() && b.defined() ? tir::FloorDiv(a, b) : PrimExpr();
    }
    if (const auto *mod = expr.as<FloorModNode>()) {
      PrimExpr a = ExtractLaneInvariantScalar(mod->a);
      PrimExpr b = ExtractLaneInvariantScalar(mod->b);
      return a.defined() && b.defined() ? tir::FloorMod(a, b) : PrimExpr();
    }
    if (const auto *call = expr.as<CallNode>()) {
      if (!IsLaneWiseScalarizableCall(call)) {
        return PrimExpr();
      }
      Array<PrimExpr> args;
      args.reserve(call->args.size());
      for (const PrimExpr &arg : call->args) {
        PrimExpr scalar_arg = ExtractLaneInvariantScalar(arg);
        if (!scalar_arg.defined() || scalar_arg.dtype().lanes() != 1) {
          return PrimExpr();
        }
        args.push_back(scalar_arg);
      }
      return tir::Call(call->dtype.with_lanes(1), call->op, args,
                       call->annotations);
    }
    return PrimExpr();
  }

  // Nested Add tree where one term is a unit-stride Ramp and every other term
  // is lane-invariant:
  //   offset_0 + ... + ramp(vec_base, 1, lanes) + ... + offset_n
  static PrimExpr ExtractNestedRampInvariantAddBase(const PrimExpr &index) {
    if (const auto *ramp = index.as<RampNode>()) {
      return is_one(ramp->stride) ? ramp->base : PrimExpr();
    }

    const auto *add = index.as<AddNode>();
    if (!add) {
      return PrimExpr();
    }

    PrimExpr lhs_base = ExtractNestedRampInvariantAddBase(add->a);
    if (lhs_base.defined()) {
      PrimExpr rhs_offset = ExtractLaneInvariantScalar(add->b);
      if (rhs_offset.defined()) {
        return tir::Add(lhs_base, rhs_offset);
      }
    }

    PrimExpr rhs_base = ExtractNestedRampInvariantAddBase(add->b);
    if (rhs_base.defined()) {
      PrimExpr lhs_offset = ExtractLaneInvariantScalar(add->a);
      if (lhs_offset.defined()) {
        return tir::Add(rhs_base, lhs_offset);
      }
    }

    return PrimExpr();
  }

  static PrimExpr ExtractVectorBase(const PrimExpr &index) {
    if (index.dtype().lanes() == 1) {
      return index;
    }
    if (const auto *broadcast = index.as<BroadcastNode>()) {
      return broadcast->value;
    }
    if (const auto *ramp = index.as<RampNode>()) {
      if (is_one(ramp->stride)) {
        return ramp->base;
      }
    }

    PrimExpr add_base = ExtractRampBroadcastAddBase(index);
    if (add_base.defined()) {
      return add_base;
    }

    PrimExpr nested_add_base = ExtractNestedRampInvariantAddBase(index);
    if (nested_add_base.defined()) {
      return nested_add_base;
    }

    return PrimExpr();
  }

  static Optional<Array<PrimExpr>>
  ExtractVectorBaseIndices(const Array<PrimExpr> &indices) {
    Array<PrimExpr> base_indices;
    base_indices.reserve(indices.size());
    for (const PrimExpr &index : indices) {
      PrimExpr base = ExtractVectorBase(index);
      if (!base.defined()) {
        return Optional<Array<PrimExpr>>();
      }
      base_indices.push_back(base);
    }
    return base_indices;
  }

  static PrimExpr MakeAccessPtrFromLoad(const BufferLoad &base_load, int extent,
                                        int rw_mask) {
    return Call(DataType::Handle(), tvm::tl::access_ptr(),
                {base_load, IntImm(DataType::Int(32), extent),
                 IntImm(DataType::Int(32), rw_mask)});
  }

  Optional<Stmt> MakeCPAsyncStmtFromLoads(const BufferStoreNode *store,
                                          const PointerTypeInfo &ptr_info,
                                          const BufferLoad &dst_base_load,
                                          const BufferLoad &src_base_load,
                                          int bytes, bool predicated,
                                          const PrimExpr &predicate_value,
                                          const PrimExpr &robust_desc) const {
    int dst_elem_count = bytes / ptr_info.dst_elem_type.bytes();
    int src_elem_count = bytes / ptr_info.src_elem_type.bytes();
    if (dst_elem_count <= 0 || src_elem_count <= 0) {
      return Optional<Stmt>();
    }

    PrimExpr dst_access_ptr =
        MakeAccessPtrFromLoad(dst_base_load, dst_elem_count, /*rw_mask=*/2);
    PrimExpr src_access_ptr =
        MakeAccessPtrFromLoad(src_base_load, src_elem_count, /*rw_mask=*/1);

    ffi::Array<PrimExpr> cp_async_args;
    Op op = tvm::tir::builtin::ptx_cp_async();
    if (robust_desc.defined()) {
      op = tl::musa_cp_async_robust();
      auto [robust_base, robust_size] = GetRobustDescArgs(robust_desc);
      cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes),
                       robust_base, robust_size};
      if (predicated && !disable_safe_robust_copy_predication_) {
        cp_async_args.push_back(predicate_value);
      }
    } else if (predicated) {
      cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes),
                       predicate_value};
    } else {
      cp_async_args = {dst_access_ptr, src_access_ptr, PrimExpr(bytes)};
    }
    return Evaluate(Call(store->buffer->dtype, op, cp_async_args));
  }

  static Stmt MakeCommitGroupStmt() {
    return Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
  }

  static Stmt MakeWaitGroupStmt(int n) {
    return Evaluate(Call(DataType::Handle(), builtin::ptx_wait_group(),
                         {IntImm(DataType::Int(32), n)}));
  }

  static std::pair<PrimExpr, PrimExpr> GetRobustDescArgs(const PrimExpr &desc) {
    const auto *call = desc.as<CallNode>();
    ICHECK(call && call->op.same_as(tl::make_robust_desc()))
        << "Expected tl.make_robust_desc call, but got " << desc;
    ICHECK_EQ(call->args.size(), 2);
    return {call->args[0], call->args[1]};
  }

  // ---- Vectorized-offset contiguity helpers ----
  static bool TryGetConstInt64(const PrimExpr &expr, int64_t *value) {
    if (const auto *imm = expr.as<IntImmNode>()) {
      *value = imm->value;
      return true;
    }
    return false;
  }

  bool HasUnitStrideForVectorizedLoop(const PrimExpr &expr,
                                      const ActiveVectorizedLoop &loop) {
    PrimExpr prev = analyzer_.Simplify(
        Substitute(expr, {{loop.loop_var, IntImm(loop.loop_var->dtype, 0)}}));

    int64_t stride = 0;
    for (int value = 1; value < loop.extent; ++value) {
      PrimExpr curr = analyzer_.Simplify(Substitute(
          expr, {{loop.loop_var, IntImm(loop.loop_var->dtype, value)}}));
      PrimExpr delta = analyzer_.Simplify(curr - prev);
      int64_t delta_value = 0;
      if (!TryGetConstInt64(delta, &delta_value)) {
        return false;
      }
      if (value == 1) {
        stride = delta_value;
      } else if (delta_value != stride) {
        return false;
      }
      prev = curr;
    }

    return stride == 1;
  }

  bool HasContiguousVectorizedOffsets(const PrimExpr &src_index,
                                      const PrimExpr &dst_index) {
    for (const auto &loop : active_vectorized_loops_) {
      if (!HasUnitStrideForVectorizedLoop(src_index, loop) ||
          !HasUnitStrideForVectorizedLoop(dst_index, loop)) {
        return false;
      }
    }
    return true;
  }

  // ---- Copy-region synchronization analysis helpers ----
  struct CopyRegionAnalysis {
    bool is_pure_copy_region = true;
    int commit = 0;
    int wait = 0;
  };

  struct VisitedStmtInfo {
    Stmt visited;
    CopyRegionAnalysis analysis;
    bool opens_copy_region{false};
    bool has_uncommitted_transfers{false};
  };

  static CopyRegionAnalysis
  MergeCopyRegionAnalysis(CopyRegionAnalysis a, const CopyRegionAnalysis &b) {
    a.is_pure_copy_region = a.is_pure_copy_region && b.is_pure_copy_region;
    a.commit += b.commit;
    a.wait += b.wait;
    return a;
  }

  static CopyRegionAnalysis AnalyzeCopyRegion(const Stmt &stmt) {
    CopyRegionAnalysis out;
    if (!stmt.defined()) {
      return out;
    }
    if (const auto *seq = stmt.as<SeqStmtNode>()) {
      for (const Stmt &s : seq->seq) {
        out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(s));
      }
      return out;
    }
    if (const auto *ite = stmt.as<IfThenElseNode>()) {
      // Ignore the condition: treat it as pure control flow, and only care
      // whether the branches are pure copy regions so we can hoist sync out.
      out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(ite->then_case));
      if (ite->else_case.defined()) {
        out = MergeCopyRegionAnalysis(
            out, AnalyzeCopyRegion(ite->else_case.value()));
      }
      return out;
    }
    if (const auto *eval = stmt.as<EvaluateNode>()) {
      if (is_const_int(eval->value)) {
        return out;
      }
      const auto *call = eval->value.as<CallNode>();
      if (!call) {
        out.is_pure_copy_region = false;
        return out;
      }
      if (call->op.same_as(builtin::ptx_cp_async()) ||
          call->op.same_as(tl::ptx_cp_async()) ||
          call->op.same_as(tl::musa_cp_async_robust())) {
        return out;
      }
      if (call->op.same_as(builtin::ptx_commit_group())) {
        out.commit += 1;
        return out;
      }
      if (call->op.same_as(builtin::ptx_wait_group())) {
        out.wait += 1;
        return out;
      }
      out.is_pure_copy_region = false;
      return out;
    }
    if (const auto *let = stmt.as<LetStmtNode>()) {
      return AnalyzeCopyRegion(let->body);
    }
    if (const auto *attr = stmt.as<AttrStmtNode>()) {
      return AnalyzeCopyRegion(attr->body);
    }
    if (const auto *loop = stmt.as<ForNode>()) {
      return AnalyzeCopyRegion(loop->body);
    }
    if (const auto *block = stmt.as<BlockNode>()) {
      if (block->init.defined()) {
        out = MergeCopyRegionAnalysis(out,
                                      AnalyzeCopyRegion(block->init.value()));
      }
      out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(block->body));
      return out;
    }
    if (const auto *realize = stmt.as<BlockRealizeNode>()) {
      // Treat the predicate as pure control flow (no side effects). We only
      // care whether the realized body is a pure copy region so we can hoist
      // the final commit+wait out of sequential loop nests.
      const BlockNode *block = realize->block.get();
      if (block->init.defined()) {
        out = MergeCopyRegionAnalysis(out,
                                      AnalyzeCopyRegion(block->init.value()));
      }
      out = MergeCopyRegionAnalysis(out, AnalyzeCopyRegion(block->body));
      return out;
    }
    out.is_pure_copy_region = false;
    return out;
  }

  VisitedStmtInfo VisitAndAnalyzeStmt(const Stmt &stmt) {
    pending_sync_copies_ = false;
    uncommitted_sync_copies_ = false;

    Stmt visited = this->VisitStmt(stmt);
    VisitedStmtInfo out;
    out.visited = visited;
    out.analysis = AnalyzeCopyRegion(visited);
    out.opens_copy_region = pending_sync_copies_;
    out.has_uncommitted_transfers = uncommitted_sync_copies_;
    return out;
  }

  // ---- Synchronization emission helpers ----
  void AppendSyncVisibility(Array<Stmt> *seq, bool include_commit) const {
    if (include_commit) {
      seq->push_back(MakeCommitGroupStmt());
    }
    seq->push_back(MakeWaitGroupStmt(0));
  }

  Stmt AppendSyncVisibilityToStmt(Stmt stmt, bool include_commit) const {
    Array<Stmt> seq;
    seq.reserve(3);
    if (!is_no_op(stmt)) {
      seq.push_back(stmt);
    }
    AppendSyncVisibility(&seq, include_commit);
    return seq.size() == 1 ? seq[0] : SeqStmt(seq);
  }

  // Note: AnalyzeCopyRegion replaces both the old `IsPureCopyRegion` and
  // `SummarizeAsyncIntrinsics` helpers to avoid redundant traversals.

  bool enable_auto_async_copy_{true};
  bool async_without_async_commit_wait_{false};
  bool disable_force_async_wait_{false};
  bool sync_inside_conditionals_{false};
  bool disable_safe_robust_copy_predication_{false};
  int explicit_async_scope_depth_{0};
  int current_vectorized_lanes_{1};
  std::vector<ActiveVectorizedLoop> active_vectorized_loops_;
  arith::Analyzer analyzer_;
  bool in_force_async_copy_{false};
  PrimExpr current_src_robust_desc_;
  bool injected_ptx_async_copy_{false};
  bool pending_sync_copies_{false};
  bool uncommitted_sync_copies_{false};
};

using namespace tir::transform;

PTXAsyncCopyInjectResult
InjectPTXAsyncCopy(const Stmt &body, bool enable_auto_async_copy,
                   bool async_without_async_commit_wait) {
  PTXAsyncCopyInjector injector(enable_auto_async_copy,
                                async_without_async_commit_wait,
                                /*disable_force_async_wait=*/false);
  Stmt injected = injector(body);
  return {injector.Finalize(injected), injector.InjectedPTXAsyncCopy()};
}

tvm::transform::Pass LowerPTXAsyncCopy() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto target_opt = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target_opt.defined()) {
      return f;
    }
    Target target = target_opt.value();
    if (!TargetIsCuda(target) && !TargetIsMusa(target)) {
      return f;
    }

    if (!TargetHasAsyncCopy(target)) {
      // Graceful fallback on older architectures.
      return f;
    }

    bool enable_auto_async_copy =
        ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
    bool disable_thread_storage_sync =
        ctx->GetConfig<Bool>(kDisableThreadStorageSync, Bool(false)).value();

    auto *n = f.CopyOnWrite();
    PTXAsyncCopyInjector injector(enable_auto_async_copy,
                                  /*async_without_async_commit_wait=*/false,
                                  disable_thread_storage_sync,
                                  /*sync_inside_conditionals=*/
                                  TargetIsMusa(target));
    n->body = injector.Finalize(injector(n->body));
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerPTXAsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerPTXAsyncCopy", LowerPTXAsyncCopy);
}

} // namespace tl
} // namespace tvm
