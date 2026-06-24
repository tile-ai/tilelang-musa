#include "support/check.h"
#include <algorithm>
#include <map>
#include <numeric>
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/runtime/logging.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/target/target.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>
#include <functional>
#include <limits>
#include <queue>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/operator.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "backend/common/target_utils.h"
#include "common/bind_utils.h"
#include "common/pipeline_utils.h"
#include "tvm/ir/expr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

class BufferRegionCollector : public StmtExprVisitor {
public:
  BufferRegionCollector(Map<Var, Buffer> buffer_data_to_buffer, Target target);

  Array<BufferRegion> GetReads() const;
  Array<BufferRegion> GetWrites() const;
  bool GetGlobalCopyPattern() const;
  bool GetTmaCopyPattern() const;
  bool HasNonCopyTileOp() const;

private:
  static bool IsGlobalLikeBuffer(const Buffer &buffer);
  Optional<Buffer> TryGetBufFromAccessPtr(const PrimExpr &expr) const;

  void HandleTileOp(const TileOperator &tile_op);
  void VisitStmt_(const BufferStoreNode *op) final;
  void VisitExpr_(const BufferLoadNode *op) final;
  void VisitExpr_(const CallNode *op) final;
  void VisitStmt_(const IfThenElseNode *op) final;

  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  Array<BufferRegion> reads_;
  Array<BufferRegion> writes_;
  bool is_global_read_ = false;
  bool is_global_copy_pattern_ = false;
  bool is_tma_copy_ = false;
  bool has_non_copy_tile_op_ = false;
  bool within_condition_expr_ = false;
};

/*!
 * \brief Check whether two regions have intersections.
 * \param region1 The first region.
 * \param region2 The second region.
 * \return Whether region1 and region2 have intersections.
 */
bool MayConflict(const Region &region1, const Region &region2) {
  ICHECK(region1.size() == region2.size());
  for (size_t i = 0; i < region1.size(); i++) {
    Range dim1 = region1[i];
    Range dim2 = region2[i];
    auto int_set1 = arith::IntSet::FromRange(dim1);
    auto int_set2 = arith::IntSet::FromRange(dim2);
    if (arith::Intersect({int_set1, int_set2}).IsNothing()) {
      return false;
    }
  }
  return true;
}

BufferRegionCollector::BufferRegionCollector(
    Map<Var, Buffer> buffer_data_to_buffer, Target target)
    : buffer_data_to_buffer_(buffer_data_to_buffer), target_(target) {}

Array<BufferRegion> BufferRegionCollector::GetReads() const { return reads_; }

Array<BufferRegion> BufferRegionCollector::GetWrites() const { return writes_; }

bool BufferRegionCollector::GetGlobalCopyPattern() const {
  return is_global_copy_pattern_;
}

bool BufferRegionCollector::GetTmaCopyPattern() const { return is_tma_copy_; }

bool BufferRegionCollector::HasNonCopyTileOp() const {
  return has_non_copy_tile_op_;
}

bool BufferRegionCollector::IsGlobalLikeBuffer(const Buffer &buffer) {
  return IsGlobalBuffer(buffer) || (buffer.defined() && buffer.scope().empty());
}

Optional<Buffer>
BufferRegionCollector::TryGetBufFromAccessPtr(const PrimExpr &expr) const {
  const auto *call = expr.as<CallNode>();
  if (!call) {
    return Optional<Buffer>();
  }
  if (call->op.same_as(builtin::tvm_access_ptr())) {
    if (call->args.size() <= 1) {
      return Optional<Buffer>();
    }
    const auto *var = call->args[1].as<VarNode>();
    if (!var) {
      return Optional<Buffer>();
    }
    auto it = buffer_data_to_buffer_.find(GetRef<Var>(var));
    return it == buffer_data_to_buffer_.end() ? Optional<Buffer>()
                                               : Optional<Buffer>((*it).second);
  }
  if (call->op.same_as(tl::access_ptr()) && !call->args.empty()) {
    if (const auto *load = call->args[0].as<BufferLoadNode>()) {
      return load->buffer;
    }
  }
  return Optional<Buffer>();
}

void BufferRegionCollector::HandleTileOp(const TileOperator &tile_op) {
  if (tile_op.as<RegionOpNode>()) {
    return;
  }
  if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
    BufferRegionCollector nested(buffer_data_to_buffer_, target_);
    nested(parallel->GetRoot());
    reads_.insert(reads_.end(), nested.GetReads().begin(),
                  nested.GetReads().end());
    writes_.insert(writes_.end(), nested.GetWrites().begin(),
                   nested.GetWrites().end());
    is_global_copy_pattern_ =
        is_global_copy_pattern_ || nested.GetGlobalCopyPattern();
    is_tma_copy_ = is_tma_copy_ || nested.GetTmaCopyPattern();
    has_non_copy_tile_op_ = has_non_copy_tile_op_ || nested.HasNonCopyTileOp();
    return;
  }
  AccessRegions access = tile_op->GetAccessRegions();
  reads_.insert(reads_.end(), access.reads.begin(), access.reads.end());
  writes_.insert(writes_.end(), access.writes.begin(), access.writes.end());
  if (const auto *copy = tile_op.as<CopyNode>()) {
    if (IsGlobalLikeBuffer(copy->src) && IsSharedBuffer(copy->dst)) {
      is_global_copy_pattern_ = true;
    }
  }
  // Im2Col always uses TMA on Hopper.
  if (const auto *im2col = tile_op.as<Im2ColOpNode>()) {
    if (IsGlobalLikeBuffer(im2col->src_) && IsSharedBuffer(im2col->dst_)) {
      is_global_copy_pattern_ = true;
      if (TargetIsHopper(target_)) {
        is_tma_copy_ = true;
      }
    }
    return;
  }
  if (!tile_op.as<CopyNode>()) {
    has_non_copy_tile_op_ = true;
  }
}

void BufferRegionCollector::VisitStmt_(const BufferStoreNode *op) {
  Buffer store_buffer = op->buffer;
  Array<PrimExpr> indices = op->indices;
  // convert indices to region
  Array<Range> region;
  for (const auto &index : indices) {
    region.push_back(Range::FromMinExtent(index, 1));
  }
  auto store_region = BufferRegion(store_buffer, region);
  writes_.push_back(store_region);

  is_global_read_ = false;
  this->VisitExpr(op->value);
  if (is_global_read_ && IsSharedBuffer(store_buffer)) {
    is_global_copy_pattern_ = true;
  }
  is_global_read_ = false;
}

void BufferRegionCollector::VisitExpr_(const BufferLoadNode *op) {
  auto load_buffer = op->buffer;
  Array<PrimExpr> indices = op->indices;
  // convert indices to region
  Array<Range> region;
  for (const auto &index : indices) {
    region.push_back(Range::FromMinExtent(index, 1));
  }
  auto load_region = BufferRegion(load_buffer, region);
  reads_.push_back(load_region);

  if (IsGlobalLikeBuffer(op->buffer) && !within_condition_expr_) {
    // skip condition expr of if_then_else node
    // shared[i] = T.if_then_else(global[i] < n, register_a[i], register_b[i])
    // is not a global read shared[i] = T.if_then_else(global[i] < n,
    // global_a[i], global_b[i]) is a global read
    is_global_read_ = true;
  }
}

void BufferRegionCollector::VisitExpr_(const CallNode *op) {
  if (auto tile_op = ParseOperator(GetRef<Call>(op)); tile_op.defined()) {
    HandleTileOp(tile_op);
    StmtExprVisitor::VisitExpr_(op);
    return;
  }
  if (op->op.same_as(builtin::address_of())) {
    BufferRegion buffer_region;
    if (const auto *load = op->args[0].as<BufferLoadNode>()) {
      buffer_region = BufferRegion::FullRegion(load->buffer);
    } else if (const auto *var_node = op->args[0].as<VarNode>()) {
      Var data_var = GetRef<Var>(var_node);
      auto it = buffer_data_to_buffer_.find(data_var);
      if (it != buffer_data_to_buffer_.end()) {
        buffer_region = BufferRegion::FullRegion((*it).second);
      }
    }
    if (buffer_region.defined()) {
      // because we only care about the buffer itself instead of indices
      reads_.push_back(buffer_region);
    }
  } else if (op->op.same_as(builtin::tvm_access_ptr())) {
    const VarNode *buffer_var = op->args[1].as<VarNode>();
    ICHECK(buffer_var);
    auto it = buffer_data_to_buffer_.find(GetRef<Var>(buffer_var));
    if (it != buffer_data_to_buffer_.end()) {
      const Buffer &buffer = (*it).second;
      const BufferRegion buffer_region = BufferRegion::FullRegion(buffer);
      // because we only care about the buffer itself instead of indices
      reads_.push_back(buffer_region);
    }
  } else if (op->op.same_as(builtin::ptx_cp_async()) ||
             op->op.same_as(tl::ptx_cp_async())) {
    if (op->args.size() >= 2) {
      auto dst_buf = TryGetBufFromAccessPtr(op->args[0]);
      auto src_buf = TryGetBufFromAccessPtr(op->args[1]);
      if (src_buf.defined()) {
        reads_.push_back(BufferRegion::FullRegion(src_buf.value()));
      }
      if (dst_buf.defined()) {
        writes_.push_back(BufferRegion::FullRegion(dst_buf.value()));
      }
      if (src_buf.defined() && dst_buf.defined() &&
          IsGlobalLikeBuffer(src_buf.value()) &&
          IsSharedBuffer(dst_buf.value())) {
        is_global_copy_pattern_ = true;
      }
    }
    if (op->args.size() == 4) {
      if (auto dst_buf = TryGetBufFromAccessPtr(op->args[0])) {
        reads_.push_back(BufferRegion::FullRegion(dst_buf.value()));
      }
      this->VisitExpr(op->args[3]);
    }
    return;
  } else if (op->op.same_as(builtin::if_then_else())) {
    within_condition_expr_ = true;
    this->VisitExpr(op->args[0]);
    within_condition_expr_ = false;
    for (auto i = 1; i < op->args.size(); i++) {
      this->VisitExpr(op->args[i]);
    }
  } else {
    StmtExprVisitor::VisitExpr_(op);
  }
}

void BufferRegionCollector::VisitStmt_(const IfThenElseNode *op) {
  within_condition_expr_ = true;
  this->VisitExpr(op->condition);
  within_condition_expr_ = false;
  this->VisitStmt(op->then_case);
  if (op->else_case.defined()) {
    within_condition_expr_ = true;
    this->VisitStmt(op->else_case.value());
    within_condition_expr_ = false;
  }
}

class PipelinePlanningBodyAnalyzer {
public:
  PipelinePlanningBodyAnalyzer(Map<Var, Buffer> buffer_data_to_buffer,
                               Target target)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        target_(std::move(target)) {}

  std::pair<Array<BufferRegion>, Array<BufferRegion>>
  CollectStmtAccessRegions(const Stmt &stmt) const {
    SBlock block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                 /*name_hint=*/"", /*body*/ stmt);
    auto collector = BufferRegionCollector(buffer_data_to_buffer_, target_);
    collector(block);
    return {collector.GetReads(), collector.GetWrites()};
  }

  BufferSet CollectPipelineWriteBuffers(const Array<Stmt> &stmts) const {
    BufferSet write_buffers;
    for (const Stmt &stmt : stmts) {
      auto [_, writes] = CollectStmtAccessRegions(stmt);
      for (const BufferRegion &write : writes) {
        write_buffers.insert(write->buffer);
      }
    }
    return write_buffers;
  }

  bool
  IsReplayableScalarBindStmt(const Stmt &stmt,
                             const BufferSet &pipeline_write_buffers) const {
    auto [reads, _] = CollectStmtAccessRegions(stmt);
    return IsReplayableScalarBind(stmt, reads, pipeline_write_buffers);
  }

  struct ScheduledStmtAnalysis {
    size_t original_stmt_count{0};
    size_t stage_stmt_count{0};
    Array<Stmt> scheduled_stmts;
    std::vector<size_t> scheduled_indices;
    std::vector<size_t> scheduled_stage_indices;
    Array<Integer> replayable_bind_mask;
    bool has_bind_stmt{false};
  };

  ScheduledStmtAnalysis AnalyzeScheduledStmts(const Array<Stmt> &stmts) const {
    BufferSet pipeline_write_buffers = CollectPipelineWriteBuffers(stmts);
    ScheduledStmtAnalysis analysis;
    analysis.original_stmt_count = stmts.size();
    analysis.replayable_bind_mask.reserve(stmts.size());
    size_t stage_stmt_index = 0;
    for (size_t i = 0; i < stmts.size(); ++i) {
      const Stmt &stmt = stmts[i];
      if (IsPipelineDeclarationStmt(stmt)) {
        continue;
      }
      analysis.has_bind_stmt = analysis.has_bind_stmt || stmt.as<BindNode>();
      bool replayable =
          IsReplayableScalarBindStmt(stmt, pipeline_write_buffers);
      analysis.replayable_bind_mask.push_back(Integer(replayable ? 1 : 0));
      if (replayable) {
        ++stage_stmt_index;
        continue;
      }
      analysis.scheduled_indices.push_back(i);
      analysis.scheduled_stage_indices.push_back(stage_stmt_index);
      analysis.scheduled_stmts.push_back(stmt);
      ++stage_stmt_index;
    }
    analysis.stage_stmt_count = stage_stmt_index;
    return analysis;
  }

  Array<Integer> FilterAnnotationsForScheduledStmts(
      const Array<Integer> &annotations,
      const ScheduledStmtAnalysis &analysis) const {
    if (annotations.size() == analysis.scheduled_stmts.size()) {
      return annotations;
    }

    Array<Integer> filtered;
    if (annotations.size() == analysis.stage_stmt_count) {
      for (size_t index : analysis.scheduled_stage_indices) {
        filtered.push_back(annotations[index]);
      }
    } else {
      ICHECK_EQ(annotations.size(), analysis.original_stmt_count)
          << "PipelinePlanning: expected pipeline annotation size to match "
             "the scheduled statement count, executable statement count, or "
             "original statement count";
      for (size_t index : analysis.scheduled_indices) {
        filtered.push_back(annotations[index]);
      }
    }
    ICHECK_EQ(filtered.size(), analysis.scheduled_stmts.size());
    return filtered;
  }

  class SeqStmtFlattener : public StmtFunctor<Array<Stmt>(const Stmt &)> {
  public:
    using Base = StmtFunctor<Array<Stmt>(const Stmt &)>;

    static Array<Stmt> Flatten(const Array<Stmt> &stmts) {
      SeqStmtFlattener flattener;
      Array<Stmt> flattened;
      for (const Stmt &stmt : stmts) {
        Array<Stmt> nested = flattener(stmt);
        flattened.insert(flattened.end(), nested.begin(), nested.end());
      }
      return flattened;
    }

    Array<Stmt> VisitStmt(const Stmt &stmt) final {
      if (!stmt.as<SeqStmtNode>()) {
        return Array<Stmt>{stmt};
      }
      return Base::VisitStmt(stmt);
    }

    Array<Stmt> VisitStmt_(const SeqStmtNode *op) final {
      Array<Stmt> flattened;
      for (const Stmt &stmt : op->seq) {
        Array<Stmt> nested = VisitStmt(stmt);
        flattened.insert(flattened.end(), nested.begin(), nested.end());
      }
      return flattened;
    }

    Array<Stmt> VisitStmtDefault_(const Object *) final {
      return Array<Stmt>();
    }
  };

private:
  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
};

/*! \brief Information about a pipeline stage
 *
 * \param reads Array of buffer regions read by this stage
 * \param writes Array of buffer regions written by this stage
 * \param original_stmt_index Original position of this stage in the pipeline
 * before reordering \param order Current position of this stage in the
 * pipeline after reordering (-1 if not yet assigned) \param stage Pipeline
 * stage number this operation belongs to (-1 if not yet assigned) \param
 * copy_stage Whether this stage is a memory copy operation \param
 * last_use_stmt_index Index of the last statement (in original order) that
 * uses the results of this stage (-1 if not yet determined). This field is
 * crucial for pipeline optimization:
 * - For copy stages: indicates the index of the last statement that reads
 * from the copied data, helping determine optimal placement of copy
 * operations
 * - Used to ensure copy operations are scheduled before their consumers
 * - A value of -1 means no subsequent statement uses this stage's output
 * - This information enables better pipeline scheduling by minimizing data
 *   dependencies and maximizing parallelism
 */
struct PipelineStageInfo {
  Array<BufferRegion> reads, writes;
  std::unordered_set<const VarNode *> scalar_defs;
  std::unordered_set<const VarNode *> scalar_uses;
  int original_stmt_index{};
  int order = -1, stage = -1;
  bool copy_stage = false;
  bool tma_copy = false; // true if this copy stage uses TMA (not cp.async)
  bool conditional_execution = false;
  bool producer_for_copy = false;
  bool cp_async_commit_stage = false;
  int cp_async_call_count = 0;
  int cp_async_commit_count = 0;
  int cp_async_wait_count = 0;
  int cp_async_wait_min_inflight = std::numeric_limits<int>::max();
  bool cp_async_wait_has_dynamic = false;
  int cp_async_group = -1;
  int last_use_stmt_index =
      -1; // Initialized to -1, indicating no consumers found yet

public:
  bool IsFirstStage() const {
    return copy_stage || producer_for_copy || cp_async_commit_stage;
  }
  bool IsCopyStage() const { return copy_stage; }
  bool IsTmaCopy() const { return tma_copy; }
  bool IsProducerForCopy() const { return producer_for_copy; }
  bool IsCpAsyncCommitStage() const { return cp_async_commit_stage; }
  bool HasCpAsyncCall() const { return cp_async_call_count > 0; }
  bool HasCpAsyncCommit() const { return cp_async_commit_count > 0; }
  bool HasCpAsyncWait() const { return cp_async_wait_count > 0; }
  bool IsLastUseStmtIndexValid() const {
    return last_use_stmt_index != -1;
  }
};

class PipelineStageAnalyzer {
public:
  PipelineStageAnalyzer(Map<Var, Buffer> buffer_data_to_buffer, Target target,
                        bool use_async_copy)
      : buffer_data_to_buffer_(std::move(buffer_data_to_buffer)),
        target_(std::move(target)), use_async_copy_(use_async_copy) {}

  class ScalarUseDefCollector : public StmtExprVisitor {
  public:
    static std::pair<std::unordered_set<const VarNode *>,
                     std::unordered_set<const VarNode *>>
    Collect(const Stmt &stmt) {
      ScalarUseDefCollector collector;
      collector(stmt);
      return {std::move(collector.scalar_defs_),
              std::move(collector.scalar_uses_)};
    }

  private:
    void VisitStmt_(const BindNode *op) final {
      this->VisitExpr(op->value);
      scalar_defs_.insert(op->var.get());
    }

    void VisitExpr_(const VarNode *op) final { scalar_uses_.insert(op); }

    std::unordered_set<const VarNode *> scalar_defs_;
    std::unordered_set<const VarNode *> scalar_uses_;
  };

  struct AsyncIntrinInfo {
    int cp_async_call_count = 0;
    int cp_async_commit_count = 0;
    int cp_async_wait_count = 0;
    int cp_async_wait_min_inflight = std::numeric_limits<int>::max();
    bool cp_async_wait_has_dynamic = false;
  };

  AsyncIntrinInfo AnalyzeAsyncIntrinsics(const Stmt &stmt) {
    AsyncIntrinInfo info;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      const auto *call = node.as<CallNode>();
      if (!call) {
        return;
      }
      if (call->op.same_as(builtin::ptx_cp_async()) ||
          call->op.same_as(tl::ptx_cp_async())) {
        ++info.cp_async_call_count;
      } else if (call->op.same_as(builtin::ptx_commit_group())) {
        ++info.cp_async_commit_count;
      } else if (call->op.same_as(builtin::ptx_wait_group())) {
        ++info.cp_async_wait_count;
        if (!call->args.empty()) {
          if (const int64_t *imm = as_const_int(call->args[0])) {
            info.cp_async_wait_min_inflight = std::min(
                info.cp_async_wait_min_inflight, static_cast<int>(*imm));
          } else {
            info.cp_async_wait_has_dynamic = true;
          }
        } else {
          info.cp_async_wait_has_dynamic = true;
        }
      }
    });
    return info;
  }

  bool MayBeConditionallyExecuted(const Stmt &stmt) const {
    bool conditional = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (conditional) {
        return;
      }
      if (const auto *if_then_else = node.as<IfThenElseNode>()) {
        conditional = true;
        return;
      }
      if (const auto *realize = node.as<SBlockRealizeNode>()) {
        if (!is_one(realize->predicate)) {
          conditional = true;
        }
      }
    });
    return conditional;
  }

  bool IsAsyncProducerCandidate(const PipelineStageInfo &pinfo) const {
    if (pinfo.conditional_execution) {
      return false;
    }
    if (pinfo.IsTmaCopy()) {
      return false;
    }
    if (pinfo.HasCpAsyncWait()) {
      return false;
    }
    if (pinfo.HasCpAsyncCommit() && !pinfo.HasCpAsyncCall()) {
      return false;
    }
    return pinfo.IsCopyStage() || pinfo.HasCpAsyncCall();
  }

  bool IsPureCopyStmt(const Stmt &stmt) const {
    auto is_global_like_buffer = [](const Buffer &buffer) {
      return IsGlobalBuffer(buffer) ||
             (buffer.defined() && buffer.scope().empty());
    };
    auto is_pure_raw_copy_value = [&](const PrimExpr &expr,
                                      const auto &self) -> bool {
      if (const auto *load = expr.as<BufferLoadNode>()) {
        return is_global_like_buffer(load->buffer);
      }
      if (const auto *cast = expr.as<CastNode>()) {
        return self(cast->value, self);
      }
      return false;
    };

    bool saw_copy = false;
    bool saw_non_copy_tile_op = false;
    bool saw_non_copy_buffer_store = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (saw_non_copy_tile_op || saw_non_copy_buffer_store) {
        return;
      }
      if (const auto *store = node.as<BufferStoreNode>()) {
        saw_copy = true;
        if ((!IsSharedBuffer(store->buffer) &&
             !IsLocalBuffer(store->buffer, /*allow_var=*/true)) ||
            !is_pure_raw_copy_value(store->value, is_pure_raw_copy_value)) {
          saw_non_copy_buffer_store = true;
        }
        return;
      }
      const auto *call = node.as<CallNode>();
      if (call == nullptr) {
        return;
      }
      auto tile_op = ParseOperator(GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      if (tile_op.as<RegionOpNode>()) {
        return;
      }
      if (const auto *parallel = tile_op.as<ParallelOpNode>()) {
        if (IsPureCopyStmt(parallel->GetRoot())) {
          saw_copy = true;
        } else {
          saw_non_copy_tile_op = true;
        }
        return;
      }
      if (tile_op.as<CopyNode>() || tile_op.as<Im2ColOpNode>()) {
        saw_copy = true;
      } else {
        saw_non_copy_tile_op = true;
      }
    });
    return saw_copy && !saw_non_copy_tile_op && !saw_non_copy_buffer_store;
  }

  Optional<TileOperator> GetSinglePureCopyTileOp(const Stmt &stmt) const {
    Optional<TileOperator> copy_tile_op;
    bool saw_non_copy_tile_op = false;
    bool saw_multiple_copy_ops = false;
    PostOrderVisit(stmt, [&](const ObjectRef &node) {
      if (saw_non_copy_tile_op || saw_multiple_copy_ops) {
        return;
      }
      const auto *call = node.as<CallNode>();
      if (call == nullptr) {
        return;
      }
      auto tile_op = ParseOperator(GetRef<Call>(call));
      if (!tile_op.defined()) {
        return;
      }
      if (tile_op.as<RegionOpNode>()) {
        return;
      }
      if (tile_op.as<CopyNode>() || tile_op.as<Im2ColOpNode>()) {
        if (copy_tile_op.defined()) {
          saw_multiple_copy_ops = true;
          copy_tile_op = Optional<TileOperator>();
        } else {
          copy_tile_op = tile_op;
        }
      } else {
        saw_non_copy_tile_op = true;
        copy_tile_op = Optional<TileOperator>();
      }
    });
    if (saw_non_copy_tile_op || saw_multiple_copy_ops) {
      return Optional<TileOperator>();
    }
    return copy_tile_op;
  }

  static bool IsGlobalLikeBuffer(const Buffer &buffer) {
    return IsGlobalBuffer(buffer) ||
           (buffer.defined() && buffer.scope().empty());
  }

  void ClassifyCopyLikeStage(const Stmt &stmt, PipelineStageInfo *pinfo) const {
    ICHECK(pinfo != nullptr);
    if (pinfo->conditional_execution) {
      return;
    }

    if (pinfo->copy_stage) {
      return;
    }

    if (pinfo->HasCpAsyncCall()) {
      pinfo->copy_stage = true;
      return;
    }

    auto copy_tile_op = GetSinglePureCopyTileOp(stmt);
    if (!copy_tile_op.defined()) {
      return;
    }

    if (const auto *copy = copy_tile_op.value().as<CopyNode>()) {
      if (!IsGlobalLikeBuffer(copy->src) || !IsSharedBuffer(copy->dst)) {
        return;
      }
      pinfo->copy_stage = true;
      return;
    }

    if (const auto *im2col = copy_tile_op.value().as<Im2ColOpNode>()) {
      if (!IsGlobalLikeBuffer(im2col->src_) || !IsSharedBuffer(im2col->dst_)) {
        return;
      }
      pinfo->copy_stage = true;
      pinfo->tma_copy = TargetIsHopper(target_);
    }
  }

  void AnalyzeCopyLastUse(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    for (auto &pinfo : *pipeline_stage_infos) {
      if (!pinfo.IsFirstStage()) {
        continue;
      }

      for (int i = pinfo.original_stmt_index + 1;
           i < static_cast<int>(pipeline_stage_infos->size()); ++i) {
        for (const BufferRegion &read : (*pipeline_stage_infos)[i].reads) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == read->buffer &&
                                    MayConflict(r->region, read->region);
                           }) != pinfo.writes.end()) {
            pinfo.last_use_stmt_index = std::max(pinfo.last_use_stmt_index, i);
          }
        }

        if (!pinfo.IsCopyStage() ||
            (pinfo.cp_async_group >= 0 &&
             pinfo.cp_async_group == (*pipeline_stage_infos)[i].cp_async_group)) {
          continue;
        }

        for (const BufferRegion &write : (*pipeline_stage_infos)[i].writes) {
          if (std::find_if(pinfo.writes.begin(), pinfo.writes.end(),
                           [&](const BufferRegion &r) {
                             return r->buffer == write->buffer &&
                                    MayConflict(r->region, write->region);
                           }) != pinfo.writes.end()) {
            LOG(FATAL) << "Pipeline planning error: Multiple writes to "
                          "overlapping buffer regions detected. "
                       << "Stage " << pinfo.original_stmt_index << " and stage "
                       << i << " are both writing to buffer '"
                       << write->buffer->name
                       << "' with overlapping regions. This is not supported "
                          "in pipeline planning.";
          }
        }
      }
    }
  }

  void PropagateBufferProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    struct CopyStageDependencyReadsManager {
      std::vector<BufferRegion> regions;

      void AddUnique(const BufferRegion &region) {
        for (const BufferRegion &copy_read : regions) {
          if (region->buffer.same_as(copy_read->buffer)) {
            return;
          }
        }
        regions.push_back(region);
      }

      bool Contains(const BufferRegion &region) const {
        for (const BufferRegion &copy_read : regions) {
          if (region->buffer.same_as(copy_read->buffer)) {
            return true;
          }
        }
        return false;
      }

      size_t Size() const { return regions.size(); }
    };

    CopyStageDependencyReadsManager copy_stage_dependency_reads_mgr;

    for (const auto &pinfo : *pipeline_stage_infos) {
      if (pinfo.IsCopyStage()) {
        for (const BufferRegion &read : pinfo.reads) {
          copy_stage_dependency_reads_mgr.AddUnique(read);
        }
      }
    }

    const size_t max_iterations = (pipeline_stage_infos->size() * 4) + 16;
    size_t iter_count = 0;

    for (auto &pinfo : *pipeline_stage_infos) {
      if (!pinfo.IsCopyStage()) {
        continue;
      }
      auto original_copy_stmt_index = pinfo.original_stmt_index;
      bool updated = true;
      while (updated) {
        updated = false;
        for (auto &pinfo_inner : *pipeline_stage_infos) {
          if (pinfo_inner.IsCopyStage()) {
            continue;
          }
          if (pinfo_inner.original_stmt_index >= original_copy_stmt_index) {
            break;
          }

          bool should_prepare = false;
          for (const BufferRegion &write : pinfo_inner.writes) {
            if (copy_stage_dependency_reads_mgr.Contains(write)) {
              should_prepare = true;
              break;
            }
          }
          if (should_prepare && !pinfo_inner.IsProducerForCopy()) {
            pinfo_inner.producer_for_copy = true;
            updated = true;
          }
          if (should_prepare) {
            for (const BufferRegion &read : pinfo_inner.reads) {
              size_t before = copy_stage_dependency_reads_mgr.Size();
              copy_stage_dependency_reads_mgr.AddUnique(read);
              if (copy_stage_dependency_reads_mgr.Size() > before) {
                updated = true;
              }
            }
          }
        }
        iter_count++;
        if (iter_count > max_iterations) {
          LOG(FATAL)
              << "Pipeline planning: Exceeded maximum iterations ("
              << max_iterations << ") in copy stage dependency propagation. "
              << "This may indicate a cyclic or pathological dependency graph.";
        }
      }
    }
  }

  std::unordered_map<const VarNode *, int> BuildScalarDefMap(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    std::unordered_map<const VarNode *, int> scalar_def_to_stmt;
    for (int i = 0; i < static_cast<int>(pipeline_stage_infos.size()); ++i) {
      for (const VarNode *var : pipeline_stage_infos[i].scalar_defs) {
        scalar_def_to_stmt.emplace(var, i);
      }
    }
    return scalar_def_to_stmt;
  }

  void PropagateScalarProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    auto scalar_def_to_stmt = BuildScalarDefMap(*pipeline_stage_infos);
    const size_t max_iterations = (pipeline_stage_infos->size() * 4) + 16;
    size_t iter_count = 0;
    bool updated = true;

    auto update_producer = [](PipelineStageInfo *producer,
                              int consumer_last_use) -> bool {
      if (consumer_last_use < 0) {
        return false;
      }
      bool changed = false;
      if (!producer->producer_for_copy) {
        producer->producer_for_copy = true;
        producer->last_use_stmt_index = consumer_last_use;
        changed = true;
      } else if (!producer->IsLastUseStmtIndexValid() ||
                 consumer_last_use < producer->last_use_stmt_index) {
        producer->last_use_stmt_index = consumer_last_use;
        changed = true;
      }
      return changed;
    };

    while (updated) {
      updated = false;
      for (int consumer_idx = 0;
           consumer_idx < static_cast<int>(pipeline_stage_infos->size());
           ++consumer_idx) {
        const auto &consumer = (*pipeline_stage_infos)[consumer_idx];
        if (!(consumer.IsFirstStage() && consumer.IsLastUseStmtIndexValid())) {
          continue;
        }
        for (const VarNode *var : consumer.scalar_uses) {
          auto it = scalar_def_to_stmt.find(var);
          if (it == scalar_def_to_stmt.end() || it->second == consumer_idx) {
            continue;
          }
          auto &producer = (*pipeline_stage_infos)[it->second];
          if (producer.IsCopyStage() || producer.IsCpAsyncCommitStage()) {
            continue;
          }
          updated |= update_producer(&producer, consumer.last_use_stmt_index);
        }
      }
      if (++iter_count > max_iterations) {
        LOG(FATAL) << "Pipeline planning: Exceeded maximum iterations while "
                      "propagating scalar producers for copy stages.";
      }
    }
  }

  void ValidateScalarDependencies(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    auto scalar_def_to_stmt = BuildScalarDefMap(pipeline_stage_infos);
    for (int consumer_idx = 0;
         consumer_idx < static_cast<int>(pipeline_stage_infos.size());
         ++consumer_idx) {
      const auto &consumer = pipeline_stage_infos[consumer_idx];
      for (const VarNode *var : consumer.scalar_uses) {
        auto it = scalar_def_to_stmt.find(var);
        if (it == scalar_def_to_stmt.end() || it->second == consumer_idx) {
          continue;
        }
        const auto &producer = pipeline_stage_infos[it->second];
        ICHECK_LE(producer.stage, consumer.stage)
            << "Pipeline planning error: scalar dependency from statement "
            << producer.original_stmt_index << " to statement "
            << consumer.original_stmt_index
            << " crosses from a later stage to an earlier stage.";
        if (producer.stage == consumer.stage) {
          ICHECK_LT(producer.order, consumer.order)
              << "Pipeline planning error: scalar dependency from statement "
              << producer.original_stmt_index << " to statement "
              << consumer.original_stmt_index
              << " is reordered within the same pipeline stage.";
        }
      }
    }
  }

  bool EmitAsyncAnnotations(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos,
      const std::vector<std::vector<int>> &explicit_async_groups,
      Map<String, Any> *annotations) const {
    if (!TargetHasAsyncCopy(target_) || !use_async_copy_) {
      return false;
    }

    std::vector<int> async_group_ids(pipeline_stage_infos.size(), -1);
    std::vector<int> stmt_indices_by_order(pipeline_stage_infos.size());
    std::iota(stmt_indices_by_order.begin(), stmt_indices_by_order.end(), 0);
    std::stable_sort(stmt_indices_by_order.begin(), stmt_indices_by_order.end(),
                     [&](int lhs, int rhs) {
                       if (pipeline_stage_infos[lhs].order !=
                           pipeline_stage_infos[rhs].order) {
                         return pipeline_stage_infos[lhs].order <
                                pipeline_stage_infos[rhs].order;
                       }
                       return lhs < rhs;
                     });

    int next_async_group_id = 0;
    for (const auto &group_stmt_indices : explicit_async_groups) {
      bool emitted_group = false;
      for (int stmt_idx : group_stmt_indices) {
        if (stmt_idx < 0 ||
            stmt_idx >= static_cast<int>(pipeline_stage_infos.size()) ||
            !IsAsyncProducerCandidate(pipeline_stage_infos[stmt_idx])) {
          continue;
        }
        async_group_ids[stmt_idx] = next_async_group_id;
        emitted_group = true;
      }
      if (emitted_group) {
        ++next_async_group_id;
      }
    }
    std::map<std::pair<int, int>, int> implicit_group_ids;
    for (int stmt_idx : stmt_indices_by_order) {
      const auto &pinfo = pipeline_stage_infos[stmt_idx];
      if (!IsAsyncProducerCandidate(pinfo) || async_group_ids[stmt_idx] != -1) {
        continue;
      }
      auto key = std::make_pair(pinfo.stage, pinfo.last_use_stmt_index);
      auto [it, inserted] =
          implicit_group_ids.emplace(key, next_async_group_id);
      if (inserted) {
        ++next_async_group_id;
      }
      async_group_ids[stmt_idx] = it->second;
    }

    if (next_async_group_id == 0) {
      return false;
    }

    std::vector<Integer> async_producers;
    std::vector<Integer> async_producer_groups;
    async_producers.reserve(pipeline_stage_infos.size());
    async_producer_groups.reserve(pipeline_stage_infos.size());
    std::unordered_set<int> async_stage_ids;
    for (size_t i = 0; i < pipeline_stage_infos.size(); ++i) {
      bool is_async_producer = async_group_ids[i] != -1;
      async_producers.push_back(Integer(is_async_producer ? 1 : 0));
      async_producer_groups.push_back(Integer(async_group_ids[i]));
      if (is_async_producer) {
        async_stage_ids.insert(pipeline_stage_infos[i].stage);
      }
    }

    annotations->Set(kPipelineAsyncProducers, Array<Integer>(async_producers));
    annotations->Set(kPipelineAsyncProducerGroups,
                     Array<Integer>(async_producer_groups));

    std::vector<int> sorted_async_stage_ids(async_stage_ids.begin(),
                                            async_stage_ids.end());
    std::sort(sorted_async_stage_ids.begin(), sorted_async_stage_ids.end());
    std::vector<Integer> async_stages;
    async_stages.reserve(sorted_async_stage_ids.size());
    for (int stage_id : sorted_async_stage_ids) {
      async_stages.push_back(Integer(stage_id));
    }
    annotations->Set(s_tir::attr::software_pipeline_async_stages,
                     Array<Integer>(async_stages));
    return true;
  }

  bool EmitImplicitAsyncAnnotations(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos,
      Map<String, Any> *annotations) const {
    return EmitAsyncAnnotations(pipeline_stage_infos, {}, annotations);
  }

  void MaybeAnnotateLegacyAsyncPipelineLoop(const Array<Stmt> &pipeline_stmts,
                                            const Array<Integer> &order_array,
                                            const Array<Integer> &stage_array,
                                            Map<String, Any> *annotations) {
    if (!TargetHasAsyncCopy(target_) || !use_async_copy_) {
      return;
    }
    ICHECK_EQ(pipeline_stmts.size(), order_array.size());
    ICHECK_EQ(pipeline_stmts.size(), stage_array.size());

    std::vector<PipelineStageInfo> pipeline_stage_infos;
    pipeline_stage_infos.reserve(pipeline_stmts.size());
    for (size_t i = 0; i < pipeline_stmts.size(); ++i) {
      auto pinfo = MakePipelineStageInfo(pipeline_stmts[i], i);
      ClassifyCopyLikeStage(pipeline_stmts[i], &pinfo);
      pinfo.order = static_cast<int>(order_array[i]->value);
      pinfo.stage = static_cast<int>(stage_array[i]->value);
      if (!pinfo.IsCopyStage() && !pinfo.conditional_execution &&
          pinfo.stage == 0) {
        bool reads_global = false;
        bool writes_shared = false;
        for (const BufferRegion &read : pinfo.reads) {
          if (IsGlobalLikeBuffer(read->buffer)) {
            reads_global = true;
            break;
          }
        }
        for (const BufferRegion &write : pinfo.writes) {
          if (IsSharedBuffer(write->buffer)) {
            writes_shared = true;
            break;
          }
        }
        if (reads_global && writes_shared) {
          pinfo.copy_stage = true;
        }
      }
      pipeline_stage_infos.push_back(std::move(pinfo));
    }

    AnalyzeCopyLastUse(&pipeline_stage_infos);
    EmitImplicitAsyncAnnotations(pipeline_stage_infos, annotations);
  }

  PipelineStageInfo MakePipelineStageInfo(Stmt stmt, int idx) {
    SBlock block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
                 /*name_hint=*/"",
                 /*body*/ std::move(stmt));
    auto collector = BufferRegionCollector(buffer_data_to_buffer_, target_);
    collector(block);
    PipelineStageInfo pinfo;
    pinfo.reads = std::move(collector.GetReads());
    pinfo.writes = std::move(collector.GetWrites());
    auto [scalar_defs, scalar_uses] =
        ScalarUseDefCollector::Collect(block->body);
    pinfo.scalar_defs = std::move(scalar_defs);
    pinfo.scalar_uses = std::move(scalar_uses);
    auto async_info = AnalyzeAsyncIntrinsics(block->body);
    pinfo.cp_async_call_count = async_info.cp_async_call_count;
    pinfo.cp_async_commit_count = async_info.cp_async_commit_count;
    pinfo.cp_async_wait_count = async_info.cp_async_wait_count;
    pinfo.cp_async_wait_min_inflight = async_info.cp_async_wait_min_inflight;
    pinfo.cp_async_wait_has_dynamic = async_info.cp_async_wait_has_dynamic;
    pinfo.original_stmt_index = idx;
    pinfo.conditional_execution = MayBeConditionallyExecuted(block->body);
    bool pure_copy_stage =
        collector.GetGlobalCopyPattern() && IsPureCopyStmt(block->body);
    pinfo.copy_stage = pure_copy_stage;
    pinfo.tma_copy = pure_copy_stage && !pinfo.conditional_execution &&
                     collector.GetTmaCopyPattern();
    ClassifyCopyLikeStage(block->body, &pinfo);
    return pinfo;
  }

private:
  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  bool use_async_copy_{};
};

class PipelinePlanner : public StmtExprMutator {
public:
  static Stmt Substitute(const PrimFunc &f, bool use_async_copy = true) {
    PipelinePlanner substituter(use_async_copy);
    for (const auto &[_, buffer] : f->buffer_map) {
      substituter.buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    ICHECK(target.defined())
        << "Pipeline_Planning: Require the target attribute";
    substituter.target_ = target.value();
    return substituter.VisitStmt(f->body);
  }

private:
  PipelinePlanner() = default;
  PipelinePlanner(bool use_async_copy) : use_async_copy_(use_async_copy) {}

  PipelineStageAnalyzer MakeStageAnalyzer() const {
    return PipelineStageAnalyzer(buffer_data_to_buffer_, target_,
                                 use_async_copy_);
  }

  void AnalyzeCopyLastUse(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    MakeStageAnalyzer().AnalyzeCopyLastUse(pipeline_stage_infos);
  }

  void PropagateBufferProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    MakeStageAnalyzer().PropagateBufferProducersForCopy(pipeline_stage_infos);
  }

  void PropagateScalarProducersForCopy(
      std::vector<PipelineStageInfo> *pipeline_stage_infos) const {
    MakeStageAnalyzer().PropagateScalarProducersForCopy(pipeline_stage_infos);
  }

  void ValidateScalarDependencies(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos) const {
    MakeStageAnalyzer().ValidateScalarDependencies(pipeline_stage_infos);
  }

  void MaybeAnnotateLegacyAsyncPipelineLoop(const Array<Stmt> &pipeline_stmts,
                                            const Array<Integer> &order_array,
                                            const Array<Integer> &stage_array,
                                            Map<String, Any> *annotations) {
    MakeStageAnalyzer().MaybeAnnotateLegacyAsyncPipelineLoop(
        pipeline_stmts, order_array, stage_array, annotations);
  }

  void EmitImplicitAsyncAnnotations(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos,
      Map<String, Any> *annotations) const {
    MakeStageAnalyzer().EmitImplicitAsyncAnnotations(pipeline_stage_infos,
                                                     annotations);
  }

  void EmitAsyncAnnotations(
      const std::vector<PipelineStageInfo> &pipeline_stage_infos,
      const std::vector<std::vector<int>> &explicit_async_groups,
      Map<String, Any> *annotations) const {
    MakeStageAnalyzer().EmitAsyncAnnotations(
        pipeline_stage_infos, explicit_async_groups, annotations);
  }

  PipelineStageInfo MakePipelineStageInfo(Stmt stmt, int idx) {
    return MakeStageAnalyzer().MakePipelineStageInfo(std::move(stmt), idx);
  }

  using ScheduledStmtAnalysis =
      PipelinePlanningBodyAnalyzer::ScheduledStmtAnalysis;
  using SeqStmtFlattener = PipelinePlanningBodyAnalyzer::SeqStmtFlattener;

  PipelinePlanningBodyAnalyzer MakeBodyAnalyzer() const {
    return PipelinePlanningBodyAnalyzer(buffer_data_to_buffer_, target_);
  }

  ScheduledStmtAnalysis AnalyzeScheduledStmts(const Array<Stmt> &stmts) const {
    return MakeBodyAnalyzer().AnalyzeScheduledStmts(stmts);
  }

  Array<Integer> FilterAnnotationsForScheduledStmts(
      const Array<Integer> &annotations,
      const ScheduledStmtAnalysis &analysis) const {
    return MakeBodyAnalyzer().FilterAnnotationsForScheduledStmts(annotations,
                                                                 analysis);
  }

  Stmt VisitStmt_(const ForNode *loop) final {
    auto order_anno = loop->annotations.Get("tl_pipeline_order");
    auto stage_anno = loop->annotations.Get("tl_pipeline_stage");
    auto num_stages_anno = loop->annotations.Get("num_stages");
    if (order_anno && stage_anno) {
      auto order_array = Downcast<Array<Integer>>(order_anno.value());
      auto stage_array = Downcast<Array<Integer>>(stage_anno.value());

      Map<String, Any> annotations;
      for (const auto &[key, value] : loop->annotations) {
        if (key != "tl_pipeline_order" && key != "tl_pipeline_stage") {
          annotations.Set(key, value);
        }
      }
      if (TargetHasAsyncCopy(target_) && use_async_copy_) {
        // Legacy explicit stage/order annotations do not carry per-statement
        // async producer metadata yet, so keep the previous stage-level
        // behavior as a fallback for these loops.
        annotations.Set(s_tir::attr::software_pipeline_async_stages,
                        Array<Integer>{0});
      }
      Array<Stmt> pipeline_body_stmts = NormalizePipelineBody(loop->body);
      Array<Stmt> pipeline_stmts =
          SeqStmtFlattener::Flatten(pipeline_body_stmts);
      ScheduledStmtAnalysis analysis = AnalyzeScheduledStmts(pipeline_stmts);
      ICHECK(!analysis.scheduled_stmts.empty())
          << "PipelinePlanning: explicit pipeline annotations have no "
             "schedulable statements after removing replayable scalar Bind "
             "statements";
      Array<Integer> filtered_order_array =
          FilterAnnotationsForScheduledStmts(order_array, analysis);
      Array<Integer> filtered_stage_array =
          FilterAnnotationsForScheduledStmts(stage_array, analysis);
      annotations.Set(s_tir::attr::software_pipeline_order,
                      filtered_order_array);
      annotations.Set(s_tir::attr::software_pipeline_stage,
                      filtered_stage_array);
      if (pipeline_stmts.size() == pipeline_body_stmts.size()) {
        bool flatten_preserved_original_order = true;
        for (size_t i = 0; i < pipeline_stmts.size(); ++i) {
          if (!pipeline_stmts[i].same_as(pipeline_body_stmts[i])) {
            flatten_preserved_original_order = false;
            break;
          }
        }
        if (flatten_preserved_original_order && analysis.has_bind_stmt) {
          annotations.Set(kPipelineReplayableScalarBinds,
                          analysis.replayable_bind_mask);
        }
      }
      MaybeAnnotateLegacyAsyncPipelineLoop(analysis.scheduled_stmts,
                                           filtered_order_array,
                                           filtered_stage_array, &annotations);
      auto for_node = GetRef<For>(loop);
      auto *n = for_node.CopyOnWrite();
      n->annotations = annotations;
      n->body = MakePipelineBody(pipeline_body_stmts);
      return for_node;
    }

    if (!num_stages_anno)
      return StmtExprMutator::VisitStmt_(loop);
    int num_stages = num_stages_anno->as<IntImmNode>()->value;
    // Skip software pipelining on ROCm targets where async-copy pipelining
    // has not been validated.  Currently only gfx950 (CDNA4 / MI350) supports
    // the full HIP async-copy pipeline path.  gfx942 (CDNA3 / MI300X) has
    // async-copy hardware but the software pipeline for that target has not
    // been validated yet, so it falls back to a plain sequential loop as well.
    // RDNA targets have no async-copy support at all and also fall back.
    if (TargetIsRocm(target_) && !TargetIsGfx950(target_) && num_stages >= 1) {
      // Strip the "num_stages" annotation before recursing so that downstream
      // passes (InjectSoftwarePipeline, MultiVersionBufferRewriter, etc.) do
      // not treat this loop as pipelined.  Leaving the annotation in place
      // would cause those passes to multi-version shared buffers and inject
      // cp.async / barrier code that is incompatible with the plain sequential
      // execution path chosen here.
      auto stripped = GetRef<For>(loop);
      Map<String, Any> annotations;
      for (const auto &[key, value] : loop->annotations) {
        if (key != "num_stages") {
          annotations.Set(key, value);
        }
      }
      stripped.CopyOnWrite()->annotations = annotations;
      return StmtExprMutator::VisitStmt_(stripped.get());
    }
    Array<Stmt> pipeline_body_stmts = NormalizePipelineBody(loop->body);

    ICHECK(num_stages >= 1);
    ICHECK(loop->kind == ForKind::kSerial);

    // Flatten nested SeqStmts so pipeline planning can assign stages to the
    // normalized top-level statement list.
    Array<Stmt> flat_stmts = SeqStmtFlattener::Flatten(pipeline_body_stmts);
    ScheduledStmtAnalysis analysis = AnalyzeScheduledStmts(flat_stmts);
    ICHECK(!analysis.scheduled_stmts.empty())
        << "PipelinePlanning: loop has no schedulable statements after "
           "removing replayable scalar Bind statements";

    std::vector<PipelineStageInfo> pipeline_stage_infos;
    for (size_t i = 0; i < analysis.scheduled_stmts.size(); i++) {
      auto pinfo = MakePipelineStageInfo(analysis.scheduled_stmts[i], i);
      pipeline_stage_infos.push_back(std::move(pinfo));
    }

    struct CPAsyncGroupInfo {
      int group_id = -1;
      int anchor_cp_async_stmt = -1;
      std::vector<int> cp_async_stmt_indices;
      std::vector<int> commit_stmt_indices;
      BufferSet written_buffers;
      int last_use_stmt_index = -1;
    };
    struct WaitDependencyInfo {
      int wait_stmt_index = -1;
      std::vector<int> required_group_ids;
    };

    std::vector<CPAsyncGroupInfo> cp_async_groups;
    std::vector<WaitDependencyInfo> wait_dependencies;
    std::vector<int> committed_groups_in_order;

    auto create_new_cp_async_group = [&]() -> int {
      int group_id = static_cast<int>(cp_async_groups.size());
      CPAsyncGroupInfo group;
      group.group_id = group_id;
      cp_async_groups.push_back(std::move(group));
      return group_id;
    };

    int open_cp_async_group = -1;
    for (size_t i = 0; i < pipeline_stage_infos.size(); ++i) {
      auto &pinfo = pipeline_stage_infos[i];
      if (pinfo.HasCpAsyncCall()) {
        if (open_cp_async_group == -1) {
          open_cp_async_group = create_new_cp_async_group();
        }
        pinfo.cp_async_group = open_cp_async_group;
        auto &group = cp_async_groups[open_cp_async_group];
        group.cp_async_stmt_indices.push_back(static_cast<int>(i));
        if (group.anchor_cp_async_stmt == -1) {
          group.anchor_cp_async_stmt = static_cast<int>(i);
        }
        for (const auto &write : pinfo.writes) {
          group.written_buffers.insert(write->buffer);
        }
      }
      if (pinfo.HasCpAsyncCommit()) {
        if (open_cp_async_group == -1) {
          open_cp_async_group = create_new_cp_async_group();
        }
        pinfo.cp_async_group = open_cp_async_group;
        cp_async_groups[open_cp_async_group].commit_stmt_indices.push_back(
            static_cast<int>(i));
        committed_groups_in_order.push_back(open_cp_async_group);
        open_cp_async_group = -1;
      }
      if (pinfo.HasCpAsyncWait()) {
        int committed_count =
            static_cast<int>(committed_groups_in_order.size());
        int retain_inflight = pinfo.cp_async_wait_has_dynamic
                                  ? 0
                                  : pinfo.cp_async_wait_min_inflight;
        int required_count =
            pinfo.cp_async_wait_has_dynamic
                ? committed_count
                : std::max(0, committed_count - retain_inflight);

        WaitDependencyInfo wait_dep;
        wait_dep.wait_stmt_index = static_cast<int>(i);
        wait_dep.required_group_ids.assign(committed_groups_in_order.begin(),
                                           committed_groups_in_order.begin() +
                                               required_count);
        wait_dependencies.push_back(std::move(wait_dep));
      }
    }

    const int pipeline_stmt_count =
        static_cast<int>(pipeline_stage_infos.size());
    auto stmt_reads_buffer_set = [&](int stmt_idx,
                                     const BufferSet &buffers) -> bool {
      if (buffers.empty() || stmt_idx < 0 || stmt_idx >= pipeline_stmt_count) {
        return false;
      }
      for (const BufferRegion &read : pipeline_stage_infos[stmt_idx].reads) {
        if (buffers.count(read->buffer)) {
          return true;
        }
      }
      return false;
    };

    BufferSet async_written_buffers;
    std::vector<int> cp_async_group_first_consumer(
        cp_async_groups.size(), std::numeric_limits<int>::max());
    for (size_t group_id = 0; group_id < cp_async_groups.size(); ++group_id) {
      const auto &group = cp_async_groups[group_id];
      async_written_buffers.insert(group.written_buffers.begin(),
                                   group.written_buffers.end());
      for (int stmt_idx = 0; stmt_idx < pipeline_stmt_count; ++stmt_idx) {
        if (pipeline_stage_infos[stmt_idx].IsFirstStage()) {
          continue;
        }
        if (stmt_reads_buffer_set(stmt_idx, group.written_buffers)) {
          cp_async_group_first_consumer[group_id] = stmt_idx;
          break;
        }
      }
    }

    int last_bound_consumer_stmt = -1;
    for (auto &wait_dep : wait_dependencies) {
      int wait_stmt_idx = wait_dep.wait_stmt_index;
      if (wait_stmt_idx < 0 || wait_stmt_idx >= pipeline_stmt_count) {
        continue;
      }
      const auto &wait_stmt_info = pipeline_stage_infos[wait_stmt_idx];
      if (!wait_stmt_info.HasCpAsyncWait() ||
          wait_stmt_info.cp_async_wait_has_dynamic ||
          wait_stmt_info.cp_async_wait_min_inflight != 0) {
        continue;
      }
      if (wait_stmt_info.HasCpAsyncCall() ||
          wait_stmt_info.HasCpAsyncCommit()) {
        continue;
      }

      int search_start =
          std::max(wait_stmt_idx + 1, last_bound_consumer_stmt + 1);
      int consumer_stmt_idx = -1;
      for (int stmt_idx = search_start; stmt_idx < pipeline_stmt_count;
           ++stmt_idx) {
        if (pipeline_stage_infos[stmt_idx].IsFirstStage()) {
          continue;
        }
        if (stmt_reads_buffer_set(stmt_idx, async_written_buffers)) {
          consumer_stmt_idx = stmt_idx;
          break;
        }
      }
      if (consumer_stmt_idx < 0) {
        continue;
      }

      std::vector<int> required_groups_for_consumer;
      for (size_t group_id = 0; group_id < cp_async_groups.size(); ++group_id) {
        if (stmt_reads_buffer_set(consumer_stmt_idx,
                                  cp_async_groups[group_id].written_buffers)) {
          required_groups_for_consumer.push_back(static_cast<int>(group_id));
        }
      }
      if (required_groups_for_consumer.empty()) {
        continue;
      }

      wait_dep.required_group_ids = std::move(required_groups_for_consumer);
      last_bound_consumer_stmt = consumer_stmt_idx;
    }

    std::vector<int> cp_async_group_schedule_order;
    cp_async_group_schedule_order.reserve(cp_async_groups.size());
    for (size_t group_id = 0; group_id < cp_async_groups.size(); ++group_id) {
      cp_async_group_schedule_order.push_back(static_cast<int>(group_id));
    }

    // Some statements before a copy are not copy operations themselves, but
    // they prepare buffers that the copy must read.  A common example is
    // producer-side initialization before a conditional or partial copy:
    //
    //   fill(shared, 0)        // writes shared
    //   copy(global, shared)   // may rely on the initialized values
    //
    // If the copy is moved to the producer side, the fill must move with it;
    // otherwise the copy could observe an uninitialized or wrong shared-buffer
    // value.  PropagateBufferProducersForCopy computes a buffer-level backward
    // dependency closure from copy-stage reads to earlier non-copy writes and
    // marks those statements as `producer_for_copy`.  They then participate in
    // the producer-stage scheduling just like the copy stages they prepare.
    PropagateBufferProducersForCopy(&pipeline_stage_infos);

    // Analysis use-def chain to determine last_use_stmt_index for copy
    // operations This step is critical for pipeline optimization as it
    // identifies the index of the last statement that consumes data produced by
    // copy stages, enabling optimal placement of copy operations in the
    // pipeline schedule.
    AnalyzeCopyLastUse(&pipeline_stage_infos);

    for (auto &group : cp_async_groups) {
      if (group.anchor_cp_async_stmt < 0) {
        continue;
      }
      int group_last_use = -1;
      int group_last_cp_async_stmt = group.anchor_cp_async_stmt;
      for (int cp_async_stmt_idx : group.cp_async_stmt_indices) {
        group_last_cp_async_stmt =
            std::max(group_last_cp_async_stmt, cp_async_stmt_idx);
        group_last_use = std::max(
            group_last_use,
            pipeline_stage_infos[cp_async_stmt_idx].last_use_stmt_index);
      }
      if (group_last_use < 0) {
        group_last_use = group_last_cp_async_stmt;
      }
      group.last_use_stmt_index = group_last_use;
      for (int cp_async_stmt_idx : group.cp_async_stmt_indices) {
        pipeline_stage_infos[cp_async_stmt_idx].last_use_stmt_index =
            group_last_use;
      }
      for (int commit_stmt_idx : group.commit_stmt_indices) {
        auto &commit_info = pipeline_stage_infos[commit_stmt_idx];
        commit_info.last_use_stmt_index = group_last_use;
        if (commit_info.HasCpAsyncCommit() &&
            !commit_info.HasCpAsyncCall()) {
          commit_info.cp_async_commit_stage = true;
        }
      }
    }

    PropagateScalarProducersForCopy(&pipeline_stage_infos);

    std::stable_sort(
        cp_async_group_schedule_order.begin(),
        cp_async_group_schedule_order.end(), [&](int lhs_group, int rhs_group) {
          int lhs_last_use = cp_async_groups[lhs_group].last_use_stmt_index;
          int rhs_last_use = cp_async_groups[rhs_group].last_use_stmt_index;
          if (lhs_last_use != rhs_last_use) {
            return lhs_last_use < rhs_last_use;
          }
          int lhs_first_consumer = cp_async_group_first_consumer[lhs_group];
          int rhs_first_consumer = cp_async_group_first_consumer[rhs_group];
          if (lhs_first_consumer != rhs_first_consumer) {
            return lhs_first_consumer < rhs_first_consumer;
          }
          return cp_async_groups[lhs_group].anchor_cp_async_stmt <
                 cp_async_groups[rhs_group].anchor_cp_async_stmt;
        });

    // Making stages and orders
    int order_idx = 0;
    // Stage 1. Create pipeline stages and assign order
    for (auto &pinfo : pipeline_stage_infos) {
      // Skip elements that must be in first stage:
      // 1. Copy stages (with active last_use_stmt_index) - these need special
      // handling
      //    because they have consumers that depend on their data
      // 2. All Producer stages for copy stages.
      if (pinfo.IsFirstStage() && pinfo.IsLastUseStmtIndexValid()) {
        continue;
      }

      // Main logic stage assignment:
      // - Increment order index
      // - Assign to new stage (current num_stages)
      pinfo.order = order_idx++;
      pinfo.stage = num_stages;

      // Schedule copy stages that have this stage as their last consumer
      // This ensures copy operations are placed right before their final
      // consumer for optimal pipeline efficiency
      for (auto &pinfo_1 : pipeline_stage_infos) {
        if ((pinfo_1.IsFirstStage() &&
             pinfo_1.last_use_stmt_index == pinfo.original_stmt_index)) {
          pinfo_1.order = order_idx++;
          pinfo_1.stage = 0; // Copy stages are typically assigned to stage 0
        }
      }
    }

    ICHECK(size_t(order_idx) == pipeline_stage_infos.size())
        << "The number of stages should be equal to the number of pipeline "
           "stages. "
        << "Got " << order_idx << " stages and " << pipeline_stage_infos.size()
        << " pipeline stages.";

    // Step 2. if all the copy is at the end of the order, we can move these
    // copy to the beginning of the order and shrink the stage offset by 1.
    int copy_stage_at_end = [&]() {
      int copy_stage_cnt = 0;
      int copy_order_min = pipeline_stage_infos.size();
      int non_copy_order_max = 0;
      for (auto &pinfo : pipeline_stage_infos) {
        if (pinfo.IsFirstStage()) {
          copy_stage_cnt++;
          copy_order_min = std::min(copy_order_min, pinfo.order);
        } else {
          non_copy_order_max = std::max(non_copy_order_max, pinfo.order);
        }
      }
      if (copy_order_min > non_copy_order_max)
        return copy_stage_cnt;
      return -1;
    }();
    if (copy_stage_at_end > 0 && num_stages >= 2) {
      for (auto &pinfo : pipeline_stage_infos) { // move copy to the beginning
        pinfo.order =
            (pinfo.order + copy_stage_at_end) % pipeline_stage_infos.size();
        if (!pinfo.IsCopyStage() && !pinfo.IsProducerForCopy() &&
            !pinfo.IsCpAsyncCommitStage())
          pinfo.stage--;
      }
    }

    for (const auto &group : cp_async_groups) {
      if (group.anchor_cp_async_stmt < 0) {
        continue;
      }
      int anchor_stage = pipeline_stage_infos[group.anchor_cp_async_stmt].stage;
      for (int commit_stmt_idx : group.commit_stmt_indices) {
        pipeline_stage_infos[commit_stmt_idx].stage = anchor_stage;
      }
    }

    for (const auto &group : cp_async_groups) {
      if (group.anchor_cp_async_stmt < 0) {
        continue;
      }
      int max_cp_async_order = -1;
      int anchor_stage = pipeline_stage_infos[group.anchor_cp_async_stmt].stage;
      for (int cp_async_stmt_idx : group.cp_async_stmt_indices) {
        if (pipeline_stage_infos[cp_async_stmt_idx].stage == anchor_stage) {
          max_cp_async_order =
              std::max(max_cp_async_order,
                       pipeline_stage_infos[cp_async_stmt_idx].order);
        }
      }
      for (int commit_stmt_idx : group.commit_stmt_indices) {
        if (pipeline_stage_infos[commit_stmt_idx].stage == anchor_stage) {
          if (pipeline_stage_infos[commit_stmt_idx].HasCpAsyncCall()) {
            continue;
          }
          ICHECK_GT(pipeline_stage_infos[commit_stmt_idx].order,
                    max_cp_async_order)
              << "Pipeline planning error: cp.async commit is scheduled before "
                 "its cp.async calls. commit_stmt="
              << commit_stmt_idx << ", commit_order="
              << pipeline_stage_infos[commit_stmt_idx].order
              << ", max_cp_async_order=" << max_cp_async_order
              << ", stage=" << anchor_stage;
        }
      }
    }

    auto get_cp_async_group_stage = [&](int group_id) -> int {
      if (group_id < 0 ||
          group_id >= static_cast<int>(cp_async_groups.size())) {
        return 0;
      }
      const auto &group = cp_async_groups[group_id];
      if (!group.commit_stmt_indices.empty()) {
        return pipeline_stage_infos[group.commit_stmt_indices.back()].stage;
      }
      if (group.anchor_cp_async_stmt >= 0) {
        return pipeline_stage_infos[group.anchor_cp_async_stmt].stage;
      }
      return 0;
    };

    for (const auto &wait_dep : wait_dependencies) {
      if (wait_dep.wait_stmt_index < 0 ||
          wait_dep.wait_stmt_index >=
              static_cast<int>(pipeline_stage_infos.size())) {
        continue;
      }
      const auto &wait_stmt_info =
          pipeline_stage_infos[wait_dep.wait_stmt_index];
      if (wait_stmt_info.HasCpAsyncCall() ||
          wait_stmt_info.HasCpAsyncCommit()) {
        continue;
      }
      if (wait_dep.required_group_ids.empty()) {
        continue;
      }

      int required_stage = pipeline_stage_infos[wait_dep.wait_stmt_index].stage;
      BufferSet waited_buffers;
      for (int group_id : wait_dep.required_group_ids) {
        required_stage =
            std::max(required_stage, get_cp_async_group_stage(group_id));
        if (group_id >= 0 &&
            group_id < static_cast<int>(cp_async_groups.size())) {
          const auto &group = cp_async_groups[group_id];
          waited_buffers.insert(group.written_buffers.begin(),
                                group.written_buffers.end());
        }
      }

      int dependent_consumer_stage = -1;
      if (!waited_buffers.empty()) {
        for (int stmt_idx = wait_dep.wait_stmt_index + 1;
             stmt_idx < static_cast<int>(pipeline_stage_infos.size());
             ++stmt_idx) {
          if (pipeline_stage_infos[stmt_idx].IsFirstStage()) {
            continue;
          }
          bool dependent_read = false;
          for (const BufferRegion &read :
               pipeline_stage_infos[stmt_idx].reads) {
            if (waited_buffers.count(read->buffer)) {
              dependent_read = true;
              break;
            }
          }
          if (dependent_read) {
            dependent_consumer_stage = pipeline_stage_infos[stmt_idx].stage;
            break;
          }
        }
      }

      if (dependent_consumer_stage >= 0) {
        ICHECK_GE(dependent_consumer_stage, required_stage)
            << "Pipeline planning error: wait_group stage cannot be after its "
               "dependent consumer stage. wait_stmt="
            << wait_dep.wait_stmt_index << ", required_stage=" << required_stage
            << ", consumer_stage=" << dependent_consumer_stage;
        pipeline_stage_infos[wait_dep.wait_stmt_index].stage =
            dependent_consumer_stage;
      } else {
        pipeline_stage_infos[wait_dep.wait_stmt_index].stage = required_stage;
      }
    }

    {
      int n = static_cast<int>(pipeline_stage_infos.size());
      std::vector<int> order_rank(n, 0);
      for (int i = 0; i < n; ++i) {
        order_rank[i] = pipeline_stage_infos[i].order;
      }

      std::vector<std::unordered_set<int>> edges(n);
      std::vector<int> indeg(n, 0);

      auto add_edge = [&](int u, int v) {
        if (u < 0 || v < 0 || u >= n || v >= n || u == v) {
          return;
        }
        if (edges[u].insert(v).second) {
          indeg[v] += 1;
        }
      };

      {
        auto scalar_def_to_stmt =
            MakeStageAnalyzer().BuildScalarDefMap(pipeline_stage_infos);
        for (int consumer_idx = 0; consumer_idx < n; ++consumer_idx) {
          const auto &consumer = pipeline_stage_infos[consumer_idx];
          for (const VarNode *var : consumer.scalar_uses) {
            auto it = scalar_def_to_stmt.find(var);
            if (it == scalar_def_to_stmt.end() || it->second == consumer_idx) {
              continue;
            }
            int producer_idx = it->second;
            if (pipeline_stage_infos[producer_idx].stage == consumer.stage) {
              add_edge(producer_idx, consumer_idx);
            }
          }
        }
      }

      auto group_schedule_key = [&](const CPAsyncGroupInfo &group) {
        int key = std::numeric_limits<int>::max();
        for (int cp_stmt_idx : group.cp_async_stmt_indices) {
          key = std::min(key, pipeline_stage_infos[cp_stmt_idx].order);
        }
        for (int commit_stmt_idx : group.commit_stmt_indices) {
          key = std::min(key, pipeline_stage_infos[commit_stmt_idx].order);
        }
        if (key == std::numeric_limits<int>::max()) {
          key = group.anchor_cp_async_stmt;
        }
        return key;
      };

      std::vector<int> ordered_cp_async_groups;
      ordered_cp_async_groups.reserve(cp_async_groups.size());
      for (size_t group_id = 0; group_id < cp_async_groups.size(); ++group_id) {
        ordered_cp_async_groups.push_back(static_cast<int>(group_id));
      }
      std::stable_sort(
          ordered_cp_async_groups.begin(), ordered_cp_async_groups.end(),
          [&](int lhs_group, int rhs_group) {
            int lhs_key = group_schedule_key(cp_async_groups[lhs_group]);
            int rhs_key = group_schedule_key(cp_async_groups[rhs_group]);
            if (lhs_key != rhs_key) {
              return lhs_key < rhs_key;
            }
            return cp_async_groups[lhs_group].anchor_cp_async_stmt <
                   cp_async_groups[rhs_group].anchor_cp_async_stmt;
          });

      for (size_t g = 0; g < cp_async_groups.size(); ++g) {
        const auto &group = cp_async_groups[g];
        for (int cp_stmt_idx : group.cp_async_stmt_indices) {
          for (int commit_stmt_idx : group.commit_stmt_indices) {
            if (pipeline_stage_infos[cp_stmt_idx].stage ==
                pipeline_stage_infos[commit_stmt_idx].stage) {
              add_edge(cp_stmt_idx, commit_stmt_idx);
            }
          }
        }
      }
      for (size_t i = 0; i + 1 < ordered_cp_async_groups.size(); ++i) {
        const auto &group = cp_async_groups[ordered_cp_async_groups[i]];
        if (group.commit_stmt_indices.empty()) {
          continue;
        }
        const auto &next_group =
            cp_async_groups[ordered_cp_async_groups[i + 1]];
        for (int commit_stmt_idx : group.commit_stmt_indices) {
          for (int next_cp_stmt_idx : next_group.cp_async_stmt_indices) {
            if (pipeline_stage_infos[commit_stmt_idx].stage ==
                pipeline_stage_infos[next_cp_stmt_idx].stage) {
              add_edge(commit_stmt_idx, next_cp_stmt_idx);
            }
          }
        }
      }

      for (const auto &wait_dep : wait_dependencies) {
        int wait_stmt_idx = wait_dep.wait_stmt_index;
        if (wait_stmt_idx < 0 || wait_stmt_idx >= n) {
          continue;
        }

        const auto &wait_stmt_info = pipeline_stage_infos[wait_stmt_idx];
        if (wait_stmt_info.HasCpAsyncCall() ||
            wait_stmt_info.HasCpAsyncCommit()) {
          continue;
        }

        BufferSet waited_buffers;
        for (int group_id : wait_dep.required_group_ids) {
          if (group_id < 0 ||
              group_id >= static_cast<int>(cp_async_groups.size())) {
            continue;
          }
          const auto &group = cp_async_groups[group_id];
          waited_buffers.insert(group.written_buffers.begin(),
                                group.written_buffers.end());

          for (int commit_stmt_idx : group.commit_stmt_indices) {
            if (pipeline_stage_infos[commit_stmt_idx].stage ==
                wait_stmt_info.stage) {
              add_edge(commit_stmt_idx, wait_stmt_idx);
            }
          }
        }

        if (waited_buffers.empty()) {
          continue;
        }

        int first_dependent_consumer_idx = -1;
        for (int consumer_stmt_idx = wait_stmt_idx + 1; consumer_stmt_idx < n;
             ++consumer_stmt_idx) {
          if (pipeline_stage_infos[consumer_stmt_idx].stage !=
              wait_stmt_info.stage) {
            continue;
          }
          bool dependent_read = false;
          for (const BufferRegion &read :
               pipeline_stage_infos[consumer_stmt_idx].reads) {
            if (waited_buffers.count(read->buffer)) {
              dependent_read = true;
              break;
            }
          }
          if (dependent_read) {
            if (first_dependent_consumer_idx == -1) {
              first_dependent_consumer_idx = consumer_stmt_idx;
            }
            add_edge(wait_stmt_idx, consumer_stmt_idx);
          }
        }

        if (first_dependent_consumer_idx != -1) {
          for (int stmt_idx = wait_stmt_idx + 1;
               stmt_idx < first_dependent_consumer_idx; ++stmt_idx) {
            const auto &mid_stmt_info = pipeline_stage_infos[stmt_idx];
            if (mid_stmt_info.stage != wait_stmt_info.stage) {
              continue;
            }
            if (mid_stmt_info.reads.empty() && mid_stmt_info.writes.empty()) {
              break;
            }
            if (mid_stmt_info.HasCpAsyncCall() ||
                mid_stmt_info.HasCpAsyncCommit() ||
                mid_stmt_info.HasCpAsyncWait()) {
              break;
            }
            bool touches_waited_buffers = false;
            for (const BufferRegion &read : mid_stmt_info.reads) {
              if (waited_buffers.count(read->buffer)) {
                touches_waited_buffers = true;
                break;
              }
            }
            if (!touches_waited_buffers) {
              for (const BufferRegion &write : mid_stmt_info.writes) {
                if (waited_buffers.count(write->buffer)) {
                  touches_waited_buffers = true;
                  break;
                }
              }
            }
            if (!touches_waited_buffers) {
              add_edge(stmt_idx, wait_stmt_idx);
            }
          }
        }
      }

      using Item = std::pair<int, int>;
      std::priority_queue<Item, std::vector<Item>, std::greater<Item>> ready;
      for (int i = 0; i < n; ++i) {
        if (indeg[i] == 0) {
          ready.push({order_rank[i], i});
        }
      }

      std::vector<int> topo_order;
      topo_order.reserve(n);
      while (!ready.empty()) {
        auto [rank, u] = ready.top();
        ready.pop();
        topo_order.push_back(u);
        for (int v : edges[u]) {
          indeg[v] -= 1;
          if (indeg[v] == 0) {
            ready.push({order_rank[v], v});
          }
        }
      }

      ICHECK_EQ(static_cast<int>(topo_order.size()), n)
          << "Pipeline planning error: cycle detected while enforcing cp.async "
             "ordering constraints.";

      for (int new_order = 0; new_order < n; ++new_order) {
        pipeline_stage_infos[topo_order[new_order]].order = new_order;
      }
    }

    ValidateScalarDependencies(pipeline_stage_infos);

    // Finally, make the pipeline annotation
    Map<String, Any> annotations;
    for (const auto &[key, value] : loop->annotations) {
      if (key != "num_stages") {
        annotations.Set(key, value);
      }
    }
    // Preserve the original TileLang pipelining depth for downstream scheduling
    // (e.g. generated async-copy wait placement). We intentionally do NOT
    // keep the legacy key "num_stages" here because multiple downstream passes
    // (e.g. internal buffer versioning / warp specialization) treat it as an
    // active pipeline marker and do not support nested pipelines.
    annotations.Set("tl_pipelined_num_stages", Integer(num_stages));

    std::vector<Integer> orders, stages;
    orders.reserve(pipeline_stage_infos.size());
    stages.reserve(pipeline_stage_infos.size());
    for (auto &pinfo : pipeline_stage_infos) {
      orders.push_back(pinfo.order);
      stages.push_back(pinfo.stage);
    }

    annotations.Set(s_tir::attr::software_pipeline_stage,
                    Array<Integer>(stages));
    annotations.Set(s_tir::attr::software_pipeline_order,
                    Array<Integer>(orders));
    if (analysis.has_bind_stmt) {
      annotations.Set(kPipelineReplayableScalarBinds,
                      analysis.replayable_bind_mask);
    }

    // Propagate per-statement TMA eligibility so InjectSoftwarePipeline can
    // rewrite TMA copies to use pipeline-level barrier management.
    {
      std::vector<Integer> tma_copies;
      tma_copies.reserve(pipeline_stage_infos.size());
      bool has_tma_copy = false;
      for (auto &pinfo : pipeline_stage_infos) {
        bool IsTmaCopy = pinfo.IsTmaCopy();
        has_tma_copy = has_tma_copy || IsTmaCopy;
        tma_copies.push_back(Integer(IsTmaCopy ? 1 : 0));
      }
      if (has_tma_copy) {
        annotations.Set(kPipelineTmaCopies, Array<Integer>(tma_copies));
      }
    }

    std::vector<std::vector<int>> explicit_async_groups;
    explicit_async_groups.reserve(cp_async_group_schedule_order.size());
    for (int group_id : cp_async_group_schedule_order) {
      explicit_async_groups.push_back(
          cp_async_groups[group_id].cp_async_stmt_indices);
    }
    EmitAsyncAnnotations(pipeline_stage_infos, explicit_async_groups,
                         &annotations);

    // Reconstruct the loop body with the flattened SeqStmt so that
    // InjectSoftwarePipeline sees the correct number of pipeline stages.
    Stmt new_body = MakePipelineBody(flat_stmts);

    return For(loop->loop_var, loop->min, loop->extent, loop->kind, new_body,
               loop->thread_binding, annotations);
  }

  Stmt VisitStmt_(const SBlockNode *op) final {
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.Set(buffer->data, buffer);
    }
    SBlock block = Downcast<SBlock>(StmtExprMutator::VisitStmt_(op));
    for (const auto &buffer : op->alloc_buffers) {
      buffer_data_to_buffer_.erase(buffer->data);
    }
    return block;
  }

  Map<Var, Buffer> buffer_data_to_buffer_;
  Target target_;
  bool use_async_copy_{};
};

tvm::transform::Pass PipelinePlanning() {
  using namespace tirx::transform;
  auto pass_func = [=](PrimFunc f, const IRModule &m, PassContext ctx) {
    bool use_async_copy =
        ctx->GetConfig<Bool>("tirx.use_async_copy", Bool(true)).value();
    PrimFuncNode *fptr = f.CopyOnWrite();
    fptr->body = PipelinePlanner::Substitute(f, use_async_copy);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.PipelinePlanning", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = reflection;
  refl::GlobalDef().def("tl.transform.PipelinePlanning", PipelinePlanning);
}

} // namespace tl
} // namespace tvm
