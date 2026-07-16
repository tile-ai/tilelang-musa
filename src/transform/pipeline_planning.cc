#include "support/check.h"
#include <tvm/arith/analyzer.h>
#include <tvm/ir/cast.h>
#include <tvm/s_tir/stmt.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/op.h>
#include <tvm/tirx/stmt.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include "../op/builtin.h"
#include "../op/copy.h"
#include "../op/parallel.h"
#include "../op/region.h"
#include "../op/utils.h"
#include "common/pipeline_utils.h"
#include "pipeline/access_analysis.h"
#include "pipeline/body_analysis.h"
#include "pipeline/stage_analysis.h"
#include <algorithm>
#include <functional>
#include <limits>
#include <queue>
#include <utility>
#include <vector>

#include "backend/common/target_utils.h"
#include "tvm/ir/expr.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

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
      if (pinfo.has_cp_async_call()) {
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
      if (pinfo.has_cp_async_commit()) {
        if (open_cp_async_group == -1) {
          open_cp_async_group = create_new_cp_async_group();
        }
        pinfo.cp_async_group = open_cp_async_group;
        cp_async_groups[open_cp_async_group].commit_stmt_indices.push_back(
            static_cast<int>(i));
        committed_groups_in_order.push_back(open_cp_async_group);
        open_cp_async_group = -1;
      }
      if (pinfo.has_cp_async_wait()) {
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
        if (pipeline_stage_infos[stmt_idx].is_first_stage()) {
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
      if (!wait_stmt_info.has_cp_async_wait() ||
          wait_stmt_info.cp_async_wait_has_dynamic ||
          wait_stmt_info.cp_async_wait_min_inflight != 0) {
        continue;
      }
      if (wait_stmt_info.has_cp_async_call() ||
          wait_stmt_info.has_cp_async_commit()) {
        continue;
      }

      int search_start =
          std::max(wait_stmt_idx + 1, last_bound_consumer_stmt + 1);
      int consumer_stmt_idx = -1;
      for (int stmt_idx = search_start; stmt_idx < pipeline_stmt_count;
           ++stmt_idx) {
        if (pipeline_stage_infos[stmt_idx].is_first_stage()) {
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
        if (commit_info.has_cp_async_commit() &&
            !commit_info.has_cp_async_call()) {
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
      if (pinfo.is_first_stage() && pinfo.is_last_use_stmt_index_valid()) {
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
        if ((pinfo_1.is_first_stage() &&
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
        if (pinfo.is_first_stage()) {
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
        if (!pinfo.is_copy_stage() && !pinfo.is_producer_for_copy() &&
            !pinfo.is_cp_async_commit_stage())
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
          if (pipeline_stage_infos[commit_stmt_idx].has_cp_async_call()) {
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
      if (wait_stmt_info.has_cp_async_call() ||
          wait_stmt_info.has_cp_async_commit()) {
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
          if (pipeline_stage_infos[stmt_idx].is_first_stage()) {
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
        if (wait_stmt_info.has_cp_async_call() ||
            wait_stmt_info.has_cp_async_commit()) {
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
            if (mid_stmt_info.has_cp_async_call() ||
                mid_stmt_info.has_cp_async_commit() ||
                mid_stmt_info.has_cp_async_wait()) {
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
        bool is_tma_copy = pinfo.is_tma_copy();
        has_tma_copy = has_tma_copy || is_tma_copy;
        tma_copies.push_back(Integer(is_tma_copy ? 1 : 0));
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
