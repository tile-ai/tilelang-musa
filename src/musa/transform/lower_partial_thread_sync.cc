/*!
 * \file tl/musa/transform/lower_partial_thread_sync.cc
 * \brief Lower partial shared-memory thread synchronization to MP31 barriers.
 */

#include "musa/op/builtin.h"

#include "op/builtin.h"
#include "op/utils.h"

#include <tvm/arith/analyzer.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/tirx/analysis.h>
#include <tvm/tirx/builtin.h>
#include <tvm/tirx/stmt_functor.h>
#include <tvm/tirx/transform.h>

#include <algorithm>
#include <cstdint>
#include <cstdlib>
#include <map>
#include <optional>
#include <sstream>
#include <string>
#include <unordered_map>
#include <unordered_set>
#include <utility>
#include <vector>

namespace tvm {
namespace tl {

using namespace tirx;

namespace {

constexpr int kMP31SquadThreads = 128;
constexpr int kMP31MaxHardwareBarriers = 16;

struct IntInterval {
  int64_t min_val{0};
  int64_t max_val{0};
  bool valid{false};
};

uint64_t NextBitMask(uint64_t v) {
  if (v == 0) {
    return 0;
  }
  uint64_t mask = 1;
  while (mask <= v) {
    mask <<= 1;
  }
  return mask - 1;
}

// Compute a sound conservative integer interval [min_val, max_val] for an index
// expression, handling bitwise_xor over non-negative bounded operands (used by
// MP31 SQMMA 256B swizzle formulas) where TVM's default ConstIntBound widens
// to [-inf, +inf].
IntInterval ComputeSoundInterval(const PrimExpr &expr,
                                 arith::Analyzer *analyzer) {
  if (const auto *imm = expr.as<IntImmNode>()) {
    return {imm->value, imm->value, true};
  }
  if (const auto *add = expr.as<AddNode>()) {
    IntInterval a = ComputeSoundInterval(add->a, analyzer);
    IntInterval b = ComputeSoundInterval(add->b, analyzer);
    if (a.valid && b.valid) {
      return {a.min_val + b.min_val, a.max_val + b.max_val, true};
    }
  } else if (const auto *sub = expr.as<SubNode>()) {
    IntInterval a = ComputeSoundInterval(sub->a, analyzer);
    IntInterval b = ComputeSoundInterval(sub->b, analyzer);
    if (a.valid && b.valid) {
      return {a.min_val - b.max_val, a.max_val - b.min_val, true};
    }
  } else if (const auto *mul = expr.as<MulNode>()) {
    IntInterval a = ComputeSoundInterval(mul->a, analyzer);
    IntInterval b = ComputeSoundInterval(mul->b, analyzer);
    if (a.valid && b.valid) {
      int64_t p1 = a.min_val * b.min_val;
      int64_t p2 = a.min_val * b.max_val;
      int64_t p3 = a.max_val * b.min_val;
      int64_t p4 = a.max_val * b.max_val;
      return {std::min({p1, p2, p3, p4}), std::max({p1, p2, p3, p4}), true};
    }
  } else if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::bitwise_xor()) && call->args.size() == 2U) {
      IntInterval a = ComputeSoundInterval(call->args[0], analyzer);
      IntInterval b = ComputeSoundInterval(call->args[1], analyzer);
      if (a.valid && b.valid && a.min_val >= 0 && b.min_val >= 0) {
        uint64_t hi =
            NextBitMask(static_cast<uint64_t>(std::max(a.max_val, b.max_val)));
        return {0, static_cast<int64_t>(hi), true};
      }
    }
  }
  arith::ConstIntBound bound = analyzer->const_int_bound(expr);
  if (bound->min_value > arith::ConstIntBound::kNegInf &&
      bound->max_value < arith::ConstIntBound::kPosInf) {
    return {bound->min_value, bound->max_value, true};
  }
  return {0, 0, false};
}

struct ParsedSQMMACall {
  int m{0};
  int n{0};
  int k{0};
  int num_warp_m{0};
  int num_warp_n{0};
  int a_elem_bits{0};
  int b_elem_bits{0};
  const VarNode *a_var{nullptr};
  const VarNode *b_var{nullptr};
  const VarNode *c_var{nullptr};
  int64_t a_offset{0};
  int64_t b_offset{0};
  int64_t stride_a{0};
  int64_t stride_b{0};
  int64_t a_col_major{0};
  int64_t b_col_major{0};
  bool valid{false};
};

int ParseSQMMATagBits(const std::string &tag) {
  if (tag.find("s4") != std::string::npos)
    return 4;
  if (tag.find("e4m3") != std::string::npos ||
      tag.find("e5m2") != std::string::npos ||
      tag.find("s8") != std::string::npos ||
      tag.find("u8") != std::string::npos)
    return 8;
  if (tag.find("f16") != std::string::npos ||
      tag.find("bf16") != std::string::npos)
    return 16;
  if (tag.find("tf32") != std::string::npos ||
      tag.find("f32") != std::string::npos)
    return 32;
  return 0;
}

std::pair<const VarNode *, int64_t> ExtractAccessPtrVarAndOffset(
    const PrimExpr &expr) {
  if (const auto *call = expr.as<CallNode>()) {
    if (call->op.same_as(builtin::tvm_access_ptr()) && call->args.size() >= 3) {
      const auto *v = call->args[1].as<VarNode>();
      const auto *off = call->args[2].as<IntImmNode>();
      if (v != nullptr && off != nullptr) {
        return {v, off->value};
      }
      return {v, -1};
    }
  }
  if (const auto *v = expr.as<VarNode>()) {
    return {v, 0};
  }
  return {nullptr, -1};
}

ParsedSQMMACall ParseSQMMASSCall(const CallNode *call) {
  ParsedSQMMACall info;
  if (!call->op.same_as(builtin::call_extern()) || call->args.size() < 9U) {
    return info;
  }
  const auto *name = call->args[0].as<StringImmNode>();
  if (name == nullptr) {
    return info;
  }
  const std::string &fn = name->value;
  constexpr const char *kPrefix = "tl::sqmma_ss<";
  if (fn.rfind(kPrefix, 0) != 0 || fn.back() != '>') {
    return info;
  }
  std::string body =
      fn.substr(std::char_traits<char>::length(kPrefix),
                fn.size() - std::char_traits<char>::length(kPrefix) - 1);
  std::vector<std::string> tokens;
  std::stringstream ss(body);
  std::string tok;
  while (std::getline(ss, tok, ',')) {
    size_t start = tok.find_first_not_of(" \t");
    size_t end = tok.find_last_not_of(" \t");
    if (start == std::string::npos) {
      return info;
    }
    tokens.push_back(tok.substr(start, end - start + 1));
  }
  if (tokens.size() < 11U) {
    return info;
  }
  info.m = std::atoi(tokens[0].c_str());
  info.n = std::atoi(tokens[1].c_str());
  info.k = std::atoi(tokens[2].c_str());
  info.num_warp_m = std::atoi(tokens[3].c_str());
  info.num_warp_n = std::atoi(tokens[4].c_str());
  info.a_elem_bits = ParseSQMMATagBits(tokens[8]);
  info.b_elem_bits = ParseSQMMATagBits(tokens[9]);
  if (info.m <= 0 || info.n <= 0 || info.k <= 0 || info.num_warp_m <= 0 ||
      info.num_warp_n <= 0 || info.a_elem_bits <= 0 || info.b_elem_bits <= 0) {
    return info;
  }

  auto [a_var, a_off] = ExtractAccessPtrVarAndOffset(call->args[1]);
  auto [b_var, b_off] = ExtractAccessPtrVarAndOffset(call->args[2]);
  auto [c_var, _] = ExtractAccessPtrVarAndOffset(call->args[3]);
  const auto *sa = call->args[4].as<IntImmNode>();
  const auto *sb = call->args[5].as<IntImmNode>();
  const auto *acm = call->args[6].as<IntImmNode>();
  const auto *bcm = call->args[7].as<IntImmNode>();
  if (!a_var || !b_var || !sa || !sb || !acm || !bcm) {
    return info;
  }
  info.a_var = a_var;
  info.b_var = b_var;
  info.c_var = c_var;
  info.a_offset = a_off;
  info.b_offset = b_off;
  info.stride_a = sa->value;
  info.stride_b = sb->value;
  info.a_col_major = acm->value;
  info.b_col_major = bcm->value;
  info.valid = true;
  return info;
}

bool IsSharedScope(const std::string &scope) {
  return scope == "shared" || scope == "shared.dyn";
}

bool ExprReferencesAnyVar(const PrimExpr &expr,
                          const std::unordered_set<const VarNode *> &vars) {
  bool found = false;
  PostOrderVisit(expr, [&](const ffi::ObjectRef &node) {
    if (found) {
      return;
    }
    if (const auto *v = node.as<VarNode>()) {
      if (vars.count(v)) {
        found = true;
      }
    }
  });
  return found;
}

// Collect all shared-memory handle Vars (alloc_buffer / decl_buffer with
// shared* scope, plus Bind aliases derived via handle_add_byte_offset).
class SharedVarCollector : public StmtExprVisitor {
public:
  void VisitStmt_(const AllocBufferNode *op) final {
    if (IsSharedScope(std::string(op->buffer.scope()))) {
      smem_vars_.insert(op->buffer->data.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const DeclBufferNode *op) final {
    if (IsSharedScope(std::string(op->buffer.scope()))) {
      smem_vars_.insert(op->buffer->data.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitStmt_(const BindNode *op) final {
    if (ExprReferencesAnyVar(op->value, smem_vars_)) {
      smem_vars_.insert(op->var.get());
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  const std::unordered_set<const VarNode *> &smem_vars() const {
    return smem_vars_;
  }

private:
  std::unordered_set<const VarNode *> smem_vars_;
};

struct AccessRecord {
  const VarNode *buf_var;
  PrimExpr index;
  std::vector<std::pair<Var, Range>> loop_bounds;
};

// Analyze a consumer branch `if (tx < thread_count)` to determine whether a
// partial barrier `partial_id` synchronizes ONLY squad-disjoint (128-thread
// partitioned) shared-memory buffers.
class SquadDisjointAnalyzer : public StmtExprVisitor {
public:
  SquadDisjointAnalyzer(int partial_id, Var tx_var, int num_squads,
                        const std::unordered_set<const VarNode *> &smem_vars)
      : partial_id_(partial_id),
        tx_var_(std::move(tx_var)),
        num_squads_(num_squads),
        smem_vars_(smem_vars) {}

  bool Analyze(const Stmt &consumer_body) {
    VisitStmt(consumer_body);
    if (bail_out_ || written_smem_.empty()) {
      return false;
    }

    Var squad("squad", DataType::Int(32));
    Var lane("lane", DataType::Int(32));
    PrimExpr tx_sub = squad * Integer(kMP31SquadThreads) + lane;
    ffi::Map<Var, PrimExpr> tx_map{{tx_var_, tx_sub}};

    std::unordered_map<const VarNode *, int64_t> stride_by_buf;

    auto verify_access = [&](const AccessRecord &rec) -> bool {
      arith::Analyzer analyzer;
      analyzer.Bind(squad, Range::FromMinExtent(0, num_squads_));
      analyzer.Bind(lane, Range::FromMinExtent(0, kMP31SquadThreads));
      for (const auto &[lv, rng] : rec.loop_bounds) {
        analyzer.Bind(lv, rng);
      }

      PrimExpr idx_sub = analyzer.Simplify(Substitute(rec.index, tx_map));
      PrimExpr rem = analyzer.Simplify(
          Substitute(idx_sub, ffi::Map<Var, PrimExpr>{{squad, Integer(0)}}));
      PrimExpr idx_1 = analyzer.Simplify(
          Substitute(idx_sub, ffi::Map<Var, PrimExpr>{{squad, Integer(1)}}));
      PrimExpr stride_expr = analyzer.Simplify(idx_1 - rem);
      const auto *stride_imm = stride_expr.as<IntImmNode>();
      if (stride_imm == nullptr || stride_imm->value <= 0) {
        return false;
      }
      int64_t stride = stride_imm->value;
      if (UsesVar(rem, [sq = squad.get()](const VarNode *v) { return v == sq; })) {
        return false;
      }
      if (!analyzer.CanProveEqual(idx_sub, squad * Integer(stride) + rem)) {
        return false;
      }
      IntInterval bounds = ComputeSoundInterval(rem, &analyzer);
      if (!bounds.valid || bounds.min_val < 0 || bounds.max_val >= stride) {
        return false;
      }

      auto it = stride_by_buf.find(rec.buf_var);
      if (it == stride_by_buf.end()) {
        stride_by_buf.emplace(rec.buf_var, stride);
      } else if (it->second != stride) {
        return false;
      }
      return true;
    };

    for (const AccessRecord &rec : writes_) {
      if (!verify_access(rec)) {
        return false;
      }
    }
    for (const AccessRecord &rec : reads_) {
      if (written_smem_.count(rec.buf_var) == 0) {
        continue;
      }
      if (!verify_access(rec)) {
        return false;
      }
    }
    for (const ParsedSQMMACall &sq : sqmma_calls_) {
      if (sq.c_var && written_smem_.count(sq.c_var)) {
        return false;
      }
      if (written_smem_.count(sq.a_var)) {
        auto it = stride_by_buf.find(sq.a_var);
        if (it == stride_by_buf.end()) {
          return false;
        }
        int k_squad_m = sq.num_warp_m / 4;
        int k_squad_n = sq.num_warp_n;
        bool a_panelized = (sq.stride_a * sq.a_elem_bits) / 8 > 256;
        if (sq.a_offset != 0 || k_squad_n != 1 || k_squad_m != num_squads_ ||
            sq.a_col_major != 0 || a_panelized || (sq.m % k_squad_m) != 0) {
          return false;
        }
        int64_t expected_stride =
            static_cast<int64_t>(sq.m / k_squad_m) * sq.stride_a;
        if (expected_stride != it->second) {
          return false;
        }
      }
      if (written_smem_.count(sq.b_var)) {
        auto it = stride_by_buf.find(sq.b_var);
        if (it == stride_by_buf.end()) {
          return false;
        }
        int k_squad_m = sq.num_warp_m / 4;
        int k_squad_n = sq.num_warp_n;
        bool b_panelized = (sq.stride_b * sq.b_elem_bits) / 8 > 256;
        if (sq.b_offset != 0 || k_squad_m != 1 || k_squad_n != num_squads_ ||
            sq.b_col_major != 1 || b_panelized || (sq.n % k_squad_n) != 0) {
          return false;
        }
        int64_t expected_stride =
            static_cast<int64_t>(sq.n / k_squad_n) * sq.stride_b;
        if (expected_stride != it->second) {
          return false;
        }
      }
    }

    return true;
  }

private:
  void VisitStmt_(const ForNode *op) final {
    if (bail_out_)
      return;
    loop_bounds_.push_back(
        {op->loop_var, Range::FromMinExtent(op->min, op->extent)});
    StmtExprVisitor::VisitStmt_(op);
    loop_bounds_.pop_back();
  }

  void VisitStmt_(const BufferStoreNode *op) final {
    if (bail_out_)
      return;
    if (IsSharedScope(std::string(op->buffer.scope())) ||
        smem_vars_.count(op->buffer->data.get())) {
      if (op->indices.size() != 1U) {
        bail_out_ = true;
        return;
      }
      written_smem_.insert(op->buffer->data.get());
      writes_.push_back({op->buffer->data.get(), op->indices[0], loop_bounds_});
    }
    StmtExprVisitor::VisitStmt_(op);
  }

  void VisitExpr_(const BufferLoadNode *op) final {
    if (bail_out_)
      return;
    if (IsSharedScope(std::string(op->buffer.scope())) ||
        smem_vars_.count(op->buffer->data.get())) {
      if (op->indices.size() != 1U) {
        bail_out_ = true;
        return;
      }
      reads_.push_back({op->buffer->data.get(), op->indices[0], loop_bounds_});
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  void VisitExpr_(const CallNode *op) final {
    if (bail_out_)
      return;
    if (op->op.same_as(builtin::tvm_storage_sync())) {
      if (op->args.size() != 3U) {
        bail_out_ = true;
        return;
      }
      const auto *bid = op->args[1].as<IntImmNode>();
      if (!bid || static_cast<int>(bid->value) != partial_id_) {
        bail_out_ = true;
        return;
      }
      return;
    }
    if (op->op.same_as(tl::tma_load()) || op->op.same_as(tl::tma_store()) ||
        op->op.same_as(tl::tma_load_im2col())) {
      bail_out_ = true;
      return;
    }
    if (op->op.same_as(builtin::call_extern())) {
      ParsedSQMMACall sq = ParseSQMMASSCall(op);
      if (sq.valid) {
        sqmma_calls_.push_back(sq);
        return;
      }
      // Any other extern referencing a shared-memory pointer (e.g. AllReduce
      // with SMEM workspace, CumSum, BlockBitonicSort) may synchronize across
      // squads, so conservatively keep the full consumer barrier.
      for (size_t i = 1; i < op->args.size(); ++i) {
        if (ExprReferencesAnyVar(op->args[i], smem_vars_)) {
          bail_out_ = true;
          return;
        }
      }
      return;
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  int partial_id_;
  Var tx_var_;
  int num_squads_;
  const std::unordered_set<const VarNode *> &smem_vars_;
  std::vector<std::pair<Var, Range>> loop_bounds_;
  std::unordered_set<const VarNode *> written_smem_;
  std::vector<AccessRecord> writes_;
  std::vector<AccessRecord> reads_;
  std::vector<ParsedSQMMACall> sqmma_calls_;
  bool bail_out_{false};
};

} // namespace

struct PartialBarrierPlan {
  int barrier_start{0};
  int num_barriers{1};
  PrimExpr thread_count;
  bool per_squad{false};
};

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
        consumer_stack_.push_back(
            {static_cast<int>(bound->value), op->then_case});
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
      int pid = static_cast<int>(barrier_id->value);
      partial_counts_.emplace(pid, op->args[2]);
      const auto *tc = op->args[2].as<IntImmNode>();
      if (tc != nullptr && !consumer_stack_.empty() &&
          consumer_stack_.back().first == static_cast<int>(tc->value)) {
        auto it = consumer_body_by_id_.find(pid);
        if (it == consumer_body_by_id_.end()) {
          consumer_body_by_id_.emplace(pid, consumer_stack_.back().second);
        } else if (!it->second.same_as(consumer_stack_.back().second)) {
          // Used in multiple distinct consumer regions: disable per-squad.
          ambiguous_ids_.insert(pid);
        }
      } else {
        ambiguous_ids_.insert(pid);
      }
    }
    StmtExprVisitor::VisitExpr_(op);
  }

  int base_count() const { return base_count_; }
  const Var &tx_var() const { return tx_var_; }
  const std::map<int, PrimExpr> &partial_counts() const {
    return partial_counts_;
  }
  const std::unordered_map<int, Stmt> &consumer_body_by_id() const {
    return consumer_body_by_id_;
  }
  const std::unordered_set<int> &ambiguous_ids() const {
    return ambiguous_ids_;
  }

private:
  int base_count_{-1};
  Var tx_var_;
  std::vector<std::pair<int, Stmt>> consumer_stack_;
  std::map<int, PrimExpr> partial_counts_;
  std::unordered_map<int, Stmt> consumer_body_by_id_;
  std::unordered_set<int> ambiguous_ids_;
};

class PartialSyncLowerer : public StmtExprMutator {
public:
  PartialSyncLowerer(int total_barriers, Var tx_var,
                     std::map<int, PartialBarrierPlan> plans)
      : total_barriers_(total_barriers),
        tx_var_(std::move(tx_var)),
        plans_(std::move(plans)) {}

  Stmt Rewrite(Stmt body) {
    body = VisitStmt(std::move(body));
    for (auto it = phase_vars_.rbegin(); it != phase_vars_.rend(); ++it) {
      Stmt init = BufferStore(*it, Integer(0), {Integer(0)});
      body = SeqStmt(ffi::Array<Stmt>{AllocBuffer(*it), init, body});
    }
    return body;
  }

private:
  PrimExpr VisitExpr_(const CallNode *op) final {
    if (op->op.same_as(builtin::create_barriers())) {
      return Call(op->dtype, op->op, {Integer(total_barriers_)});
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
      for (const auto &[partial_id, plan] : plans_) {
        for (int s = 0; s < plan.num_barriers; ++s) {
          PrimExpr barrier = Integer(plan.barrier_start + s);
          seq.push_back(Evaluate(Call(DataType::Handle(),
                                      builtin::ptx_init_barrier_thread_count(),
                                      {barrier, plan.thread_count})));
        }
      }
      return SeqStmt(seq);
    }

    if (!call->op.same_as(builtin::tvm_storage_sync()) ||
        call->args.size() != 3U) {
      return StmtExprMutator::VisitStmt_(op);
    }

    int partial_id = Downcast<IntImm>(call->args[1])->value;
    const PartialBarrierPlan &plan = plans_.at(partial_id);
    PrimExpr barrier =
        plan.per_squad
            ? (Integer(plan.barrier_start) +
               FloorDiv(tx_var_, Integer(kMP31SquadThreads)))
            : PrimExpr(Integer(plan.barrier_start));
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

  int total_barriers_;
  Var tx_var_;
  std::map<int, PartialBarrierPlan> plans_;
  std::map<int, Buffer> phase_vars_by_id_;
  std::vector<Buffer> phase_vars_;
};

PrimFunc LowerPartialThreadSync(PrimFunc func) {
  PartialSyncCollector collector;
  collector(func->body);
  if (collector.partial_counts().empty()) {
    return func;
  }
  ICHECK_GE(collector.base_count(), 0)
      << "MUSA partial thread sync requires an existing TME barrier pool";

  SharedVarCollector smem_collector;
  smem_collector(func->body);

  // Densely allocate hardware barrier IDs starting right after base_count_ so
  // sparse partial_id numbers never waste MP31 hardware barrier slots, and
  // split into per-squad (128T) barriers whenever the consumer region only
  // synchronizes squad-disjoint shared-memory buffers and fits within the
  // MP31 16-barrier pool.
  int next_barrier = collector.base_count() + 1;
  int remaining_partials = static_cast<int>(collector.partial_counts().size());
  std::map<int, PartialBarrierPlan> plans;

  for (const auto &[pid, tc_expr] : collector.partial_counts()) {
    --remaining_partials;
    bool use_per_squad = false;
    int num_squads = 1;
    if (collector.tx_var().defined() &&
        collector.ambiguous_ids().count(pid) == 0) {
      if (const auto *tc = tc_expr.as<IntImmNode>()) {
        if (tc->value > kMP31SquadThreads &&
            tc->value % kMP31SquadThreads == 0) {
          int cand_squads = static_cast<int>(tc->value / kMP31SquadThreads);
          // Reserve at least 1 slot for each remaining partial barrier.
          if (next_barrier + cand_squads - 1 + remaining_partials <=
              kMP31MaxHardwareBarriers) {
            auto body_it = collector.consumer_body_by_id().find(pid);
            if (body_it != collector.consumer_body_by_id().end()) {
              SquadDisjointAnalyzer analyzer(pid, collector.tx_var(),
                                             cand_squads,
                                             smem_collector.smem_vars());
              if (analyzer.Analyze(body_it->second)) {
                use_per_squad = true;
                num_squads = cand_squads;
              }
            }
          }
        }
      }
    }

    PartialBarrierPlan plan;
    plan.barrier_start = next_barrier;
    if (use_per_squad) {
      plan.num_barriers = num_squads;
      plan.thread_count = Integer(kMP31SquadThreads);
      plan.per_squad = true;
      next_barrier += num_squads;
    } else {
      plan.num_barriers = 1;
      plan.thread_count = tc_expr;
      plan.per_squad = false;
      next_barrier += 1;
    }
    plans.emplace(pid, std::move(plan));
  }

  int total_barriers = next_barrier - 1;
  auto *node = func.CopyOnWrite();
  node->body =
      PartialSyncLowerer(total_barriers, collector.tx_var(), std::move(plans))
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
