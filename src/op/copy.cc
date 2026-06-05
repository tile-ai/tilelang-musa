/*!
 * \file tl/op/copy.cc
 * \brief Define copy operator for various memory transfer strategies (Normal,
 *        Bulk/TMA, LDSM/STSM) and lowering logic for GPU code generation.
 *
 * implementing memory copy operations that can target CPUs or GPUs with
 * optimization for different instructions like bulk copy, matrix load/store,
 * and Hopper's new TMA (Tensor Memory Accelerator).
 */

#include "copy.h"
#include "../layout/tcgen05_layout.h"
#include "../target/musa.h"
#include "../target/utils.h"
#include "../transform/common/loop_fusion_utils.h"
#include "../transform/loop_partition.h"
#include "../transform/loop_vectorize.h"
#include "../transform/ptx_async_copy_injector.h"
#include "utils.h"

#include "builtin.h"
#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

/// Build a TMA leader-thread condition using tl_shuffle_elect.
/// \param thread_extent The number of threads in the current group
///        (e.g., full block extent for non-WS, producer_extent for WS).
///        The elected thread will be the first lane of the first warp in
///        the group.
static PrimExpr MakeTmaLeaderCondition(PrimExpr thread_extent) {
  return Call(DataType::Bool(), tl_shuffle_elect(), {std::move(thread_extent)});
}

PrimExpr GetCopyMbarPhaseExpr(const Map<String, ObjectRef> &annotations,
                              const LowerArgs &T) {
  PrimExpr phase = T.mbar_phase_expr;
  if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations)) {
    phase = explicit_phase.value();
  }
  return phase;
}

// Map TVM dtype to MUSA tensor descriptor dtype for MUSA TMA paths.
static int to_MUtensorDescriptorDataType(DataType dtype) {
  MUtensorDescriptorDataType tp;
  if (dtype.is_float()) {
    switch (dtype.bits()) {
    case 64:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT64;
      break;
    case 32:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32;
      break;
    case 16:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT16;
      break;
    case 8:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_bfloat16()) {
    tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16;
  } else if (dtype.is_float8()) {
    tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
  } else if (dtype.is_int()) {
    switch (dtype.bits()) {
    case 64:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT64;
      break;
    case 32:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32;
      break;
    case 16:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else if (dtype.is_uint()) {
    switch (dtype.bits()) {
    case 64:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT64;
      break;
    case 32:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32;
      break;
    case 16:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16;
      break;
    case 8:
      tp = MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8;
      break;
    default:
      ICHECK(0) << dtype;
    }
  } else {
    ICHECK(0) << dtype;
  }
  return static_cast<int>(tp);
}

} // namespace

static Optional<Layout> GetActiveLayout(const LowerArgs &T, const Buffer &buf) {
  if (T.layout_map.count(buf)) {
    return T.layout_map[buf];
  }
  for (const auto &[key, layout] : T.layout_map) {
    if (key->data.same_as(buf->data) || key->name == buf->name) {
      return layout;
    }
  }
  return Optional<Layout>();
}

static bool IsMatrixLinearLayout(const Layout &layout) {
  if (!layout.defined() || layout->InputDim() < 2) {
    return false;
  }
  size_t ndim = layout->InputDim();
  Array<PrimExpr> matrix_shape{layout->InputShape()[ndim - 2],
                               layout->InputShape()[ndim - 1]};
  Layout linear_layout = makeLinearLayout(matrix_shape);
  if (ndim > 2) {
    Array<PrimExpr> leading_shape;
    for (size_t i = 0; i + 2 < ndim; ++i) {
      leading_shape.push_back(layout->InputShape()[i]);
    }
    linear_layout = linear_layout->Expand(leading_shape);
  }
  return StructuralEqual()(layout, linear_layout);
}

static bool ShapesEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs,
                        arith::Analyzer *analyzer) {
  if (lhs.size() != rhs.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs.size(); ++i) {
    if (!analyzer->CanProveEqual(lhs[i], rhs[i])) {
      return false;
    }
  }
  return true;
}

static bool IsLinearLayout(const Layout &layout) {
  return layout.defined() &&
         StructuralEqual()(layout, makeLinearLayout(layout->InputShape()));
}

static bool IsSharedLayoutContiguousFor1DTMA(const LayoutMap &layout_map,
                                             const Buffer &shared_tensor,
                                             arith::Analyzer *analyzer) {
  auto is_compatible = [&](const Layout &layout) {
    if (ShapesEqual(layout->InputShape(), shared_tensor->shape, analyzer)) {
      return StructuralEqual()(layout, makeLinearLayout(shared_tensor->shape));
    }
    return IsLinearLayout(layout);
  };

  if (layout_map.count(shared_tensor)) {
    auto layout = layout_map.Get(shared_tensor).value().as<Layout>().value();
    if (!is_compatible(layout)) {
      return false;
    }
  }

  for (const auto &[buffer, layout] : layout_map) {
    if (buffer.same_as(shared_tensor)) {
      continue;
    }
    if ((buffer->data.same_as(shared_tensor->data) ||
         buffer->name == shared_tensor->name) &&
        !is_compatible(layout)) {
      return false;
    }
  }
  return true;
}

static bool
IsSharedLayoutRepresentableForTMABulkStore(const LayoutMap &layout_map,
                                           const Buffer &shared_tensor,
                                           arith::Analyzer *analyzer) {
  auto is_representable = [&](const Layout &layout) {
    if (ShapesEqual(layout->InputShape(), shared_tensor->shape, analyzer)) {
      return layout->InputDim() >= 2 || IsLinearLayout(layout);
    }
    return IsLinearLayout(layout);
  };

  if (layout_map.count(shared_tensor)) {
    auto layout = layout_map.Get(shared_tensor).value().as<Layout>().value();
    if (!is_representable(layout)) {
      return false;
    }
  }

  for (const auto &[buffer, layout] : layout_map) {
    if (buffer.same_as(shared_tensor)) {
      continue;
    }
    if ((buffer->data.same_as(shared_tensor->data) ||
         buffer->name == shared_tensor->name) &&
        !is_representable(layout)) {
      return false;
    }
  }
  return true;
}

template <typename TValue>
static Optional<TValue> GetLayoutHintByKey(const Map<Layout, TValue> &hint_map,
                                           const Layout &layout) {
  if (hint_map.count(layout)) {
    return Optional<TValue>(hint_map[layout]);
  }
  for (const auto &[key, value] : hint_map) {
    if (StructuralEqual()(key, layout)) {
      return Optional<TValue>(value);
    }
  }
  return Optional<TValue>();
}

static Optional<bool> GetLayoutKMajor(const LowerArgs &T, const Buffer &buf) {
  auto layout = GetActiveLayout(T, buf);
  if (!layout.defined()) {
    return Optional<bool>();
  }
  auto k_major = GetLayoutHintByKey(T.layout_k_major, layout.value());
  if (k_major.has_value()) {
    return Optional<bool>(k_major.value()->value);
  }
  return Optional<bool>();
}

static Optional<bool> GetLayoutSQMMA(const LowerArgs &T, const Buffer &buf) {
  auto layout = GetActiveLayout(T, buf);
  if (!layout.defined()) {
    return Optional<bool>();
  }
  auto is_sqmma = GetLayoutHintByKey(T.layout_sqmma, layout.value());
  if (is_sqmma.has_value()) {
    return Optional<bool>(is_sqmma.value()->value);
  }
  return Optional<bool>();
}

static Optional<int> GetLayoutSqmmaInstSplit(const LowerArgs &T,
                                             const Buffer &buf) {
  auto layout = GetActiveLayout(T, buf);
  if (!layout.defined()) {
    return Optional<int>();
  }
  auto inst_split_expr =
      GetLayoutHintByKey(T.layout_sqmma_inst_split, layout.value());
  if (!inst_split_expr.has_value()) {
    return Optional<int>();
  }
  if (const auto *imm = inst_split_expr.value().as<IntImmNode>()) {
    return Optional<int>(static_cast<int>(imm->value));
  }
  return Optional<int>();
}

// Constructs a Copy operator node from call arguments and annotations.
// args[0]: source region, args[1]: destination region
// annotations: Map containing coalesced_width, disable_tma, eviction_policy,
// etc.
Copy::Copy(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<CopyNode> node = tvm::ffi::make_object<CopyNode>();
  auto src_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto dst_access = NormalizeToAccessRegion(args[1], kAccessWrite);
  node->src = src_access.region->buffer;
  node->dst = dst_access.region->buffer;
  node->src_range = src_access.region->region;
  node->dst_range = dst_access.region->region;
  node->SetAccessRegions({src_access, dst_access});
  // Copy annotations from the Call node
  node->annotations = annotations;
  data_ = std::move(node);
}

// Creates a shallow clone of this CopyNode.
TileOperator CopyNode::Clone() const {
  auto op = tvm::ffi::make_object<CopyNode>(*this);
  if (par_op_.defined()) {
    op->par_op_ = Downcast<ParallelOp>(par_op_->Clone());
  }
  return Copy(op);
}

// Creates iterator variables for dimensions with extent > 1.
Array<IterVar> CopyNode::MakeIterVars() const {
  // Choose the range set from the lowest-level memory scope between src and
  // dst. Scope levels: global < shared/shared.dyn/shared.tmem < local.fragment
  // (fragment)
  auto scope_level = [](const Buffer &b) -> int {
    String s = b.scope();
    if (s == "local.fragment" || s == "local")
      return 2;
    if (s == "shared" || s == "shared.dyn" || s == "shared.tmem")
      return 1;
    // default to global level for unknown scopes
    return 0;
  };

  int src_level = scope_level(src);
  int dst_level = scope_level(dst);
  bool base_is_src = (src_level >= dst_level);
  const Array<Range> &base_ranges = base_is_src ? src_range : dst_range;

  // Sanity check: when switching away from the original (src_range),
  // ensure the chosen base ranges are not provably smaller than the original
  // per dimension. This guards against generating undersized loop domains.
  // Improved logic: use two pointers to traverse both base_ranges and
  // src_range, skipping dimensions with extent == 1. The number of non-1
  // extents must match.
  arith::Analyzer analyzer;

  size_t base_dim = 0, src_dim = 0;
  while (base_dim < base_ranges.size() && src_dim < src_range.size()) {
    // Skip base extents that are 1
    while (base_dim < base_ranges.size() &&
           is_one(base_ranges[base_dim]->extent)) {
      ++base_dim;
    }
    // Skip src extents that are 1
    while (src_dim < src_range.size() && is_one(src_range[src_dim]->extent)) {
      ++src_dim;
    }
    // Both indices now at non-1, or at end
    if (base_dim < base_ranges.size() && src_dim < src_range.size()) {
      PrimExpr base_ext = base_ranges[base_dim]->extent;
      PrimExpr src_ext = src_range[src_dim]->extent;
      // Only fail if base extent is provably smaller than src extent
      if (analyzer.CanProve(base_ext < src_ext)) {
        std::ostringstream oss;
        oss << "Selected loop range is smaller than original src range at "
               "matched non-1 dimension: "
            << "base(extent=" << base_ext
            << ", scope=" << (base_is_src ? src.scope() : dst.scope())
            << ", min=" << base_ranges[base_dim]->min
            << ", base_dim=" << base_dim << ") < src(extent=" << src_ext
            << ", min=" << src_range[src_dim]->min << ", src_dim=" << src_dim
            << ", scope=" << src.scope() << ") for src=" << src->name
            << ", dst=" << dst->name << "\n";
        oss << "src buffer: " << src->name << ", scope=" << src.scope() << "\n";
        oss << "dst buffer: " << dst->name << ", scope=" << dst.scope() << "\n";
        oss << "base_ranges[" << base_dim
            << "]: min=" << base_ranges[base_dim]->min
            << ", extent=" << base_ext << "\n";
        oss << "src_ranges[" << src_dim << "]: min=" << src_range[src_dim]->min
            << ", extent=" << src_ext << "\n";
        LOG(FATAL) << oss.str();
      }
      ++base_dim;
      ++src_dim;
    }
  }

  // Any remaining unmatched dimensions in either range must all have extent ==
  // 1
  while (base_dim < base_ranges.size()) {
    ICHECK(is_one(base_ranges[base_dim]->extent))
        << "base_ranges has extra non-1 extent at dim " << base_dim;
    ++base_dim;
  }
  while (src_dim < src_range.size()) {
    ICHECK(is_one(src_range[src_dim]->extent))
        << "src_range has extra non-1 extent at dim " << src_dim;
    ++src_dim;
  }

  Array<IterVar> loop_vars;
  size_t idx = 0;
  for (size_t i = 0; i < base_ranges.size(); i++) {
    if (is_one(base_ranges[i]->extent))
      continue;
    Var var = Var(std::string{char('i' + idx)}, base_ranges[i]->extent->dtype);
    idx++;
    loop_vars.push_back(
        {Range(0, base_ranges[i]->extent), var, IterVarType::kDataPar});
  }
  return loop_vars;
}

// Generates index expressions for accessing src (src_dst=0) or dst (src_dst=1)
// buffers.
Array<PrimExpr> CopyNode::MakeIndices(const Array<IterVar> &ivs,
                                      int src_dst) const {
  Array<PrimExpr> indices;
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      indices.push_back(ranges[i]->min);
    else {
      indices.push_back(ranges[i]->min + ivs[idx]->var);
      idx++;
    }
  }
  ICHECK(idx == ivs.size())
      << "idx = " << idx << ", ivs.size() = " << ivs.size()
      << "src name = " << src->name << ", dst name = " << dst->name;
  return indices;
}

// Builds a boundary predicate for memory accesses.
// Returns a conjunction of bounds checks, or empty PrimExpr if all checks pass.
PrimExpr CopyNode::MakePredicate(arith::Analyzer *analyzer,
                                 const Array<IterVar> &ivs,
                                 Array<PrimExpr> extents, int src_dst) const {
  Array<Range> ranges = src_dst == 0 ? src_range : dst_range;

  Array<PrimExpr> cond_list;
  ICHECK(extents.size() == ranges.size()) << extents << " " << ranges;
  size_t idx = 0;
  for (size_t i = 0; i < ranges.size(); i++) {
    if (is_one(ranges[i]->extent))
      continue;
    PrimExpr cond = ranges[i]->min + ivs[idx]->var < extents[i];
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    cond = ranges[i]->min + ivs[idx]->var >= 0;
    if (!analyzer->CanProve(cond, arith::ProofStrength::kSymbolicBound)) {
      cond_list.push_back(cond);
    }
    idx++;
  }
  if (cond_list.empty())
    return {};
  else {
    PrimExpr cond = cond_list[0];
    for (size_t i = 1; i < cond_list.size(); i++)
      cond = And(cond, cond_list[i]);
    return cond;
  }
}

// Constructs a SIMT-style nested loop that implements the copy.
For CopyNode::MakeSIMTLoop(arith::Analyzer *analyzer,
                           bool disable_safe_copy_predication) const {
  Array<IterVar> loop_vars = MakeIterVars();
  bool is_scalar = loop_vars.empty();

  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  ICHECK(loop_vars.size() <= src_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", src_range.size() = " << src_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  ICHECK(loop_vars.size() <= dst_range.size())
      << "loop_vars.size() = " << loop_vars.size()
      << ", dst_range.size() = " << dst_range.size() << ", src = " << src->name
      << ", dst = " << dst->name;

  Array<PrimExpr> src_indices = MakeIndices(loop_vars, 0);
  Array<PrimExpr> dst_indices = MakeIndices(loop_vars, 1);

  PrimExpr src_predicate;
  PrimExpr dst_predicate;
  if (!disable_safe_copy_predication) {
    src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
    dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);
  }

  PrimExpr value = BufferLoad(src, src_indices);
  if (src->dtype != dst->dtype)
    value = Cast(dst->dtype, value);
  if (src_predicate.defined())
    value = if_then_else(src_predicate, value, make_zero(dst->dtype));

  Stmt body = BufferStore(dst, value, dst_indices);
  if (dst_predicate.defined())
    body = IfThenElse(dst_predicate, body);
  if (is_scalar) {
    return For(Var("i"), 0, 1, ForKind::kSerial, body);
  }

  for (int i = loop_vars.size() - 1; i >= 0; i--) {
    Map<String, ObjectRef> loop_annotations;

    // Only attach the parallel related annotations on the outermost loop (i ==
    // 0)
    if (i == 0) {
      if (annotations.count(attr::kCoalescedWidth)) {
        loop_annotations.Set(attr::kCoalescedWidth,
                             annotations.Get(attr::kCoalescedWidth).value());
      }
      if (annotations.count(attr::kParallelLoopLayout)) {
        loop_annotations.Set(
            attr::kParallelLoopLayout,
            annotations.Get(attr::kParallelLoopLayout).value());
      }
    }

    body = For(loop_vars[i]->var, 0, loop_vars[i]->dom->extent,
               ForKind::kParallel, body, std::nullopt, loop_annotations);
  }
  return Downcast<For>(body);
}

// Computes a linearized shared-memory layout for TMA transfers.
// Maps [i, j] -> [i // 256, j // 256, i % 256, j % 256]
Layout CopyNode::ComputeLinearLayout(const Buffer &shared_tensor) const {
  Array<PrimExpr> input_size = shared_tensor->shape;
  Array<PrimExpr> forward_vars;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_vars.push_back(InputPlaceholder(i));
  }
  // [i, j] -> [i // 256, j // 256, i % 256, j % 256]
  Array<PrimExpr> forward_index;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorDiv(forward_vars[i], 256));
  }
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorMod(forward_vars[i], 256));
  }
  return Layout(input_size, forward_index);
}

// Infers memory layouts for this Copy operation based on target and copy
// instruction.
LayoutMap CopyNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  auto target = T.target;
  CopyInst copy_inst;
  if (GetIsAsyncCopy()) {
    // Layout inference does not require a full cp.async legality proof (which
    // depends on final vectorization decisions). Keep the op as CPAsync for
    // inference, and enforce legality during lowering.
    if (!TargetHasAsyncCopy(target)) {
      LOG(FATAL) << "T.async_copy is only supported on targets with cp.async "
                    "support (SM80+). Got target="
                 << target;
    }
    if (!IsGlobalBuffer(src) || !IsSharedBuffer(dst)) {
      LOG(FATAL)
          << "T.async_copy only supports global->shared/shared.dyn copies. "
          << "Got src=" << src->name << " (scope=" << src.scope()
          << "), dst=" << dst->name << " (scope=" << dst.scope() << ").";
    }
    if (src->dtype != dst->dtype) {
      LOG(FATAL) << "T.async_copy requires equal byte-addressable dtypes. "
                 << "Got src dtype=" << src->dtype
                 << ", dst dtype=" << dst->dtype << ".";
    }
    copy_inst = CopyInst::kCPAsync;
  } else {
    copy_inst = GetCopyInst(target, T.layout_map, T.analyzer, T.buffer_oob);
  }

  // If user annotated a loop layout on T.copy, enforce SIMT (normal) copy.
  // Parallel-loop layout only applies to SIMT-style loops we generate here;
  // other copy instructions (TMA/LDSM/STSM/TMem) are incompatible.
  if (annotations.count(attr::kParallelLoopLayout)) {
    if (copy_inst != CopyInst::kNormal && copy_inst != CopyInst::kCPAsync) {
      std::ostringstream oss;
      oss << "T.copy loop layout annotation requires SIMT copy; got "
          << CopyInstToString(copy_inst) << " for src=" << src->name
          << ", dst=" << dst->name
          << ". Remove loop_layout or change copy pattern.";
      LOG(FATAL) << oss.str();
    }
  }

  // Handle tensor memory (tmem) layout inference for both load and store
  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    // TODO (mzw) Add support for tcgen05.cp (in conj. with LowerTmemCopy)
    LayoutMap results;
    bool is_tmem_load = (copy_inst == CopyInst::kTMemLoad);
    Buffer tmem_buf = is_tmem_load ? src : dst;
    Buffer reg_buf = is_tmem_load ? dst : src;

    if (!T.layout_map.count(reg_buf) && T.layout_map.count(tmem_buf)) {
      Layout tmem_layout = T.layout_map[tmem_buf];
      Array<IterVar> logical_coords = MakeIterVars();
      Array<PrimExpr> logical_coords_var = {logical_coords[0]->var,
                                            logical_coords[1]->var};
      Array<PrimExpr> phy_indices = tmem_layout->Forward(logical_coords_var);

      // Tmem physical coord range analysis
      auto analyzer = std::make_shared<arith::Analyzer>();
      for (const auto &iv : logical_coords)
        analyzer->Bind(iv->var, iv->dom);
      arith::ConstIntBound phy_row_bounds =
          analyzer->const_int_bound(phy_indices[0]);
      arith::ConstIntBound phy_col_bounds =
          analyzer->const_int_bound(phy_indices[1]);
      Range row_dom = Range((int)(phy_row_bounds->min_value),
                            (int)(phy_row_bounds->max_value + 1));
      Range col_dom = Range((int)(phy_col_bounds->min_value),
                            (int)(phy_col_bounds->max_value + 1));

      constexpr int WARP_SIZE = 32;
      constexpr int WARPGROUP_SIZE = 4 * WARP_SIZE;
      ICHECK(is_const_int(T.thread_bounds->extent))
          << "Tensor memory copy requires thread_bounds->extent (num_threads) "
             "to be constant integers";
      int num_threads = *as_const_int(T.thread_bounds->extent);
      ICHECK(num_threads % WARPGROUP_SIZE == 0)
          << "Tensor memory copy requires thread bounds to be aligned to "
             "warpgroups, but found "
          << "thread range = " << T.thread_bounds;

      for (int num_useful_wgs = num_threads / WARPGROUP_SIZE;
           num_useful_wgs >= 1; --num_useful_wgs) {
        int num_useful_threads = num_useful_wgs * WARPGROUP_SIZE;
        Tcgen05Meta meta = getTcgen05MetaLd_32dp32b();
        auto [is_success, tmem_coord2frag, num_chunks_each_wg] =
            expandTcgen05Layout(
                meta, phy_col_bounds->max_value - phy_col_bounds->min_value + 1,
                num_useful_threads, row_dom, col_dom);
        if (!is_success) {
          continue;
        }
        Fragment logical_coord2frag =
            Fragment(logical_coords, tmem_coord2frag->Forward(phy_indices),
                     tmem_coord2frag->ForwardThread(phy_indices, std::nullopt),
                     make_itervar("rep", 1));
        results.Set(reg_buf,
                    logical_coord2frag->BindThreadRange(T.thread_bounds));
        break;
      }
    }

    return results;
  }

  if (copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore ||
      copy_inst == CopyInst::kBulkLoad1D ||
      copy_inst == CopyInst::kBulkStore1D) {
    // if can apply swizzling, we skip layout inference
    // for bulk load/store, we can directly apply the layout of normal copy
    // This must be a global/shared layout, so we can skip the parallel op
    // layout inference (parallel layout inference only annotate the loop layout
    // and the register layout).
    Map<Buffer, Layout> result_map;

    bool is_tma_1d = copy_inst == CopyInst::kBulkLoad1D ||
                     copy_inst == CopyInst::kBulkStore1D;
    bool is_load =
        copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkLoad1D;
    bool is_store = copy_inst == CopyInst::kBulkStore ||
                    copy_inst == CopyInst::kBulkStore1D;
    auto global_tensor = is_load ? src : dst;
    auto shared_tensor = is_load ? dst : src;
    auto shared_range = is_load ? dst_range : src_range;

    if (is_tma_1d && shared_range.size() == 1) {
      // 1D TMA Store with single dimension can not be swizzled
      // But 1D TMA can also have multiple dimensions when the last
      // dimension is continuous.
      return result_map;
    }

    // Collect fragment buffers from indices and mark them as fully replicated
    // For Bulk Load/Store, fragment buffers used as indices should be
    // replicated across all threads
    PrimExpr thread_extent = T.thread_bounds->extent;
    for (const auto &range : src_range) {
      CollectFragmentLayouts(range->min, T.let_var_to_expr, T.layout_map,
                             thread_extent, T.thread_bounds, result_map);
      CollectFragmentLayouts(range->extent, T.let_var_to_expr, T.layout_map,
                             thread_extent, T.thread_bounds, result_map);
    }
    for (const auto &range : dst_range) {
      CollectFragmentLayouts(range->min, T.let_var_to_expr, T.layout_map,
                             thread_extent, T.thread_bounds, result_map);
      CollectFragmentLayouts(range->extent, T.let_var_to_expr, T.layout_map,
                             thread_extent, T.thread_bounds, result_map);
    }

    // check shared layout is non-swizzle
    // skip layout inference if shared layout is already annotated
    if (level == InferLevel::kFree && !T.layout_map.count(shared_tensor)) {
      if (is_store && !TargetIsMusa(T.target)) {
        // For BulkStore, we should perform swizzle if possible.
        // TMA Store is always 1d like, we can directly use the last two
        // dimensions to analysis swizzling.
        int dim = shared_tensor->shape.size();
        const int64_t mat_stride = *as_const_int(shared_tensor->shape[dim - 2]);
        const int64_t mat_continuous =
            *as_const_int(shared_tensor->shape[dim - 1]);
        Layout swizzle_layout_2d = makeGemmABLayoutHopper(
            mat_stride, mat_continuous, mat_continuous,
            shared_tensor->dtype.bits(), /*k_inner=*/true);
        // If makeGemmABLayoutHopper returns a linear layout, fallback to
        // ComputeLinearLayout which handles arbitrary tensor shapes correctly.
        if (StructuralEqual()(
                swizzle_layout_2d,
                makeLinearLayout(Array<PrimExpr>{Integer(mat_stride),
                                                 Integer(mat_continuous)}))) {
          result_map.Set(shared_tensor, ComputeLinearLayout(shared_tensor));
        } else {
          result_map.Set(shared_tensor, ExpandLayoutToMatchBuffer(
                                            swizzle_layout_2d, shared_tensor));
        }
      } else if (level == InferLevel::kFree) {
        // Keep MUSA BulkStore shared layout linear. MUSA may select
        // swizzle-none for tma_store descriptor in some kernels; using a
        // swizzled producer layout here would cause store/readback mismatch.
        // create a new layout map for tma linear layout
        Layout linear_layout = ComputeLinearLayout(shared_tensor);
        result_map.Set(shared_tensor, linear_layout);
      }
    }
    return result_map;
  }

  // for LDSM/STSM, the layout was deduced from register layout
  // so we can directly apply the layout of normal copy
  // Use parallel op to infer the layout
  if (!par_op_.defined()) {
    arith::Analyzer analyzer;
    par_op_ = ParallelOp((MakeSIMTLoop(&analyzer)));
  }
  auto layout_map = par_op_->InferLayout(T, level);
  return layout_map;
}
// Shared stride validation for TMA bulk load/store.
bool CopyNode::CheckGlobalStrides(const Buffer &buffer,
                                  arith::Analyzer *analyzer) {
  Array<PrimExpr> strides = buffer->strides;
  if (strides.empty()) {
    PrimExpr stride = 1;
    strides.resize(buffer->shape.size());
    for (int i = static_cast<int>(buffer->shape.size()) - 1; i >= 0; --i) {
      strides.Set(i, stride);
      stride *= buffer->shape[i];
    }
  }

  if (!strides.empty() &&
      analyzer->CanProve(strides[strides.size() - 1] != 1,
                         arith::ProofStrength::kSymbolicBound)) {
    LOG(WARNING) << "TMA bulk copy requires contiguous innermost global stride"
                 << ", but got " << strides[strides.size() - 1]
                 << " for buffer " << buffer->name
                 << ", fallback to normal copy.";
    return false;
  }

  for (size_t i = 0; i + 1 < strides.size(); ++i) {
    PrimExpr stride_bytes =
        cast(DataType::Int(64), strides[i]) * buffer->dtype.bytes();
    if (analyzer->CanProve(
            FloorMod(stride_bytes, IntImm(DataType::Int(64), 16)) != 0,
            arith::ProofStrength::kSymbolicBound)) {
      LOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                   << stride_bytes << " for buffer " << buffer->name
                   << ", fallback to normal copy.";
      return false;
    }
    if (const int64_t *stride =
            as_const_int(analyzer->Simplify(stride_bytes))) {
      if (*stride >= (int64_t{1} << 40)) {
        LOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                     << stride_bytes << " for buffer " << buffer->name
                     << ", fallback to normal copy.";
        return false;
      }
    }
  }
  return true;
}

// Checks if this copy can be lowered to a Bulk Load (TMA) instruction.
// Requires: TMA support, global->shared scope, matching dtypes.
bool CopyNode::CheckBulkLoad(Target target, arith::Analyzer *analyzer,
                             bool check_last_dim) const {
  // 1. arch must have bulk copy support
  if (!TargetHasBulkCopy(target))
    return false;
  // 2. src and dst must be global and shared
  if (src.scope() != "global" ||
      (dst.scope() != "shared.dyn" && dst.scope() != "shared"))
    return false;
  // 3. check shape.
  // last dim of src * dtype.bits() must be a multiple of 16
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // now we check src (gmem) as tma box dim is deduced from src
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(src_range[src_range.size() - 1]->extent * src->dtype.bytes(),
                   16) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    LOG(WARNING)
        << "src range must have last dim multiple of 16 for tma bulk load "
        << src->name << " range " << src_range[src_range.size() - 1]->extent
        << " * " << src->dtype.bytes() << " % 16 != 0";
    return false;
  }

  // 4. src and dst must have the same dtype
  if (src->dtype != dst->dtype) {
    LOG(WARNING) << "src and dst must have the same dtype for tma load "
                 << src->name << " vs. " << dst->name << " dtype " << src->dtype
                 << " vs. " << dst->dtype << " will be fallback to normal copy";
    return false;
  }
  if (!CheckGlobalStrides(src, analyzer))
    return false;
  return true;
}

bool CopyNode::CheckBulkCopy1D(const Buffer &global_tensor,
                               const Buffer &shared_tensor,
                               const Array<Range> &global_range,
                               const Array<Range> &shared_range,
                               const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) const {

  // Step 1: check shared is contiguous. A reshaped buffer may not be present
  // as the direct layout-map key, so inspect layouts on aliases sharing data.
  bool shared_is_contiguous =
      IsSharedLayoutContiguousFor1DTMA(layout_map, shared_tensor, analyzer);
  // Step 2: check global is contiguous
  bool global_is_contiguous = true;
  bool global_not_full_dim_encounter = false;
  for (int i = global_range.size() - 1; i >= 0; i--) {
    if (!global_not_full_dim_encounter) {
      if (!analyzer->CanProve(global_range[i]->extent ==
                                      global_tensor->shape[i] &&
                                  global_range[i]->min == 0,
                              arith::ProofStrength::kSymbolicBound)) {
        global_not_full_dim_encounter = true;
      }
    } else {
      if (!analyzer->CanProve(global_range[i]->extent == 1,
                              arith::ProofStrength::kSymbolicBound)) {
        global_is_contiguous = false;
        break;
      }
    }
  }

  // Step 3: check element match and no OOB
  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }
  PrimExpr global_elements = 1;
  for (size_t i = 0; i < global_range.size(); i++) {
    global_elements *= global_range[i]->extent;
  }
  bool element_match =
      analyzer->CanProveEqual(shared_elements, global_elements);

  return (shared_is_contiguous && global_is_contiguous && element_match);
}

bool CopyNode::CheckBulkLoad1D(Target target, const LayoutMap &layout_map,
                               arith::Analyzer *analyzer) const {
  if (!CheckBulkLoad(target, analyzer, false))
    return false;
  auto global_tensor = src;
  auto shared_tensor = dst;
  auto global_range = src_range;
  auto shared_range = dst_range;
  return CheckBulkCopy1D(global_tensor, shared_tensor, global_range,
                         shared_range, layout_map, analyzer);
}

bool CopyNode::CheckBulkStore1D(Target target, const LayoutMap &layout_map,
                                arith::Analyzer *analyzer) const {
  if (!CheckBulkStore(target, analyzer, false))
    return false;
  auto shared_tensor = src;
  auto global_tensor = dst;
  auto shared_range = src_range;
  auto global_range = dst_range;
  return CheckBulkCopy1D(global_tensor, shared_tensor, global_range,
                         shared_range, layout_map, analyzer);
}

// Checks if this copy can be lowered to a Bulk Store (TMA) instruction.
// Requires: TMA support, shared->global scope, matching dtypes.
bool CopyNode::CheckBulkStore(Target target, arith::Analyzer *analyzer,
                              bool check_last_dim) const {
  // 1. arch must have bulk copy support
  if (!TargetHasBulkCopy(target))
    return false;
  // 2. src and dst must be shared.dyn and local.fragment
  if ((src.scope() != "shared.dyn" && src.scope() != "shared") ||
      dst.scope() != "global")
    return false;
  // 3. check shape.
  // last dim of dst * dtype.bits() must be a multiple of 16
  // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
  // now we check dst (gmem) as tma box dim is deduced from dst
  if (check_last_dim &&
      analyzer->CanProve(
          FloorMod(dst_range[dst_range.size() - 1]->extent * dst->dtype.bytes(),
                   16) != 0,
          arith::ProofStrength::kSymbolicBound)) {
    LOG(WARNING)
        << "dst range must have last dim multiple of 16 for tma bulk store "
        << dst->name << " range " << dst_range[dst_range.size() - 1]->extent
        << " * " << dst->dtype.bytes() << " % 16 != 0";
    return false;
  }
  // 4. src and dst must have the same dtype
  if (src->dtype != dst->dtype) {
    LOG(WARNING) << "src and dst must have the same dtype for tma store "
                 << src->name << " vs. " << dst->name << " dtype " << src->dtype
                 << " vs. " << dst->dtype << " will be fallback to normal copy";
    return false;
  }
  if (!CheckGlobalStrides(dst, analyzer))
    return false;
  return true;
}

// Checks if copy can use CUDA's Load Matrix (LDSM) instruction.
// Requires: LDMATRIX support, shared->fragment scope.
bool CopyNode::CheckLDSMCopy(Target target) const {
  return TargetHasLdmatrix(target) && IsSharedBuffer(src) &&
         IsFragmentBuffer(dst);
}

// Checks if copy can use CUDA's Store Matrix (STSM) instruction.
// Requires: STMATRIX support, fragment->shared scope.
bool CopyNode::CheckSTSMCopy(Target target) const {
  return TargetHasStmatrix(target) && IsFragmentBuffer(src) &&
         IsSharedBuffer(dst);
}

// Checks if copy can use tensor memory load (tcgen05.ld).
// Requires: tmem support, shared.tmem->fragment scope.
bool CopyNode::CheckTMemLoad(Target target) const {
  return TargetHasTmem(target) && src.scope() == "shared.tmem" &&
         IsFragmentBuffer(dst);
}

// Checks if copy can use tensor memory store (tcgen05.st).
// Requires: tmem support, fragment->shared.tmem scope.
bool CopyNode::CheckTMemStore(Target target) const {
  return TargetHasTmem(target) && IsFragmentBuffer(src) &&
         dst.scope() == "shared.tmem";
}

// Checks if copy can use cp.async global->shared path.
// Requirements:
// - target has async copy capability
// - source is global and destination is shared/shared.dyn
// - source/destination dtypes match
// - vectorized copy width (bytes) is one of {4, 8, 16}
// - if OOB guards are required, only a *uniform* (scalar) source predicate
//   is supported (dst must be in-bounds)
bool CopyNode::CheckCPAsyncCopyPreconditions() const {
  if (!IsGlobalBuffer(src) || !IsSharedBuffer(dst)) {
    return false;
  }
  if (src->dtype != dst->dtype) {
    return false;
  }
  return true;
}

bool CopyNode::CheckPipelineManagedCPAsyncCopy() const {
  return !GetIsTmaCopy() && !GetIsAsyncCopy() &&
         CheckCPAsyncCopyPreconditions();
}

bool CopyNode::CheckPipelineManagedCPAsyncCopy(
    Target target, arith::Analyzer *analyzer) const {
  return CheckPipelineManagedCPAsyncCopy() &&
         CheckCPAsyncCopy(target, LayoutMap(), analyzer);
}

bool CopyNode::CheckCPAsyncCopy(Target target, const LayoutMap &layout_map,
                                arith::Analyzer *analyzer) const {
  if (!TargetHasAsyncCopy(target)) {
    return false;
  }
  if (!CheckCPAsyncCopyPreconditions()) {
    return false;
  }
  // Skip vectorize size check here because, during the Infer Layout stage,
  // the layout is not stable and the vectorized size cannot be determined.
  return true;
}

// Selects the most specific copy instruction for the given target and buffers.
// Priority: BulkLoad1D, BulkStore1D, BulkLoad, BulkStore, LDSM, STSM,
// TMemLoad, TMemStore, CPAsync, Normal.
CopyInst CopyNode::GetCopyInst(Target target, const LayoutMap &layout_map,
                               arith::Analyzer *analyzer,
                               bool buffer_oob) const {
  // When is_tma_copy is set (from T.tma_copy()), force TMA path.
  if (GetIsTmaCopy()) {
    // Check if target is CuTeDSL backend
    bool is_cutedsl = TargetIsCuTeDSL(target);
    if (!is_cutedsl && !buffer_oob &&
        CheckBulkLoad1D(target, layout_map, analyzer)) {
      return CopyInst::kBulkLoad1D;
    } else if (!is_cutedsl && !buffer_oob &&
               CheckBulkStore1D(target, layout_map, analyzer)) {
      return CopyInst::kBulkStore1D;
    } else if (CheckBulkLoad(target, analyzer)) {
      return CopyInst::kBulkLoad;
    } else if (CheckBulkStore(target, analyzer) &&
               IsSharedLayoutRepresentableForTMABulkStore(layout_map, src,
                                                          analyzer)) {
      return CopyInst::kBulkStore;
    } else {
      LOG(FATAL) << "T.tma_copy() requires TMA-capable target and "
                    "global<->shared copy pattern, but TMA is not available "
                    "for src="
                 << src->name << ", dst=" << dst->name;
    }
  }

  bool is_async_copy = GetIsAsyncCopy();
  PrimExpr src_robust_desc = GetSourceRobustDesc();
  bool no_implicit_commit_wait = GetNoImplicitAsyncCommitWait();
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  bool disable_tma_lower =
      GetDisableTMA() ||
      pass_ctx->GetConfig<Bool>(kDisableTMALower, Bool(false)).value();

  if (src_robust_desc.defined()) {
    return CopyInst::kNormal;
  }

  if (is_async_copy || no_implicit_commit_wait) {
    bool cp_async_supported = CheckCPAsyncCopy(target, layout_map, analyzer);
    ICHECK(cp_async_supported)
        << "Explicit async copy semantics require cp.async lowering, but "
           "constraints were not satisfied. Got src="
        << src->name << " (scope=" << src.scope() << ", dtype=" << src->dtype
        << "), dst=" << dst->name << " (scope=" << dst.scope()
        << ", dtype=" << dst->dtype << ").";
    return CopyInst::kCPAsync;
  }

  // Plain T.copy does not auto-upgrade to TMA loads anymore. Store-side TMA
  // remains allowed because it is self-synchronized locally and does not
  // participate in pipeline producer scheduling. The deprecated global pass
  // config is still honored for backward compatibility. WS still rewrites
  // eligible load-side copies to T.tma_copy explicitly.
  if (!disable_tma_lower) {
    bool is_cutedsl = TargetIsCuTeDSL(target);
    if (!is_cutedsl && !buffer_oob &&
        CheckBulkStore1D(target, layout_map, analyzer)) {
      return CopyInst::kBulkStore1D;
    } else if (CheckBulkStore(target, analyzer) &&
               IsSharedLayoutRepresentableForTMABulkStore(layout_map, src,
                                                          analyzer)) {
      return CopyInst::kBulkStore;
    }
  }

  // Check tensor memory operations first (highest priority for SM100/Blackwell)
  if (CheckLDSMCopy(target)) {
    return CopyInst::kLDSM;
  } else if (CheckSTSMCopy(target)) {
    return CopyInst::kSTSM;
  } else if (CheckTMemLoad(target)) {
    return CopyInst::kTMemLoad;
  } else if (CheckTMemStore(target)) {
    return CopyInst::kTMemStore;
  } else {
    return CopyInst::kNormal;
  }
}

// Lowers the copy operation to PTX code by dispatching to specialized lowering
// functions.
Stmt CopyNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  Target target = T.target;
  auto copy_inst =
      GetCopyInst(target, T.layout_map, analyzer, /*buffer_oob=*/false);
  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    auto tmem_copy = LowerTmemCopy(T, analyzer);
    ICHECK(tmem_copy.defined()) << "Failed to lower tensor memory copy";
    return tmem_copy;
  } else if (copy_inst == CopyInst::kBulkLoad1D ||
             copy_inst == CopyInst::kBulkStore1D) {
    auto bulk_copy = LowerBulkCopy1D(T, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load 1d";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kBulkLoad ||
             copy_inst == CopyInst::kBulkStore) {
    auto bulk_copy = LowerBulkCopy(T, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load/store";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM) {
    auto ldsm_copy = LowerLDSMCopy(T, analyzer, copy_inst);
    ICHECK(ldsm_copy.defined()) << "Failed to lower ptx matrix copy";
    return ldsm_copy;
  } else if (copy_inst == CopyInst::kCPAsync) {
    auto cp_async_copy = LowerCPAsyncCopy(T, analyzer);
    ICHECK(cp_async_copy.defined()) << "Failed to lower cp.async copy";
    return cp_async_copy;
  } else if (copy_inst == CopyInst::kNormal) {
    return LowerNormalCopy(T, analyzer);
  } else {
    LOG(FATAL) << "Unsupported copy inst " << static_cast<int>(copy_inst);
  }
}

// Lowers copy to cp.async global->shared transfers.
// - T.copy annotated for cp.async keeps synchronous semantics by committing
//   and waiting after the loop.
// - T.async_copy commits but does not wait (explicit async semantics).
// - Copies annotated with kAsyncCopyNoImplicitCommitWait emit only cp.async;
//   an enclosing pass is responsible for commit/wait placement.
Stmt CopyNode::LowerCPAsyncCopy(const LowerArgs &T,
                                arith::Analyzer *analyzer) const {
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  bool enable_async_copy =
      pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
  bool no_implicit_commit_wait = GetNoImplicitAsyncCommitWait();
  bool explicit_async_semantics = no_implicit_commit_wait || GetIsAsyncCopy();
  if (!enable_async_copy && !explicit_async_semantics) {
    return LowerNormalCopy(T, analyzer);
  }

  auto simt_loop = MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);

  std::vector<InferLevel> levels = {InferLevel::kCommon, InferLevel::kStrict,
                                    InferLevel::kFree};
  for (auto level : levels) {
    par_op->InferLayout({T.target,
                         T.thread_bounds,
                         T.layout_map,
                         analyzer,
                         false,
                         T.buffer_remap,
                         {}},
                        level);
  }
  auto loop_layout = par_op->GetLoopLayout();
  Stmt lowered_loop =
      LowerParallelLoop(par_op->GetRoot(), loop_layout, T.thread_var, analyzer,
                        T.layout_map, par_op->GetPredicate(T.thread_var));

  bool async_without_implicit_commit_wait =
      no_implicit_commit_wait || GetIsAsyncCopy();
  auto inject_result =
      InjectPTXAsyncCopy(lowered_loop, /*enable_auto_async_copy=*/true,
                         async_without_implicit_commit_wait);
  Stmt cp_async_loop = inject_result.stmt;
  if (!inject_result.injected_ptx_async_copy) {
    LOG(WARNING) << "cp.async rewrite miss for copy src=" << src->name
                 << " (scope=" << src.scope() << ", dtype=" << src->dtype
                 << "), dst=" << dst->name << " (scope=" << dst.scope()
                 << ", dtype=" << dst->dtype
                 << "), no_implicit_async_commit_wait="
                 << no_implicit_commit_wait
                 << ", is_async_copy=" << GetIsAsyncCopy();
    if (no_implicit_commit_wait) {
      LOG(WARNING)
          << "Pipeline-managed async copy fallback to normal copy because "
             "cp.async rewrite found no eligible global->shared store.";
      return lowered_loop;
    }
    if (explicit_async_semantics) {
      LOG(FATAL) << "Explicit async copy semantics require cp.async lowering, "
                    "but no eligible global->shared store was rewritten.";
    }
    LOG(WARNING) << "Fallback to normal copy because cp.async rewrite found "
                    "no eligible global->shared store.";
    return LowerNormalCopy(T, analyzer);
  }
  if (no_implicit_commit_wait) {
    return cp_async_loop;
  }
  if (GetIsAsyncCopy()) {
    Stmt commit_group =
        Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
    return SeqStmt({cp_async_loop, commit_group});
  }
  return cp_async_loop;
}

// Lowers the copy using standard load/store with loop transformations.
Stmt CopyNode::LowerNormalCopy(const LowerArgs &T,
                               arith::Analyzer *analyzer) const {
  using namespace tvm::transform;
  PassContext pass_ctx = PassContext::Current();
  bool disable_safe_copy_predication =
      pass_ctx->GetConfig<Bool>(kDisableSafeCopyPredication, Bool(false))
          .value();

  auto lower_single_copy = [&](const CopyNode &node) -> Stmt {
    bool is_cpu_target = T.target->GetTargetDeviceType() == kDLCPU;
    auto simt_loop = node.MakeSIMTLoop(analyzer, disable_safe_copy_predication);
    auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));

    Stmt lowered_body;
    auto par_op = ParallelOp(fused_loop);

    if (is_cpu_target || IsLocalBuffer(node.src) || IsLocalBuffer(node.dst)) {
      if (IsLocalBuffer(node.src) && !IsLocalBuffer(node.dst)) {
        // A conflict write only occurs when multiple threads write to the same
        // global address. If any dst_range dimension's min depends on the
        // thread variable, each thread targets a distinct location and there is
        // no conflict.
        bool dst_depends_on_thread = false;
        for (const auto &range : node.dst_range) {
          if (tir::UsesVar(range->min, [&](const VarNode *v) {
                return v == T.thread_var.get();
              })) {
            dst_depends_on_thread = true;
            break;
          }
        }
        if (!dst_depends_on_thread) {
          LOG(WARNING) << "Copy from local buffer `" << node.src->name
                       << "` to " << node.dst.scope() << " buffer `"
                       << node.dst->name << "` may cause conflicted write.";
        }
      }
      lowered_body = VectorizeLoop(fused_loop, T.layout_map);
    } else {
      std::vector<InferLevel> levels = {InferLevel::kCommon,
                                        InferLevel::kStrict, InferLevel::kFree};
      for (auto level : levels) {
        par_op->InferLayout({T.target, T.thread_bounds, T.layout_map, analyzer,
                             false, T.buffer_remap, T.let_var_to_expr, false},
                            level);
      }
      auto loop_layout = par_op->GetLoopLayout();
      lowered_body = LowerParallelLoop(par_op->GetRoot(), loop_layout,
                                       T.thread_var, analyzer, T.layout_map,
                                       par_op->GetPredicate(T.thread_var));
    }

    Stmt body = lowered_body;
    PrimExpr src_robust_desc = node.GetSourceRobustDesc();
    if (src_robust_desc.defined()) {
      ICHECK(TargetIsMusa(T.target))
          << "src_robust_desc is only supported when targeting MUSA.";
      ICHECK(node.src.scope() == "global")
          << "src_robust_desc requires a global-memory source, but got `"
          << node.src.scope() << "`.";
      body = AttrStmt(node.src->data, attr::kSourceRobustDesc, src_robust_desc,
                      body);
    }
    if (node.GetForceAsyncCopy()) {
      body = AttrStmt(make_zero(DataType::Int(32)), attr::kForceAsyncCopy, 1,
                      body);
    }
    return body;
  };

  auto dst_k_major_opt = GetLayoutKMajor(T, dst);
  auto dst_sqmma_opt = GetLayoutSQMMA(T, dst);
  bool is_musa_sqmma_norm_copy =
      TargetIsMusa(T.target) && src.scope() == "global" &&
      (dst.scope() == "shared" || dst.scope() == "shared.dyn") &&
      !src_range.empty() && !dst_range.empty() && dst_k_major_opt.has_value() &&
      dst_sqmma_opt.value_or(false);
  bool dst_is_k_major = dst_k_major_opt.value_or(true);

  bool need_sqmma_split = false;
  int split_inner_extent = -1;
  int split_count = -1;
  ICHECK(!src_range.empty() && !dst_range.empty());
  size_t split_src_dim_idx = src_range.size() - 1;
  size_t split_dst_dim_idx = dst_range.size() - 1;
  if (is_musa_sqmma_norm_copy) {
    if (!dst_is_k_major) {
      int sqmma_inst_split = GetLayoutSqmmaInstSplit(T, dst).value_or(-1);
      if (sqmma_inst_split > 0) {
        auto dst_split_extent =
            as_const_int(dst_range[split_dst_dim_idx]->extent);
        if (dst_split_extent != nullptr &&
            (*dst_split_extent) > sqmma_inst_split) {
          auto src_split_extent =
              as_const_int(src_range[split_src_dim_idx]->extent);
          bool dst_divisible = ((*dst_split_extent) % sqmma_inst_split) == 0;
          bool src_divisible = src_split_extent == nullptr ||
                               ((*src_split_extent) % sqmma_inst_split) == 0;
          if (dst_divisible && src_divisible) {
            need_sqmma_split = true;
            split_inner_extent = sqmma_inst_split;
            split_count = (*dst_split_extent) / sqmma_inst_split;
          }
        }
      }
    } else {
      int elem_bytes = dst->dtype.bytes();
      int max_inner_elems = elem_bytes > 0 ? 256 / elem_bytes : 0;
      auto dst_split_extent =
          as_const_int(dst_range[split_dst_dim_idx]->extent);
      if (max_inner_elems > 0 && dst_split_extent != nullptr &&
          (*dst_split_extent) * elem_bytes > 256) {
        auto src_split_extent =
            as_const_int(src_range[split_src_dim_idx]->extent);
        bool dst_divisible = ((*dst_split_extent) % max_inner_elems) == 0;
        bool src_divisible = src_split_extent == nullptr ||
                             ((*src_split_extent) % max_inner_elems) == 0;
        if (dst_divisible && src_divisible) {
          need_sqmma_split = true;
          split_inner_extent = max_inner_elems;
          split_count = (*dst_split_extent) / max_inner_elems;
        }
      }
    }
  }

  if (!need_sqmma_split) {
    return lower_single_copy(*this);
  }

  auto split_op = tvm::ffi::make_object<CopyNode>(*this);
  Var split_var("sqmma_split");
  auto new_src_range = split_op->src_range;
  auto new_dst_range = split_op->dst_range;
  new_src_range.Set(split_src_dim_idx,
                    Range::FromMinExtent(new_src_range[split_src_dim_idx]->min +
                                             split_var * split_inner_extent,
                                         split_inner_extent));
  new_dst_range.Set(split_dst_dim_idx,
                    Range::FromMinExtent(new_dst_range[split_dst_dim_idx]->min +
                                             split_var * split_inner_extent,
                                         split_inner_extent));
  split_op->src_range = new_src_range;
  split_op->dst_range = new_dst_range;
  return For(split_var, 0, split_count, ForKind::kUnrolled,
             lower_single_copy(*split_op));
}

// Lowers copy to LDSM/STSM (warp-level 8x8 matrix) instructions.
// Falls back to LowerNormalCopy if hardware constraints are not met.
Stmt CopyNode::LowerLDSMCopy(const LowerArgs &T, arith::Analyzer *analyzer,
                             CopyInst copy_inst) const {
  ICHECK(copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_ldmatrix = copy_inst == CopyInst::kLDSM;

  // Check no predicates
  Array<IterVar> loop_vars = MakeIterVars();
  if (loop_vars.size() < 2) {
    // cannot support 1-d case
    return LowerNormalCopy(T, analyzer);
  }
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);
  if (src_predicate.defined() || dst_predicate.defined()) {
    // stmatrix and ldmatrix can only support no predicate
    return LowerNormalCopy(T, analyzer);
  }

  Buffer shared_tensor = is_ldmatrix ? src : dst;
  Buffer local_tensor = is_ldmatrix ? dst : src;
  Array<Range> local_region = is_ldmatrix ? src_range : dst_range;
  bool is_full_range = true;
  for (size_t i = 0; i < local_region.size(); i++) {
    if (!analyzer->CanProveEqual(local_region[i]->extent,
                                 local_tensor->shape[i])) {
      is_full_range = false;
      break;
    }
  }
  if (!is_full_range) {
    // ldmatrix/stmatrix can only support full range, will be fallback to
    // normal copy
    return LowerNormalCopy(T, analyzer);
  }

  Array<PrimExpr> local_indices = MakeIndices(loop_vars, is_ldmatrix ? 1 : 0);
  Fragment local_layout = Downcast<Fragment>(T.layout_map[local_tensor]);
  Array<PrimExpr> local_indices_transformed =
      local_layout->Forward(local_indices);
  local_tensor = T.buffer_remap[local_tensor];
  // currently only support 1-d case
  if (local_layout->OutputDim() != 1) {
    // TMA ldmatrix/stmatrix cannot support non-1-d layout, will be fallback to
    // normal copy
    return LowerNormalCopy(T, analyzer);
  }

  Array<PrimExpr> shared_indices = MakeIndices(loop_vars, is_ldmatrix ? 0 : 1);
  // Check local_layout follows 8x8 layout
  // LDSM/STSM instructions require 8x8 matrix fragment layout
  // This matches the warp-level matrix multiplication pattern used in tensor
  // cores We check both normal and transposed layouts to support different
  // access patterns
  bool is_transposed;
  IterVar col_var = loop_vars[loop_vars.size() - 1];
  IterVar row_var = loop_vars[loop_vars.size() - 2];
  PrimExpr local_layout_thread_map =
      FloorMod(local_layout->ForwardThread(local_indices, std::nullopt), 32);
  PrimExpr matrix_8x8_thread_map = makeGemmFragment8x8()->ForwardThread(
      {FloorMod(row_var, 8), FloorMod(col_var, 8)}, std::nullopt);
  PrimExpr matrix_8x8_thread_map_trans =
      makeGemmFragment8x8Transposed()->ForwardThread(
          {FloorMod(row_var, 8), FloorMod(col_var, 8)}, std::nullopt);
  PrimExpr local_indices_flattened =
      local_tensor.OffsetOf(local_indices_transformed).back();
  if (analyzer->CanProveEqual(matrix_8x8_thread_map, local_layout_thread_map) &&
      IndicesCanVectorize(local_indices_flattened, col_var->var,
                          col_var->dom->extent, 2, analyzer)) {
    is_transposed = false;
  } else if (analyzer->CanProveEqual(matrix_8x8_thread_map_trans,
                                     local_layout_thread_map) &&
             IndicesCanVectorize(local_indices_flattened, row_var->var,
                                 row_var->dom->extent, 2, analyzer)) {
    is_transposed = true;
  } else {
    // TMA ldmatrix/stmatrix cannot support non-8x8 layout, will be fallback to
    // normal copy
    return LowerNormalCopy(T, analyzer);
  }
  // Check shared_layout is 16 bytes continuous
  // LDSM/STSM instructions require 16-byte aligned data (half-precision floats)
  // This is a hardware constraint for matrix load/store operations
  if (shared_tensor->dtype.bytes() != 2) {
    // TMA ldmatrix/stmatrix cannot support non-16 bytes continuous layout, will
    // be fallback to normal copy
    return LowerNormalCopy(T, analyzer);
  }
  PrimExpr flattened_indice = shared_tensor.OffsetOf(shared_indices).back();
  if (!IndicesCanVectorize(flattened_indice, loop_vars.back()->var,
                           loop_vars.back()->dom->extent, 8, analyzer)) {
    // TMA ldmatrix/stmatrix cannot support non-16 bytes continuous layout, will
    // be fallback to normal copy
    return LowerNormalCopy(T, analyzer);
  }

  // Can only support local_range to be a full range
  for (size_t i = 0; i < dst_range.size(); i++) {
    if (!is_zero(dst_range[i]->min) ||
        !analyzer->CanProveEqual(dst_range[i]->extent, dst->shape[i]))
      // TMA ldmatrix/stmatrix cannot support non-full range, will be fallback
      // to normal copy
      return LowerNormalCopy(T, analyzer);
  }

  // Do the lowering here, try vectorized ldmatrix/stmatrix by 4/2/1
  // now, local_tensor is local instead of shared.
  PrimExpr extent = local_tensor->shape[0];
  int num = 1;
  if (analyzer->CanProveEqual(FloorMod(extent, 8), 0))
    // 16x16 -> full warp, we use x4, for 32 threads in a warp, each thread can
    // hold 4 elements
    num = 4;
  else if (analyzer->CanProveEqual(FloorMod(extent, 4), 0))
    // 8x16 -> half warp, we use x2, for 32 threads in a warp, each thread can
    // hold 2 elements
    num = 2;

  Array<PrimExpr> args;
  const Op &op = is_ldmatrix ? tl::ptx_ldmatrix() : tl::ptx_stmatrix();
  args.push_back(static_cast<int>(is_transposed));
  args.push_back(num);

  // Create shared address with regard to local address
  // if not transpose
  // coords = Inverse(base + 2 * (thread / 8) % num, warp + (thread % 8) * 4))
  // if transpose
  // coords = Inverse(base + 2 * (thread / 8) % num + thread % 2, warp + thread
  // % 8 / 2)
  Var local_iter("i");
  Layout inv = local_layout->Inverse();
  Array<PrimExpr> shared_coords;
  PrimExpr warp = FloorDiv(T.thread_var, 32) * 32;
  if (!is_transposed) {
    auto local_index = analyzer->Simplify(
        local_iter * 2 * num + 2 * FloorMod(FloorDiv(T.thread_var, 8), num));
    auto thread_index =
        analyzer->Simplify(warp + FloorMod(T.thread_var, 8) * 4);
    shared_coords = inv->Forward({local_index, thread_index});
  } else {
    auto local_index = analyzer->Simplify(
        local_iter * 2 * num + 2 * FloorMod(FloorDiv(T.thread_var, 8), num) +
        FloorMod(T.thread_var, 2));
    auto thread_index =
        analyzer->Simplify(warp + FloorDiv(FloorMod(T.thread_var, 8), 2));
    shared_coords = inv->Forward({local_index, thread_index});
  }
  shared_coords.pop_back(); // remove rep
  PrimExpr shared_addr = shared_tensor.access_ptr(
      is_ldmatrix ? 1 : 2, DataType::Handle(), 1,
      shared_tensor.OffsetOf(shared_coords).back(), PrimExpr(2 * num));
  args.push_back(shared_addr);

  if (is_ldmatrix) {
    // Can only support same dtype for ldmatrx
    if (local_tensor->dtype != shared_tensor->dtype) {
      // TMA ldmatrix cannot support different dtype, will be fallback to normal
      // copy
      return LowerNormalCopy(T, analyzer);
    }
    PrimExpr local_addr = local_tensor.access_ptr(
        2, DataType::Handle(), 1, local_iter * 2 * num, PrimExpr(2 * num));
    args.push_back(local_addr);
  } else {
    for (int i = 0; i < num; i++) {
      PrimExpr value0 =
          BufferLoad(local_tensor, {local_iter * 2 * num + 2 * i});
      PrimExpr value1 =
          BufferLoad(local_tensor, {local_iter * 2 * num + 2 * i + 1});
      if (local_tensor->dtype != shared_tensor->dtype) {
        value0 = Cast(shared_tensor->dtype, value0);
        value1 = Cast(shared_tensor->dtype, value1);
      }
      PrimExpr value_packed =
          Call(DataType::Int(32), pack_b16(), {value0, value1});
      args.push_back(value_packed);
    }
  }

  auto body = Evaluate(Call(DataType::Handle(), op, args));
  For for_node =
      For(local_iter, 0, FloorDiv(extent, 2 * num), ForKind::kSerial, body);
  for_node = PragmaUnrollLoop(for_node);
  auto range = T.thread_bounds;
  if (range.defined()) {
    auto thread_var = T.thread_var;
    auto thread_var_with_offset = thread_var - range->min;
    for_node.CopyOnWrite()->body =
        Substitute(for_node->body, {{thread_var, thread_var_with_offset}});
  }
  return for_node;
}

// Lowers tensor memory copy operations (tcgen05.ld/st/cp).
// Currently only tcgen05.ld is fully supported.
Stmt CopyNode::LowerTmemCopy(const LowerArgs &T,
                             arith::Analyzer *analyzer) const {
  if (src.scope() != "shared.tmem" && dst.scope() != "shared.tmem") {
    return Stmt();
  }
  ICHECK(TargetHasTmem(T.target)) << "Target " << T.target->ToDebugString()
                                  << " does not support tensor memory copy";

  // Decide copy type
  bool is_ld = false; // tcgen05.ld (tensor memory -> register)
  bool is_st = false; // tcgen05.st (register -> tensor memory)
  bool is_cp = false; // tcgen05.cp (shared memory -> tensor memory)
  bool src_needs_pack =
      16 == src->dtype.bits(); // if needs .pack::16b when is_ld
  bool dst_needs_unpack =
      16 == dst->dtype.bits(); // if needs .unpack::16b when is_st

  if (src.scope() == "shared.tmem" && IsFragmentBuffer(dst)) {
    is_ld = true;
  } else if (IsFragmentBuffer(src) && dst.scope() == "shared.tmem") {
    is_st = true;
  } else if (src.scope() == "shared.dyn" && dst.scope() == "shared.tmem") {
    is_cp = true;
  } else {
    LOG(FATAL) << "Unsupported tensor memory copy: "
               << "src scope = " << src.scope()
               << ", dst scope = " << dst.scope();
  }
  // Currently tcgen05.cp is not supported
  // TODO (mzw) Support tcgen05.cp
  ICHECK(!is_cp)
      << "Copy from shared memory to tensor memory is not supported yet";
  // Extract loop variables and ranges
  Array<IterVar> loop_vars = MakeIterVars();
  ICHECK(loop_vars.size() == 2) << "Only support 2D tensor memory copy, got "
                                << loop_vars.size() << " dimensions";
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = MakePredicate(analyzer, loop_vars, dst->shape, 1);
  ICHECK(!src_predicate.defined() && !dst_predicate.defined())
      << "Tensor memory copy does not support predicates, got " << src_predicate
      << " and " << dst_predicate;
  ICHECK(is_const_int(loop_vars[0]->dom->min) &&
         is_const_int(loop_vars[0]->dom->extent) &&
         is_const_int(loop_vars[1]->dom->min) &&
         is_const_int(loop_vars[1]->dom->extent))
      << "Tensor memory copy requires loop bounds to be constant integers";
  int64_t logical_row_min = *as_const_int(loop_vars[0]->dom->min);
  int64_t logical_row_extent = *as_const_int(loop_vars[0]->dom->extent);
  int64_t logical_col_min = *as_const_int(loop_vars[1]->dom->min);
  int64_t logical_col_extent = *as_const_int(loop_vars[1]->dom->extent);

  // Extract thread bounds
  constexpr int WARP_SIZE = 32; // Set to 32 since only sm100 is supported
  constexpr int WARPGROUP_SIZE = 4 * WARP_SIZE;
  ICHECK(is_const_int(T.thread_bounds->extent))
      << "Tensor memory copy requires thread_bounds->extent (num_threads) to "
         "be constant integers";
  int num_threads = *as_const_int(T.thread_bounds->extent);
  ICHECK(analyzer->CanProveEqual(FloorMod(T.thread_bounds->min, WARPGROUP_SIZE),
                                 0) &&
         num_threads % WARPGROUP_SIZE == 0)
      << "Tensor memory copy requires thread bounds to be aligned to "
         "warpgroups, but found "
      << "thread range = " << T.thread_bounds;

  // TODO (mzw) Buffer remap for shared.dyn when is_cp is true?

  // Determine tmem and register buffers based on copy direction
  Buffer tmem_buf = is_ld ? src : dst;
  Buffer reg_buf = is_ld ? dst : src;
  int tmem_side = is_ld ? 0 : 1;
  bool needs_pack_unpack = is_ld ? src_needs_pack : dst_needs_unpack;

  // Retrieve layout
  ICHECK(T.layout_map.count(tmem_buf)) << "Tmem buffer " << tmem_buf->name
                                       << " does not have a layout specified";
  ICHECK(T.layout_map.count(reg_buf)) << "Register buffer " << reg_buf->name
                                      << " does not have a layout specified";
  Layout tmem_layout = T.layout_map[tmem_buf];
  Fragment reg_layout = Downcast<Fragment>(T.layout_map[reg_buf]);

  // Check layout
  Array<PrimExpr> logical_indices = MakeIndices(loop_vars, tmem_side);
  Array<PrimExpr> phy_indices =
      tmem_layout->Forward(logical_indices); // "phy" for "physical"

  // Analyse the range of tmem_phy_row and tmem_phy_col
  arith::ConstIntBound phy_row_bounds =
      analyzer->const_int_bound(phy_indices[0]);
  arith::ConstIntBound phy_col_bounds =
      analyzer->const_int_bound(phy_indices[1]);
  int tmem_phy_row_min = phy_row_bounds->min_value;
  int tmem_phy_row_max = phy_row_bounds->max_value;
  int tmem_phy_col_min = phy_col_bounds->min_value;
  int tmem_phy_col_max = phy_col_bounds->max_value;
  int tmem_phy_row_extent = tmem_phy_row_max - tmem_phy_row_min + 1;
  int tmem_phy_col_extent = tmem_phy_col_max - tmem_phy_col_min + 1;
  Range row_dom = Range(tmem_phy_row_min, tmem_phy_row_max + 1);
  Range col_dom = Range(tmem_phy_col_min, tmem_phy_col_max + 1);

  bool have_succeeded = false;
  Stmt body;

  auto try_tcgen05_instruction = [&](Tcgen05Meta meta) {
    if (have_succeeded) {
      return;
    }
    if (tmem_phy_row_min != 0 || tmem_phy_row_max != 127) {
      return;
    }
    if (tmem_phy_col_min % meta.width != 0 ||
        (tmem_phy_col_max + 1) % meta.width != 0) {
      return;
    }

    for (int num_useful_wgs = num_threads / WARPGROUP_SIZE; num_useful_wgs >= 1;
         num_useful_wgs--) {
      int num_useful_threads = num_useful_wgs * WARPGROUP_SIZE;
      auto [is_success, target_frag, num_chunks_each_wg] = expandTcgen05Layout(
          meta, tmem_phy_col_extent, num_useful_threads, row_dom, col_dom);
      if (!is_success) {
        continue;
      }

      PrimExpr target_thread =
          target_frag->ForwardThread(phy_indices, std::nullopt);
      PrimExpr reg_thread =
          reg_layout->ForwardThread(logical_indices, std::nullopt);
      if (!analyzer->CanProveEqual(target_thread, reg_thread)) {
        continue;
      }
      PrimExpr target_reg = target_frag->Forward(phy_indices)[0];
      PrimExpr reg_val = reg_layout->Forward(logical_indices)[0];
      if (!analyzer->CanProveEqual(target_reg, reg_val)) {
        continue;
      }

      // All checks passed, we can use this instruction
      PrimExpr relative_wg_idx =
          FloorDiv(Sub(T.thread_var, T.thread_bounds->min), WARPGROUP_SIZE);
      PrimExpr col_offset =
          num_useful_threads == WARPGROUP_SIZE
              ? PrimExpr(0)
              : relative_wg_idx * (num_chunks_each_wg * meta.width);
      have_succeeded = true;
      Array<PrimExpr> args;
      // For tcgen05_st, bf16 data should be stored packed (without
      // unpack::16b) so MMA TS reads correctly packed bf16 from TMEM columns.
      // For tcgen05_ld, pack::16b is still needed when reading unpacked data.
      bool use_pack_unpack_modifier = is_ld ? needs_pack_unpack : false;
      const char *bool_str = use_pack_unpack_modifier ? "true" : "false";
      int effective_chunks =
          needs_pack_unpack ? num_chunks_each_wg / 2 : num_chunks_each_wg;
      args.push_back(StringImm(meta.intrinsics_name + "<" +
                               std::to_string(effective_chunks) + ", " +
                               bool_str + ">"));
      args.push_back(
          BufferLoad(tmem_buf, {(int)logical_row_min,
                                (int)logical_col_min})); // Will be translated
                                                         // later in
                                                         // lower_shared_tmem
                                                         // pass
      args.push_back(col_offset);
      int reg_access_mode = is_ld ? 2 : 1;
      args.push_back(reg_buf.access_ptr(reg_access_mode, DataType::Handle(), 1,
                                        0, PrimExpr(tmem_phy_col_extent)));

      Stmt call =
          Evaluate(Call(DataType::Handle(), builtin::call_extern(), args));
      if (num_useful_threads != num_threads) {
        body =
            IfThenElse(T.thread_var < T.thread_bounds->min + num_useful_threads,
                       call, // No-op for unused threads
                       Stmt());
      } else {
        body = call;
      }
      break;
    }
  };

  if (is_ld) {
    try_tcgen05_instruction(getTcgen05MetaLd_32dp32b());
    try_tcgen05_instruction(getTcgen05MetaLd_32dp64b());
    try_tcgen05_instruction(getTcgen05MetaLd_32dp128b());
    try_tcgen05_instruction(getTcgen05MetaLd_32dp256b());
  } else {
    try_tcgen05_instruction(getTcgen05MetaSt_32dp32b());
    try_tcgen05_instruction(getTcgen05MetaSt_32dp64b());
    try_tcgen05_instruction(getTcgen05MetaSt_32dp128b());
    try_tcgen05_instruction(getTcgen05MetaSt_32dp256b());
  }

  ICHECK(have_succeeded) << "Failed to find a suitable instruction for tcgen05."
                         << (is_ld ? "ld" : "st") << ". Check your layout.";

  return body;
}

#include "copy_tma_impl.inc"

void CopyNode::CollectFragmentLayouts(const PrimExpr &expr,
                                      const Map<Var, PrimExpr> &let_var_to_expr,
                                      const LayoutMap &existing_layouts,
                                      PrimExpr thread_extent,
                                      Range thread_bounds,
                                      Map<Buffer, Layout> &result_map) const {
  PostOrderVisit(expr, [&](const ObjectRef &node) {
    if (auto bl = node.as<BufferLoadNode>()) {
      if (IsFragmentBuffer(bl->buffer) && !existing_layouts.count(bl->buffer) &&
          !result_map.count(bl->buffer)) {
        auto f = Fragment::FullyReplicated(bl->buffer->shape, thread_extent);
        result_map.Set(bl->buffer, f->BindThreadRange(thread_bounds));
      }
    } else if (auto var_node = node.as<VarNode>()) {
      auto var = tvm::ffi::GetRef<Var>(var_node);
      if (let_var_to_expr.count(var)) {
        CollectFragmentLayouts(let_var_to_expr[var], let_var_to_expr,
                               existing_layouts, thread_extent, thread_bounds,
                               result_map);
      }
    }
  });
}

// Register the Copy operation with TVM's TIR system
// This makes the copy operation available for use in TVM programs
// - Takes 5 inputs: src_buffer, dst_buffer, coalesced_width, disable_tma,
// eviction_policy
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(Copy, copy)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.tileop.async_copy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "async_copy")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_async_copy",
                                       IntImm(DataType::Int(32), 1));
                               return Copy(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Register the tma_copy operation — same as copy but forces TMA path
// and emits only expect_tx + tma_load (no wait).
TVM_REGISTER_OP("tl.tileop.tma_copy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "tma_copy")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_tma_copy",
                                       IntImm(DataType::Int(32), 1));
                               return Copy(args, ann);
                             })
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

// Layout inference hook - returns empty map (no layout suggestions).
LayoutMap Conv2DIm2ColOpNode::InferLayout(const LayoutInferArgs &T,
                                          InferLevel level) const {
  return {};
}

// Register the Conv2DIm2Col operation with TVM's TIR system
// This operation performs im2col transformation for 2D convolutions using TMA
// - Takes 9 inputs: src_buffer, dst_buffer, nhw_step, c_step, kernel, stride,
// dilation, padding, eviction_policy
// - Marked as opaque since it has side effects (memory writes)
TIR_REGISTER_TL_TILE_OP(Conv2DIm2ColOp, c2d_im2col)
    .set_num_inputs(9)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() {
  CopyNode::RegisterReflection();
  Conv2DIm2ColOpNode::RegisterReflection();
}
} // namespace tl
} // namespace tvm
