/*!
 * \file tl/backend/musa/op/copy.cc
 * \brief MUSA implementation for tl.copy lowering.
 */

#include "op/copy.h"

#include "backend/musa/op/copy.h"
#include "layout/tcgen05_layout.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "target/musa.h"
#include "target/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"
#include "transform/loop_vectorize.h"
#include "transform/ptx_async_copy_injector.h"

#include <tvm/tir/analysis.h>
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include <cstdint>
#include <sstream>
#include <vector>

namespace tvm {
namespace tl {

using namespace tir;

namespace {

PrimExpr MakeTmaLeaderCondition(PrimExpr thread_extent) {
  return Call(DataType::Bool(), tl_shuffle_elect(), {std::move(thread_extent)});
}

PrimExpr TMABytesFromElements(PrimExpr elements, DataType dtype) {
  PrimExpr elements_i64 = cast(DataType::Int(64), elements);
  int bits = dtype.bits();
  if (bits % 8 == 0) {
    return elements_i64 * IntImm(DataType::Int(64), bits / 8);
  }
  return FloorDiv(elements_i64 * IntImm(DataType::Int(64), bits) +
                      IntImm(DataType::Int(64), 7),
                  IntImm(DataType::Int(64), 8));
}

int64_t TMABytesFromElements(int64_t elements, DataType dtype) {
  return (elements * dtype.bits() + 7) / 8;
}

int64_t TMAElementsForBytes(int64_t bytes, DataType dtype) {
  ICHECK_EQ((bytes * 8) % dtype.bits(), 0)
      << bytes << " bytes cannot be represented as whole elements of " << dtype;
  return bytes * 8 / dtype.bits();
}

PrimExpr GetCopyMbarPhaseExpr(const Map<String, ObjectRef> &annotations,
                              const LowerArgs &T) {
  PrimExpr phase = T.mbar_phase_expr;
  if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations)) {
    phase = explicit_phase.value();
  }
  return phase;
}

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto val = op.annotations.Get(key)) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value != 0;
    }
  }
  return false;
}

bool GetDisableTMA(const CopyNode &op) {
  return GetBoolAnnotation(op, "disable_tma");
}

bool GetIsTmaCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "is_tma_copy");
}

int GetEvictionPolicy(const CopyNode &op) {
  if (auto val = op.annotations.Get("eviction_policy")) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value;
    }
  }
  return 0; // default: evict_normal
}

int GetInnerCachePolicy(const CopyNode &op) {
  if (auto val = op.annotations.Get("inner_cache_policy")) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value;
    }
  }
  return 2; // default: cache_normal
}

int GetOuterCachePolicy(const CopyNode &op) {
  if (auto val = op.annotations.Get("outer_cache_policy")) {
    if (auto int_val = val->as<IntImmNode>()) {
      return int_val->value;
    }
  }
  return 2; // default: cache_normal
}

bool GetForceAsyncCopy(const CopyNode &op) {
  return GetBoolAnnotation(op, "force_async_copy");
}

PrimExpr GetSourceRobustDesc(const CopyNode &op) {
  if (auto val = op.annotations.Get("src_robust_desc")) {
    return Downcast<PrimExpr>(val.value());
  }
  return PrimExpr();
}

PrimExpr GetTmaBarrier(const CopyNode &op) {
  if (auto val = op.annotations.Get("barrier")) {
    return Downcast<PrimExpr>(val.value());
  }
  return IntImm(DataType::Int(32), 0);
}

bool GetIsAsyncCopy(const CopyNode &op) {
  if (GetBoolAnnotation(op, "is_async_copy")) {
    return true;
  }
  // Backward-compatibility with historical annotation key.
  return GetBoolAnnotation(op, "force_cp_async");
}

bool GetNoImplicitAsyncCommitWait(const CopyNode &op) {
  return GetBoolAnnotation(op, attr::kAsyncCopyNoImplicitCommitWait);
}

int to_MUtensorDescriptorDataType(DataType dtype) {
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

Optional<Layout> GetActiveLayout(const LowerArgs &T, const Buffer &buf) {
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

bool IsMatrixLinearLayout(const Layout &layout) {
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

template <typename TValue>
Optional<TValue> GetLayoutHintByKey(const Map<Layout, TValue> &hint_map,
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

Optional<bool> GetLayoutKMajor(const LowerArgs &T, const Buffer &buf) {
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

Optional<bool> GetLayoutSQMMA(const LowerArgs &T, const Buffer &buf) {
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

Optional<int> GetLayoutSqmmaInstSplit(const LowerArgs &T, const Buffer &buf) {
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

} // namespace

namespace musa {

struct TMAIm2ColDesc {
  size_t rank;
  int data_type;
  Array<PrimExpr> global_shape;
  Array<PrimExpr> global_stride;
  Array<PrimExpr> elem_stride;
  Array<PrimExpr> lower_corner;
  Array<PrimExpr> upper_corner;
  PrimExpr global_addr;
  int smem_box_pixel;
  int smem_box_channel;
  int swizzle;
  int interleave;
  int oob_fill;
  int l2_promotion;

  Array<PrimExpr> EncodeCallArgs() const {
    Array<PrimExpr> args;
    args.reserve(rank * 5 + 5);

    args.push_back(data_type);
    args.push_back(static_cast<int>(rank));
    args.push_back(global_addr);
    for (auto e : global_shape)
      args.push_back(e);
    for (auto e : global_stride)
      args.push_back(e);
    for (auto e : elem_stride)
      args.push_back(e);
    for (auto e : lower_corner)
      args.push_back(e);
    for (auto e : upper_corner)
      args.push_back(e);
    args.push_back(smem_box_pixel);
    args.push_back(smem_box_channel);
    args.push_back(interleave);
    args.push_back(swizzle);
    args.push_back(l2_promotion);
    args.push_back(oob_fill);

    return args;
  }
};

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                               InferLevel level);

  static Stmt Lower(const CopyNode &op, const LowerArgs &T,
                    arith::Analyzer *analyzer);

private:
  static Layout ComputeLinearLayout(const Buffer &shared_tensor);

  static void CollectFragmentLayouts(const PrimExpr &expr,
                                     const Map<Var, PrimExpr> &let_var_to_expr,
                                     const LayoutMap &existing_layouts,
                                     PrimExpr thread_extent,
                                     Range thread_bounds,
                                     Map<Buffer, Layout> &result_map);

  static CopyInst SelectInst(const CopyNode &op, Target target,
                             const LayoutMap &layout_map,
                             arith::Analyzer *analyzer, bool buffer_oob);

  static void CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst);

  static LayoutMap InferTMemLayout(const CopyNode &op, const LayoutInferArgs &T,
                                   CopyInst copy_inst);

  static LayoutMap InferBulkLayout(const CopyNode &op, const LayoutInferArgs &T,
                                   InferLevel level, CopyInst copy_inst);

  static Stmt LowerNormal(const CopyNode &op, const LowerArgs &T,
                          arith::Analyzer *analyzer);

  static Stmt LowerCPAsync(const CopyNode &op, const LowerArgs &T,
                           arith::Analyzer *analyzer);

  static Stmt LowerLDSM(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer, CopyInst copy_inst);

  static Stmt LowerTmem(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer);

  static Stmt LowerBulk(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer, CopyInst copy_inst);

  static Stmt LowerBulk1D(const CopyNode &op, const LowerArgs &T,
                          arith::Analyzer *analyzer, CopyInst copy_inst);
};

Layout Copy::ComputeLinearLayout(const Buffer &shared_tensor) {
  Array<PrimExpr> input_size = shared_tensor->shape;
  Array<PrimExpr> forward_vars;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_vars.push_back(InputPlaceholder(i));
  }

  Array<PrimExpr> forward_index;
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorDiv(forward_vars[i], 256));
  }
  for (size_t i = 0; i < input_size.size(); i++) {
    forward_index.push_back(FloorMod(forward_vars[i], 256));
  }
  return Layout(input_size, forward_index);
}

void Copy::CollectFragmentLayouts(const PrimExpr &expr,
                                  const Map<Var, PrimExpr> &let_var_to_expr,
                                  const LayoutMap &existing_layouts,
                                  PrimExpr thread_extent, Range thread_bounds,
                                  Map<Buffer, Layout> &result_map) {
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

LayoutMap Copy::InferLayout(const CopyNode &op, const LayoutInferArgs &T,
                            InferLevel level) {
  CopyInst copy_inst =
      SelectInst(op, T.target, T.layout_map, T.analyzer, T.buffer_oob);
  CheckParallelLoopLayout(op, copy_inst);

  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    return InferTMemLayout(op, T, copy_inst);
  }
  if (copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore ||
      copy_inst == CopyInst::kBulkLoad1D ||
      copy_inst == CopyInst::kBulkStore1D) {
    return InferBulkLayout(op, T, level, copy_inst);
  }

  // For normal/cp.async/LDSM/STSM, layout inference follows the generated
  // SIMT loop. CUDA-specific explicit layout cases are handled above.
  return op.InferSIMTLayout(T, level);
}

void Copy::CheckParallelLoopLayout(const CopyNode &op, CopyInst copy_inst) {
  if (!op.annotations.count(attr::kParallelLoopLayout)) {
    return;
  }
  if (copy_inst == CopyInst::kNormal || copy_inst == CopyInst::kCPAsync) {
    return;
  }

  std::ostringstream oss;
  oss << "T.copy loop layout annotation requires SIMT copy; got "
      << CopyInstToString(copy_inst) << " for src=" << op.src->name
      << ", dst=" << op.dst->name
      << ". Remove loop_layout or change copy pattern.";
  LOG(FATAL) << oss.str();
}

LayoutMap Copy::InferTMemLayout(const CopyNode &op, const LayoutInferArgs &T,
                                CopyInst copy_inst) {
  // TODO (mzw) Add support for tcgen05.cp in CUDA tmem lowering.
  LayoutMap results;
  bool is_tmem_load = copy_inst == CopyInst::kTMemLoad;
  Buffer tmem_buf = is_tmem_load ? op.src : op.dst;
  Buffer reg_buf = is_tmem_load ? op.dst : op.src;

  if (!T.layout_map.count(reg_buf) && T.layout_map.count(tmem_buf)) {
    Layout tmem_layout = T.layout_map[tmem_buf];
    Array<IterVar> logical_coords = op.MakeIterVars();
    Array<PrimExpr> logical_coords_var = {logical_coords[0]->var,
                                          logical_coords[1]->var};
    Array<PrimExpr> phy_indices = tmem_layout->Forward(logical_coords_var);

    arith::Analyzer analyzer;
    for (const auto &iv : logical_coords) {
      analyzer.Bind(iv->var, iv->dom);
    }
    arith::ConstIntBound phy_row_bounds =
        analyzer.const_int_bound(phy_indices[0]);
    arith::ConstIntBound phy_col_bounds =
        analyzer.const_int_bound(phy_indices[1]);
    Range row_dom = Range(static_cast<int>(phy_row_bounds->min_value),
                          static_cast<int>(phy_row_bounds->max_value + 1));
    Range col_dom = Range(static_cast<int>(phy_col_bounds->min_value),
                          static_cast<int>(phy_col_bounds->max_value + 1));

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

    for (int num_useful_wgs = num_threads / WARPGROUP_SIZE; num_useful_wgs >= 1;
         --num_useful_wgs) {
      int num_useful_threads = num_useful_wgs * WARPGROUP_SIZE;
      Tcgen05Meta meta = getTcgen05MetaLd_32dp32b();
      auto [is_success, tmem_coord2frag, num_chunks_each_wg] =
          expandTcgen05Layout(
              meta, phy_col_bounds->max_value - phy_col_bounds->min_value + 1,
              num_useful_threads, row_dom, col_dom);
      (void)num_chunks_each_wg;
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

LayoutMap Copy::InferBulkLayout(const CopyNode &op, const LayoutInferArgs &T,
                                InferLevel level, CopyInst copy_inst) {
  Map<Buffer, Layout> result_map;

  bool is_tma_1d =
      copy_inst == CopyInst::kBulkLoad1D || copy_inst == CopyInst::kBulkStore1D;
  bool is_load =
      copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkLoad1D;
  Buffer shared_tensor = is_load ? op.dst : op.src;
  Array<Range> shared_range = is_load ? op.dst_range : op.src_range;

  if (is_tma_1d && shared_range.size() == 1) {
    // 1D TMA Store with single dimension can not be swizzled. 1D TMA can also
    // have multiple dimensions when the last dimension is continuous.
    return result_map;
  }

  // Fragment buffers used as TMA indices should be replicated on all threads.
  PrimExpr thread_extent = T.thread_bounds->extent;
  for (const auto &range : op.src_range) {
    CollectFragmentLayouts(range->min, T.let_var_to_expr, T.layout_map,
                           thread_extent, T.thread_bounds, result_map);
    CollectFragmentLayouts(range->extent, T.let_var_to_expr, T.layout_map,
                           thread_extent, T.thread_bounds, result_map);
  }
  for (const auto &range : op.dst_range) {
    CollectFragmentLayouts(range->min, T.let_var_to_expr, T.layout_map,
                           thread_extent, T.thread_bounds, result_map);
    CollectFragmentLayouts(range->extent, T.let_var_to_expr, T.layout_map,
                           thread_extent, T.thread_bounds, result_map);
  }

  if (is_tma_1d) {
    // 1D TMA requires contiguous shared memory. Do not infer a swizzled shared
    // layout here, otherwise final instruction selection may fall back to
    // descriptor-based multidimensional TMA.
    return result_map;
  }

  if (level == InferLevel::kFree && !T.layout_map.count(shared_tensor)) {
    // Keep MUSA BulkLoad/BulkStore shared layout linear. The MUSA descriptor
    // lowering may select swizzle-none or hardware swizzle from layout hints;
    // inferring a CUDA-style producer swizzle here can mismatch store/readback.
    result_map.Set(shared_tensor, ComputeLinearLayout(shared_tensor));
  }

  return result_map;
}

CopyInst Copy::SelectInst(const CopyNode &op, Target target,
                          const LayoutMap &layout_map,
                          arith::Analyzer *analyzer, bool buffer_oob) {
  CopyAnalysisContext ctx;
  ctx.target = target;
  ctx.layout_map = &layout_map;
  ctx.analyzer = analyzer;
  ctx.buffer_oob = buffer_oob;
  ctx.emit_diagnostics = true;
  auto result = SelectCopyInstForLowering(op, ctx);
  ICHECK(result.supported) << result.reason;
  return result.inst;
}

Stmt Copy::Lower(const CopyNode &op, const LowerArgs &T,
                 arith::Analyzer *analyzer) {
  auto copy_inst =
      SelectInst(op, T.target, T.layout_map, analyzer, /*buffer_oob=*/false);
  if (copy_inst == CopyInst::kTMemLoad || copy_inst == CopyInst::kTMemStore) {
    auto tmem_copy = LowerTmem(op, T, analyzer);
    ICHECK(tmem_copy.defined()) << "Failed to lower tensor memory copy";
    return tmem_copy;
  } else if (copy_inst == CopyInst::kBulkLoad1D ||
             copy_inst == CopyInst::kBulkStore1D) {
    auto bulk_copy = LowerBulk1D(op, T, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load 1d";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kBulkLoad ||
             copy_inst == CopyInst::kBulkStore) {
    auto bulk_copy = LowerBulk(op, T, analyzer, copy_inst);
    ICHECK(bulk_copy.defined()) << "Failed to lower bulk load/store";
    return bulk_copy;
  } else if (copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM) {
    auto ldsm_copy = LowerLDSM(op, T, analyzer, copy_inst);
    ICHECK(ldsm_copy.defined()) << "Failed to lower ptx matrix copy";
    return ldsm_copy;
  } else if (copy_inst == CopyInst::kCPAsync) {
    auto cp_async_copy = LowerCPAsync(op, T, analyzer);
    ICHECK(cp_async_copy.defined()) << "Failed to lower cp.async copy";
    return cp_async_copy;
  } else if (copy_inst == CopyInst::kNormal) {
    return LowerNormal(op, T, analyzer);
  } else {
    LOG(FATAL) << "Unsupported copy inst " << static_cast<int>(copy_inst);
  }
}

Stmt Copy::LowerCPAsync(const CopyNode &op, const LowerArgs &T,
                        arith::Analyzer *analyzer) {
  using namespace tvm::transform;

  PassContext pass_ctx = PassContext::Current();
  bool enable_async_copy =
      pass_ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
  bool no_implicit_commit_wait = GetNoImplicitAsyncCommitWait(op);
  bool explicit_async_semantics = no_implicit_commit_wait || GetIsAsyncCopy(op);
  if (!enable_async_copy && !explicit_async_semantics) {
    return LowerNormal(op, T, analyzer);
  }

  auto simt_loop = op.MakeSIMTLoop(analyzer);
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

  auto inject_result =
      InjectPTXAsyncCopy(lowered_loop, /*enable_auto_async_copy=*/true,
                         /*async_without_async_commit_wait=*/
                         no_implicit_commit_wait || GetIsAsyncCopy(op));
  Stmt cp_async_loop = inject_result.stmt;
  if (!inject_result.injected_ptx_async_copy) {
    DLOG(WARNING) << "cp.async rewrite miss for copy src=" << op.src->name
                  << " (scope=" << op.src.scope() << ", dtype=" << op.src->dtype
                  << "), dst=" << op.dst->name << " (scope=" << op.dst.scope()
                  << ", dtype=" << op.dst->dtype
                  << "), no_implicit_async_commit_wait="
                  << no_implicit_commit_wait
                  << ", is_async_copy=" << GetIsAsyncCopy(op);
    if (no_implicit_commit_wait) {
      DLOG(WARNING)
          << "Pipeline-managed async copy fallback to normal copy because "
             "cp.async rewrite found no eligible global->shared store.";
      return lowered_loop;
    }
    if (explicit_async_semantics) {
      LOG(FATAL) << "Explicit async copy semantics require cp.async lowering, "
                    "but no eligible global->shared store was rewritten.";
    }
    DLOG(WARNING) << "Fallback to normal copy because cp.async rewrite found "
                     "no eligible global->shared store.";
    return LowerNormal(op, T, analyzer);
  }
  if (no_implicit_commit_wait) {
    return cp_async_loop;
  }
  if (GetIsAsyncCopy(op)) {
    Stmt commit_group =
        Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
    return SeqStmt({cp_async_loop, commit_group});
  }
  return cp_async_loop;
}

Stmt Copy::LowerNormal(const CopyNode &op, const LowerArgs &T,
                       arith::Analyzer *analyzer) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
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
    PrimExpr src_robust_desc = GetSourceRobustDesc(node);
    if (src_robust_desc.defined()) {
      ICHECK(TargetIsMusa(T.target))
          << "src_robust_desc is only supported when targeting MUSA.";
      ICHECK(node.src.scope() == "global")
          << "src_robust_desc requires a global-memory source, but got `"
          << node.src.scope() << "`.";
      body = AttrStmt(node.src->data, attr::kSourceRobustDesc, src_robust_desc,
                      body);
    }
    if (GetForceAsyncCopy(node)) {
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
    return lower_single_copy(op);
  }

  auto split_op = tvm::ffi::make_object<CopyNode>(op);
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

Stmt Copy::LowerLDSM(const CopyNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer, CopyInst copy_inst) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;

  ICHECK(copy_inst == CopyInst::kLDSM || copy_inst == CopyInst::kSTSM)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_ldmatrix = copy_inst == CopyInst::kLDSM;

  Array<IterVar> loop_vars = op.MakeIterVars();
  if (loop_vars.size() < 2) {
    return LowerNormal(op, T, analyzer);
  }
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = op.MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = op.MakePredicate(analyzer, loop_vars, dst->shape, 1);
  if (src_predicate.defined() || dst_predicate.defined()) {
    return LowerNormal(op, T, analyzer);
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
    return LowerNormal(op, T, analyzer);
  }

  Array<PrimExpr> local_indices =
      op.MakeIndices(loop_vars, is_ldmatrix ? 1 : 0);
  Fragment local_layout = Downcast<Fragment>(T.layout_map[local_tensor]);
  Array<PrimExpr> local_indices_transformed =
      local_layout->Forward(local_indices);
  local_tensor = T.buffer_remap[local_tensor];
  if (local_layout->OutputDim() != 1) {
    return LowerNormal(op, T, analyzer);
  }

  Array<PrimExpr> shared_indices =
      op.MakeIndices(loop_vars, is_ldmatrix ? 0 : 1);
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
    return LowerNormal(op, T, analyzer);
  }
  if (shared_tensor->dtype.bytes() != 2) {
    return LowerNormal(op, T, analyzer);
  }
  PrimExpr flattened_indice = shared_tensor.OffsetOf(shared_indices).back();
  if (!IndicesCanVectorize(flattened_indice, loop_vars.back()->var,
                           loop_vars.back()->dom->extent, 8, analyzer)) {
    return LowerNormal(op, T, analyzer);
  }

  for (size_t i = 0; i < dst_range.size(); i++) {
    if (!is_zero(dst_range[i]->min) ||
        !analyzer->CanProveEqual(dst_range[i]->extent, dst->shape[i]))
      return LowerNormal(op, T, analyzer);
  }

  PrimExpr extent = local_tensor->shape[0];
  int num = 1;
  if (analyzer->CanProveEqual(FloorMod(extent, 8), 0))
    num = 4;
  else if (analyzer->CanProveEqual(FloorMod(extent, 4), 0))
    num = 2;

  Array<PrimExpr> args;
  const Op &copy_op = is_ldmatrix ? tl::ptx_ldmatrix() : tl::ptx_stmatrix();
  args.push_back(static_cast<int>(is_transposed));
  args.push_back(num);

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
  shared_coords.pop_back();
  PrimExpr shared_addr =
      Call(DataType::Handle(), tl::access_ptr(),
           {BufferLoad(shared_tensor, shared_coords), PrimExpr(2 * num),
            make_const(DataType::Int(32), is_ldmatrix ? 1 : 2)});
  args.push_back(shared_addr);

  if (is_ldmatrix) {
    if (local_tensor->dtype != shared_tensor->dtype) {
      return LowerNormal(op, T, analyzer);
    }
    PrimExpr local_addr =
        Call(DataType::Handle(), tl::access_ptr(),
             {BufferLoad(local_tensor, {local_iter * 2 * num}),
              PrimExpr(2 * num), make_const(DataType::Int(32), 2)});
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

  auto body = Evaluate(Call(DataType::Handle(), copy_op, args));
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

Stmt Copy::LowerTmem(const CopyNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
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
  Array<IterVar> loop_vars = op.MakeIterVars();
  ICHECK(loop_vars.size() == 2) << "Only support 2D tensor memory copy, got "
                                << loop_vars.size() << " dimensions";
  for (const auto &iv : loop_vars)
    analyzer->Bind(iv->var, iv->dom);
  PrimExpr src_predicate = op.MakePredicate(analyzer, loop_vars, src->shape, 0);
  PrimExpr dst_predicate = op.MakePredicate(analyzer, loop_vars, dst->shape, 1);
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
  Array<PrimExpr> logical_indices = op.MakeIndices(loop_vars, tmem_side);
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

Stmt Copy::LowerBulk(const CopyNode &op, const LowerArgs &T,
                     arith::Analyzer *analyzer, CopyInst copy_inst) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
  const Map<String, ObjectRef> &annotations = op.annotations;
  ICHECK(copy_inst == CopyInst::kBulkLoad || copy_inst == CopyInst::kBulkStore)
      << "Invalid copy inst " << static_cast<int>(copy_inst);
  bool is_load = copy_inst == CopyInst::kBulkLoad;
  Buffer global_tensor = is_load ? src : dst;
  Buffer shared_tensor = is_load ? dst : src;
  Buffer shared_tensor_unmapped = shared_tensor;
  PrimExpr tma_barrier = GetTmaBarrier(op);
  Array<Range> global_range = is_load ? src_range : dst_range;
  Array<Range> shared_range = is_load ? dst_range : src_range;
  PrimExpr total_copy_bytes = shared_tensor->dtype.bytes();
  for (const auto &range : shared_range) {
    total_copy_bytes *= range->extent;
  }
  total_copy_bytes = analyzer->Simplify(total_copy_bytes);
  auto lower_barriered_normal_fallback = [&]() -> Stmt {
    bool needs_external_arrive = TargetIsMusa(T.target) && is_load &&
                                 GetIsTmaCopy(op) &&
                                 annotations.Get("barrier").has_value();
    if (!needs_external_arrive) {
      return LowerNormal(op, T, analyzer);
    }

    auto fallback_op = tvm::ffi::make_object<CopyNode>(op);
    auto fallback_annotations = fallback_op->annotations;
    // The fallback must be complete before releasing the WS mbarrier. Avoid
    // re-injecting a bare force_async_copy here because MUSA lacks a
    // cp.async-mbarrier coupling instruction.
    fallback_annotations.erase("force_async_copy");
    fallback_annotations.erase("is_async_copy");
    fallback_annotations.erase("force_cp_async");
    fallback_annotations.erase(attr::kAsyncCopyNoImplicitCommitWait);
    fallback_op->annotations = fallback_annotations;

    Stmt copy = LowerNormal(*fallback_op, T, analyzer);
    Stmt arrive =
        Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier(),
                      {GetTmaBarrier(op)}));
    return SeqStmt({copy, arrive});
  };
  // TMA bulk copy cannot support a non-swizzled global layout, will be fallback
  // to normal copy
  if (T.layout_map.count(global_tensor)) {
    LOG(WARNING) << "TMA bulk copy cannot support a non-swizzled global "
                    "layout, fallback to normal copy.";
    return lower_barriered_normal_fallback();
  }

  // linear layout must be computed before remapping
  auto linear_layout = ComputeLinearLayout(shared_tensor);

  Array<PrimExpr> shared_indices;
  for (auto r : shared_range)
    shared_indices.push_back(r->min);
  std::vector<PrimExpr> shared_strides;
  PrimExpr shared_stride = 1;
  for (size_t i = 0; i < shared_tensor->shape.size(); i++) {
    auto s = shared_tensor->shape[shared_tensor->shape.size() - i - 1];
    shared_strides.insert(shared_strides.begin(), shared_stride);
    shared_stride *= s;
  }

  Array<PrimExpr> global_indices;
  for (auto r : global_range) {
    global_indices.push_back(r->min);
  }
  std::vector<PrimExpr> global_strides;
  PrimExpr global_stride = 1;
  for (size_t i = 0; i < global_tensor->shape.size(); i++) {
    auto s = global_tensor->shape[global_tensor->shape.size() - i - 1];
    global_strides.insert(global_strides.begin(), global_stride);
    global_stride *= s;
  }

  ICHECK(shared_strides.size() == shared_indices.size())
      << "shared_strides.size() != shared_indices.size()"
      << shared_strides.size() << " " << shared_indices.size();
  PrimExpr shared_offset = 0;
  for (size_t i = 0; i < shared_indices.size(); i++) {
    shared_offset += shared_indices[i] * shared_strides[i];
  }
  PrimExpr global_offset = 0;
  for (size_t i = 0; i < global_indices.size(); i++) {
    global_offset += global_indices[i] * global_strides[i];
  }

  TMADesc desc;
  // Verify copy rank
  desc.rank = global_tensor->shape.size();
  ICHECK(desc.rank >= 1 && desc.rank <= 5) << desc.rank;

  // Verify datatype
  ICHECK(global_tensor->dtype == shared_tensor->dtype)
      << "Copy between buffer " << global_tensor->name << " and "
      << shared_tensor->name << " with different data type "
      << global_tensor->dtype << " and " << shared_tensor->dtype;

  desc.data_type = to_MUtensorDescriptorDataType(global_tensor->dtype);

  // Global Tensor Shape and Stride
  desc.global_addr = global_tensor->data;
  desc.global_shape = ReverseArray(global_tensor->shape);
  Array<PrimExpr> global_coords =
      ReverseArray(global_range.Map([](Range r) { return r->min; }));
  if (!global_tensor->strides.empty()) {
    desc.global_stride = ReverseArray(global_tensor->strides);
  } else {
    // Create stride from shape
    PrimExpr stride = 1;
    desc.global_stride.reserve(desc.rank);
    for (size_t i = 0; i < desc.rank; i++) {
      desc.global_stride.push_back(stride);
      stride *= desc.global_shape[i];
    }
  }
  // The first stride element should be 1
  ICHECK(is_one(desc.global_stride[0])) << desc.global_stride;
  // Make global stride in bytes
  auto stride_dtype =
      TargetIsMusa(T.target) ? DataType::UInt(64) : DataType::Int(64);
  desc.global_stride = desc.global_stride.Map([&](PrimExpr e) {
    return cast(stride_dtype, e) * global_tensor->dtype.bytes();
  });
  for (size_t i{1}; i < desc.global_stride.size(); i++) {
    auto stride = desc.global_stride[i].as<IntImmNode>();
    if (stride != nullptr) {
      // otherwise, the stride is symbolic, we need to check in future with
      // assumptions
      if (stride->value % 16 != 0 || stride->value >= (1ULL << 40)) {
        LOG(WARNING) << "TMA bulk copy cannot support a global stride of "
                     << desc.global_stride[i] << ", fallback to normal copy.";
        return lower_barriered_normal_fallback();
      }
    }
  }

  // Smem Box
  // check smem range and global range is legal
  auto s_range_idx = 0;
  for (size_t i = 0; i < global_range.size(); i++) {
    auto g_range = global_range[i];
    if (is_one(g_range->extent)) {
      continue;
    }
    // skip one range if it is 1
    // in case of global range is [128, 64], while shared range is [1, 128, 64]
    // A_shared[0, :, :].
    while (s_range_idx < shared_range.size() &&
           is_one(shared_range[s_range_idx]->extent)) {
      s_range_idx++;
    }
    if (s_range_idx >= shared_range.size()) {
      LOG(FATAL) << "TMA bulk copy cannot support a global range of "
                 << global_range << ", shared_range " << shared_range;
    }
    auto s_range = shared_range[s_range_idx];
    s_range_idx++;

    auto g_extent = analyzer->Simplify(g_range->extent);
    auto s_extent = analyzer->Simplify(s_range->extent);
    ICHECK(analyzer->CanProveEqual(g_extent, s_extent))
        << global_tensor->name << "[" << i << "] is illegal, "
        << global_tensor->name << "[" << i << "] = " << g_range->extent << ", "
        << shared_tensor->name << "[" << s_range_idx
        << "] = " << s_range->extent;
  }
  // TODO(lei): find a much smarter way to deduce smem box dim
  // instead of using global_range
  desc.smem_box =
      ReverseArray(global_range.Map([](Range r) { return r->extent; }));

  desc.smem_stride = Array<PrimExpr>(desc.rank, PrimExpr(1));
  // L2 & OOB
  desc.l2_promotion = static_cast<int>(CU_TENSOR_MAP_L2_PROMOTION_L2_128B);
  desc.oob_fill = static_cast<int>(CU_TENSOR_MAP_FLOAT_OOB_FILL_NONE);

  // Detect smem layout
  // Shared memory swizzling is crucial for TMA performance
  // It determines how data is arranged in shared memory banks to minimize bank
  // conflicts Different swizzle patterns (32B, 64B, 128B) offer different
  // trade-offs between access efficiency and memory usage
  desc.interleave = TargetIsMusa(T.target)
                        ? static_cast<int>(MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE)
                        : static_cast<int>(CU_TENSOR_MAP_INTERLEAVE_NONE);
  desc.swizzle = TargetIsMusa(T.target)
                     ? static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_NONE)
                     : static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  Layout shared_layout;
  auto active_layout = GetActiveLayout(T, shared_tensor);
  if (active_layout.has_value()) {
    shared_layout = active_layout.value();
  }
  if (T.buffer_remap.count(shared_tensor)) {
    shared_tensor = T.buffer_remap.at(shared_tensor);
  }
  bool musa_force_swizzle_none = false;
  if (!shared_layout.defined()) {
    desc.swizzle = TargetIsMusa(T.target)
                       ? static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_NONE)
                       : static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
  } else if (StructuralEqual()(shared_layout, linear_layout)) {
    desc.swizzle = TargetIsMusa(T.target)
                       ? static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_NONE)
                       : static_cast<int>(CU_TENSOR_MAP_SWIZZLE_NONE);
    musa_force_swizzle_none = TargetIsMusa(T.target);
  } else if (TargetIsMusa(T.target) && IsMatrixLinearLayout(shared_layout)) {
    desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_NONE);
    musa_force_swizzle_none = true;
  } else if (!TargetIsMusa(T.target)) {
    if (shared_layout->InputDim() < 2) {
      LOG(WARNING) << "TMA bulk copy cannot support shared layout with input "
                   << "dimension " << shared_layout->InputDim()
                   << ", fallback to normal copy.";
      return lower_barriered_normal_fallback();
    }
    const int ndim = static_cast<int>(shared_layout->InputDim());
    auto stride = as_const_int(shared_layout->InputShape()[ndim - 2]);
    auto continuous = as_const_int(shared_layout->InputShape()[ndim - 1]);
    ICHECK(stride != nullptr && continuous != nullptr);
    // We also need to check if the shape satisfies the following doc:
    // https://docs.nvidia.com/cuda/cuda-driver-api/group__CUDA__TENSOR__MEMORY.html#group__CUDA__TENSOR__MEMORY_1ga7c7d2aaac9e49294304e755e6f341d7
    SwizzleMode swizzle_mode =
        DetectSwizzleMode(shared_layout, shared_tensor_unmapped);
    if (swizzle_mode == SwizzleMode::kQuarter) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B);
    } else if (swizzle_mode == SwizzleMode::kHalf) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B);
    } else if (swizzle_mode == SwizzleMode::kFull) {
      desc.swizzle = static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B);
    } else if (StructuralEqual()(
                   shared_layout,
                   makeGemmABLayoutPadded(*stride, *continuous,
                                          shared_tensor->dtype.bits()))) {
      LOG(WARNING) << "Bulk copy cannot support a padded layout for src: "
                   << src->name << ", dst: " << dst->name
                   << ", fallback to normal copy";
      return lower_barriered_normal_fallback();
    } else {
      LOG(WARNING) << "Came across unsupported swizzle layout for src: "
                   << src->name << ", dst: " << dst->name
                   << ", fallback to normal copy";
      return lower_barriered_normal_fallback();
    }
  }

  auto get_k_major = [&](const Buffer &buf) -> int {
    auto k_major_opt = GetLayoutKMajor(T, buf);
    if (k_major_opt.has_value()) {
      return k_major_opt.value() ? 1 : 0;
    }
    return -1;
  };
  // set swizzle granularity by elem_bytes and k_major
  if (TargetIsMusa(T.target) && !musa_force_swizzle_none) {
    int elem_bytes = shared_tensor->dtype.bytes();
    int is_k_major = get_k_major(shared_tensor);
    if (is_k_major < 0 && !shared_tensor.same_as(shared_tensor_unmapped)) {
      is_k_major = get_k_major(shared_tensor_unmapped);
    }
    if (elem_bytes == 1) {
      desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B16);
    } else if (elem_bytes == 2) {
      if (is_k_major == 1) {
        desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B16);
      } else if (is_k_major == 0) {
        desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B32);
      } else {
        LOG(INFO) << src->name << " use elem_bytes " << elem_bytes
                  << ", unknown matrix layout for swizzle";
      }
    } else if (elem_bytes == 4) {
      if (is_k_major == 1) {
        desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B16);
      } else if (is_k_major == 0) {
        desc.swizzle = static_cast<int>(MU_SMEM_SWIZZLE_GRANULARITY_B64);
      } else {
        LOG(INFO) << src->name << " use elem_bytes " << elem_bytes
                  << ", unknown matrix layout for swizzle";
      }
    } else {
      LOG(WARNING) << src->name << " use elem_bytes " << elem_bytes
                   << ", swizzle not set";
    }
  }

  int split_count = 1;
  PrimExpr split_stride = 0;
  PrimExpr split_size_expr = 0;
  if (TargetIsMusa(T.target) && is_load) {
    auto sqmma_opt = GetLayoutSQMMA(T, shared_tensor);
    auto k_major_opt = GetLayoutKMajor(T, shared_tensor);
    if (!sqmma_opt.has_value() &&
        !shared_tensor.same_as(shared_tensor_unmapped)) {
      sqmma_opt = GetLayoutSQMMA(T, shared_tensor_unmapped);
    }
    if (!k_major_opt.has_value() &&
        !shared_tensor.same_as(shared_tensor_unmapped)) {
      k_major_opt = GetLayoutKMajor(T, shared_tensor_unmapped);
    }
    bool is_musa_sqmma_tma_load =
        src.scope() == "global" &&
        (dst.scope() == "shared" || dst.scope() == "shared.dyn") &&
        k_major_opt.has_value() && sqmma_opt.value_or(false);
    if (is_musa_sqmma_tma_load) {
      auto n_dim = as_const_int(desc.smem_box[0]);
      auto make_split_stride = [&](PrimExpr split_size) {
        PrimExpr stride = split_size;
        for (size_t i = 1; i < desc.smem_box.size(); ++i) {
          stride *= desc.smem_box[i];
        }
        return analyzer->Simplify(stride);
      };
      bool is_k_major = k_major_opt.value();
      if (!is_k_major) {
        int sqmma_inst_split =
            GetLayoutSqmmaInstSplit(T, shared_tensor).value_or(-1);
        if (sqmma_inst_split < 0 &&
            !shared_tensor.same_as(shared_tensor_unmapped)) {
          sqmma_inst_split =
              GetLayoutSqmmaInstSplit(T, shared_tensor_unmapped).value_or(-1);
        }
        if (sqmma_inst_split > 0 && n_dim != nullptr &&
            (*n_dim) > sqmma_inst_split && ((*n_dim) % sqmma_inst_split) == 0) {
          split_count = (*n_dim) / sqmma_inst_split;
          split_size_expr = IntImm(DataType::Int(32), sqmma_inst_split);
          split_stride = make_split_stride(split_size_expr);
          desc.smem_box.Set(0, split_size_expr);
        }
      } else {
        int elem_bytes = shared_tensor->dtype.bytes();
        int max_inner_elems = elem_bytes > 0 ? 256 / elem_bytes : 0;
        if (max_inner_elems > 0 && n_dim != nullptr &&
            (*n_dim) * elem_bytes > 256 && ((*n_dim) % max_inner_elems) == 0) {
          split_count = (*n_dim) / max_inner_elems;
          split_size_expr = IntImm(DataType::Int(32), max_inner_elems);
          split_stride = make_split_stride(split_size_expr);
          desc.smem_box.Set(0, split_size_expr);
        }
      }
    }
  }

  auto inner_box_dim = as_const_int(desc.smem_box[0]);
  if (inner_box_dim == nullptr) {
    LOG(WARNING) << "inner_box_dim " << desc.smem_box[0]
                 << " can only be a constant integer for TMA bulk copy, "
                    "fallback to normal copy";
    return lower_barriered_normal_fallback();
  }
  int instruction_dim = *inner_box_dim;
  if (!TargetIsMusa(T.target)) {
    if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B)) {
      instruction_dim = 64 / src->dtype.bytes();
    } else if (desc.swizzle == static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B)) {
      instruction_dim = 128 / src->dtype.bytes();
    }
  }
  if (!TargetIsMusa(T.target) && instruction_dim > 256) {
    // smem_box dim must be in [0, 256]
    // if is 512, we need to split the copy into two parts
    ICHECK((*inner_box_dim) % 256 == 0)
        << "inner_box_dim: " << *inner_box_dim << " is not divisible by 256";
    instruction_dim = 256;
  }
  ICHECK((*inner_box_dim) % instruction_dim == 0)
      << "inner_box_dim: " << *inner_box_dim
      << " is not divisible by instruction_dim: " << instruction_dim;
  desc.smem_box.Set(0, PrimExpr(instruction_dim));

  int inner_box_dim_ = instruction_dim * shared_tensor->dtype.bytes();

  // Check inner_box_dim_ for each swizzle type in a cleaner way
  struct SwizzleCheck {
    int swizzle;
    int max_dim;
  };
  static const std::vector<SwizzleCheck> swizzle_checks = {
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_32B), 32},
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_64B), 64},
      {static_cast<int>(CU_TENSOR_MAP_SWIZZLE_128B), 128},
  };
  if (!TargetIsMusa(T.target)) {
    for (const auto &check : swizzle_checks) {
      if (desc.swizzle == check.swizzle && inner_box_dim_ > check.max_dim) {
        LOG(WARNING) << "TMA bulk copy cannot support a swizzled global layout "
                        "with inner_box_dim_ > "
                     << check.max_dim << ", will be fallback to normal copy";
        return lower_barriered_normal_fallback();
      }
    }
  }

  if (TargetIsMusa(T.target)) {
    desc.global_shape = desc.global_shape.Map(
        [](PrimExpr e) { return cast(DataType::UInt(64), e); });
  }

  Call create_descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());

  // For TMA loads, allocate mbarrier(s) for synchronous semantics.
  // Determine the mbarrier handle for TMA loads.
  // T.tma_copy(): requires user-provided barrier
  // T.copy(): allocates internal mbarrier via AllocMBarrier
  int barrier_base_id = -1;
  PrimExpr mbar_handle;
  bool is_cluster_barrier = false;
  if (is_load) {
    if (auto user_barrier = annotations.Get("barrier")) {
      mbar_handle = Downcast<PrimExpr>(user_barrier.value());
      barrier_base_id = 0;
      if (auto bl = mbar_handle.as<BufferLoadNode>()) {
        is_cluster_barrier = bl->buffer.scope() == "shared.cluster_barrier";
      }
    } else if (GetIsTmaCopy(op)) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx]).";
    } else if (T.AllocMBarrier) {
      // Internal mbarrier (T.copy()): allocate a single barrier slot.
      // Pipeline buffer versioning expands it per stage when needed.
      barrier_base_id = T.AllocMBarrier(1);
      PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
      mbar_handle = BufferLoad(T.mbarrier_buffer->value(), {mbar_idx});
    }
  }
  PrimExpr load_barrier_arg = barrier_base_id >= 0 ? mbar_handle : tma_barrier;
  auto tma_op = is_load ? tma_load() : tma_store();

  Stmt tma_copy;
  PrimExpr total_elements = 1;
  for (auto e : desc.smem_box)
    total_elements *= e;

  if (TargetIsMusa(T.target) && split_count > 1) {
    Array<PrimExpr> base_args;
    base_args.reserve(desc.rank + 4);
    base_args.push_back(create_descriptor);
    if (is_load)
      base_args.push_back(load_barrier_arg);

    auto emit_tma_copy = [&](PrimExpr shared_addr,
                             const Array<PrimExpr> &coords) -> Stmt {
      Array<PrimExpr> call_args = base_args;
      call_args.push_back(shared_addr);
      for (auto coord : coords)
        call_args.push_back(coord);
      int need_reduce = 0;
      if (!is_load)
        call_args.push_back(need_reduce);
      call_args.push_back(GetEvictionPolicy(op));
      call_args.push_back(GetInnerCachePolicy(op));
      call_args.push_back(GetOuterCachePolicy(op));
      return Evaluate(Call(DataType::Handle(), tma_op, call_args));
    };

    auto make_inner_copy = [&](PrimExpr base_shared_offset,
                               const Array<PrimExpr> &base_coords) -> Stmt {
      if ((*inner_box_dim) != instruction_dim) {
        Var loop_var("i");
        int loop_extent = (*inner_box_dim) / instruction_dim;
        Array<PrimExpr> coords = base_coords;
        coords.Set(0, coords[0] + instruction_dim * loop_var);
        PrimExpr shared_addr = shared_tensor.access_ptr(
            is_load ? 2 : 1, DataType::Handle(), 1,
            base_shared_offset + total_elements * loop_var, total_elements);
        return For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                   emit_tma_copy(shared_addr, coords));
      } else {
        PrimExpr shared_addr =
            shared_tensor.access_ptr(is_load ? 2 : 1, DataType::Handle(), 1,
                                     base_shared_offset, total_elements);
        return emit_tma_copy(shared_addr, base_coords);
      }
    };

    Var split_var("s");
    Array<PrimExpr> coords = global_coords;
    coords.Set(0, coords[0] + split_size_expr * split_var);
    PrimExpr base_shared_offset = shared_offset + split_stride * split_var;
    tma_copy = For(split_var, 0, split_count, ForKind::kUnrolled,
                   make_inner_copy(base_shared_offset, coords));
  } else {
    Array<PrimExpr> args;
    args.reserve(desc.rank + 4);
    args.push_back(create_descriptor);
    if (is_load)
      args.push_back(load_barrier_arg);
    if ((*inner_box_dim) != instruction_dim) {
      Var loop_var("i");
      int loop_extent = (*inner_box_dim) / instruction_dim;
      PrimExpr shared_addr = shared_tensor.access_ptr(
          is_load ? 2 : 1, DataType::Handle(), 1,
          shared_offset + total_elements * loop_var, total_elements);
      args.push_back(shared_addr);
      global_coords.Set(0, global_coords[0] + instruction_dim * loop_var);
      for (auto coord : global_coords)
        args.push_back(coord);
      int need_reduce = 0;
      if (!is_load)
        args.push_back(need_reduce);
      args.push_back(GetEvictionPolicy(op));
      args.push_back(GetInnerCachePolicy(op));
      args.push_back(GetOuterCachePolicy(op));
      tma_copy = For(loop_var, 0, loop_extent, ForKind::kUnrolled,
                     Evaluate(Call(DataType::Handle(), tma_op, args)));
    } else {
      PrimExpr shared_addr =
          shared_tensor.access_ptr(is_load ? 2 : 1, DataType::Handle(), 1,
                                   shared_offset, total_elements);
      args.push_back(shared_addr);
      for (auto coord : global_coords)
        args.push_back(coord);
      int need_reduce = 0;
      if (!is_load)
        args.push_back(need_reduce);
      args.push_back(GetEvictionPolicy(op));
      args.push_back(GetInnerCachePolicy(op));
      args.push_back(GetOuterCachePolicy(op));
      tma_copy = Evaluate(Call(DataType::Handle(), tma_op, args));
    }
  }

  // Bulk TMA stores participate in the async store group mechanism, so they
  // must be committed. T.copy() keeps synchronous semantics by waiting here;
  // T.tma_copy() leaves the wait to the user for explicit batching.
  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    if (!GetIsTmaCopy(op)) {
      seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                  {IntImm(DataType::Int(32), 0)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  // For TMA loads with inline mbarrier: emit expect_tx before tma_load
  // (inside thread-gated block), and wait_parity after (all threads).
  // The producer is annotated with the shared buffer so PipelinePlanning can
  // detect it as a copy stage and schedule it at pipeline stage 0.
  if (is_load && barrier_base_id >= 0) {
    // Compute total bytes for all TMA sub-copies in this operation
    PrimExpr total_bytes;
    if ((*inner_box_dim) != instruction_dim) {
      int loop_extent = (*inner_box_dim) / instruction_dim;
      total_bytes = total_elements * loop_extent * shared_tensor->dtype.bytes();
    } else {
      total_bytes = total_elements * shared_tensor->dtype.bytes();
    }
    if (TargetIsMusa(T.target) && split_count > 1) {
      total_bytes *= split_count;
    }
    total_bytes = analyzer->Simplify(total_bytes);

    Stmt barrier_before_tma_stmt;
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (GetIsTmaCopy(op)) {
      // T.tma_copy(): only expect_tx (no arrive). User must call
      // T.barrier_arrive() explicitly. This allows multiple tma_copy operations
      // to share a single arrive.
      if (is_cluster_barrier) {
        // For cluster barriers in 2CTA mode: all CTAs' TMA arrivals go to
        // CTA 0's barrier (via tma_load_2sm peer-bit clearing). So expect_tx
        // must account for ALL CTAs' bytes and only execute on CTA 0.
        PrimExpr cluster_total_bytes =
            total_bytes * IntImm(DataType::Int(32), T.cluster_size);
        Stmt expect_stmt =
            Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                          {mbar_handle, cluster_total_bytes}));
        PrimExpr rank = Call(DataType::Int(32), block_rank_in_cluster(), {});
        barrier_before_tma_stmt =
            IfThenElse(EQ(rank, IntImm(DataType::Int(32), 0)), expect_stmt);
      } else {
        barrier_before_tma_stmt =
            Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                          {mbar_handle, total_bytes}));
      }
      // When emit_arrive is set (by InjectSoftwarePipeline for pipeline-level
      // barrier management), also emit arrive inside the thread-0 guard.
      if (auto emit_arrive_val = annotations.Get("emit_arrive")) {
        if (Downcast<IntImm>(emit_arrive_val.value())->value != 0) {
          barrier_after_tma_stmt =
              Evaluate(Call(DataType::Handle(), builtin::ptx_arrive_barrier(),
                            {mbar_handle}));
        }
      }
    } else {
      // T.copy() with TMA: keep expect_tx and arrive as separate control ops.
      // This lets downstream WS/barrier passes reason about the arrival
      // domain explicitly when TMA shares a stage barrier with cp.async.
      barrier_before_tma_stmt =
          Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
      barrier_after_tma_stmt = Evaluate(Call(
          DataType::Handle(), builtin::ptx_arrive_barrier(), {mbar_handle}));
    }

    Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy};
    if (barrier_after_tma_stmt.defined()) {
      producer_seq.push_back(barrier_after_tma_stmt.value());
    }

    // Thread-gated block: expect_tx + tma_load (+ optional arrive)
    Stmt producer = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                               SeqStmt(producer_seq));

    // tma_copy (from T.tma_copy()) is fire-and-forget: only emit the
    // producer (expect_tx + tma_load). The user manages synchronization
    // (arrive + wait) explicitly.
    if (GetIsTmaCopy(op)) {
      return producer;
    }

    // For T.copy() with TMA: emit producer + wait pair so the pipeline/WS
    // passes can split them into different stages.
    Stmt wait_stmt =
        Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                      {mbar_handle, GetCopyMbarPhaseExpr(annotations, T)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy =
      IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent), tma_copy);

  return tma_copy;
}

Stmt Copy::LowerBulk1D(const CopyNode &op, const LowerArgs &T,
                       arith::Analyzer *analyzer, CopyInst copy_inst) {
  const Buffer &src = op.src;
  const Buffer &dst = op.dst;
  const Array<Range> &src_range = op.src_range;
  const Array<Range> &dst_range = op.dst_range;
  const Map<String, ObjectRef> &annotations = op.annotations;
  ICHECK(copy_inst == CopyInst::kBulkLoad1D ||
         copy_inst == CopyInst::kBulkStore1D);

  // Add 1D TMA copy when the global and shared memory is contiguous
  // Check if shared_tensor->name is present in T.buffer_var_gemm
  // (Array<PrimExpr>) to avoid use 1D TMA copy for swizzled layout
  bool is_load = copy_inst == CopyInst::kBulkLoad1D;
  auto shared_range = is_load ? dst_range : src_range;
  auto global_range = is_load ? src_range : dst_range;
  auto shared_tensor = is_load ? dst : src;
  auto global_tensor = is_load ? src : dst;
  PrimExpr tma_barrier = GetTmaBarrier(op);

  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); i++) {
    shared_elements *= shared_range[i]->extent;
  }

  std::vector<PrimExpr> shared_strides;
  PrimExpr shared_stride = 1;
  for (size_t i = 0; i < shared_tensor->shape.size(); i++) {
    auto s = shared_tensor->shape[shared_tensor->shape.size() - i - 1];
    shared_strides.insert(shared_strides.begin(), shared_stride);
    shared_stride *= s;
  }

  Array<PrimExpr> shared_indices;
  for (auto r : shared_range)
    shared_indices.push_back(r->min);

  Array<PrimExpr> global_indices;
  for (auto r : global_range) {
    global_indices.push_back(r->min);
  }
  std::vector<PrimExpr> global_strides;
  PrimExpr global_stride = 1;
  for (size_t i = 0; i < global_tensor->shape.size(); i++) {
    auto s = global_tensor->shape[global_tensor->shape.size() - i - 1];
    global_strides.insert(global_strides.begin(), global_stride);
    global_stride *= s;
  }

  PrimExpr global_offset = 0;
  for (size_t i = 0; i < global_indices.size(); i++) {
    global_offset += global_indices[i] * global_strides[i];
  }

  PrimExpr shared_offset = 0;
  for (size_t i = 0; i < shared_indices.size(); i++) {
    shared_offset += shared_indices[i] * shared_strides[i];
  }

  PrimExpr elements = analyzer->Simplify(shared_elements);
  PrimExpr shared_addr = shared_tensor.access_ptr(
      is_load ? 2 : 1, DataType::Handle(), 1, shared_offset, elements);
  PrimExpr global_addr = global_tensor.access_ptr(
      is_load ? 1 : 2, DataType::Handle(), 1, global_offset, elements);
  PrimExpr total_bytes =
      analyzer->Simplify(elements * shared_tensor->dtype.bytes());
  int barrier_base_id = -1;
  PrimExpr mbar_handle;
  if (is_load) {
    if (auto user_barrier = annotations.Get("barrier")) {
      mbar_handle = Downcast<PrimExpr>(user_barrier.value());
      barrier_base_id = 0;
    } else if (GetIsTmaCopy(op)) {
      LOG(FATAL) << "T.tma_copy() requires a barrier argument. "
                 << "Use T.tma_copy(src, dst, barrier=mbar[idx]).";
    } else if (T.AllocMBarrier) {
      // Internal mbarrier (T.copy()): allocate a single barrier slot.
      // Pipeline buffer versioning expands it per stage when needed.
      barrier_base_id = T.AllocMBarrier(1);
      PrimExpr mbar_idx = IntImm(DataType::Int(32), barrier_base_id);
      mbar_handle = BufferLoad(T.mbarrier_buffer->value(), {mbar_idx});
    }
  }
  PrimExpr load_barrier_arg = barrier_base_id >= 0 ? mbar_handle : tma_barrier;

  Stmt tma_copy;
  if (is_load) {
    tma_copy =
        Evaluate(Call(DataType::Handle(), tma_load(),
                      {shared_addr, global_addr, load_barrier_arg, total_bytes,
                       GetEvictionPolicy(op), GetInnerCachePolicy(op),
                       GetOuterCachePolicy(op)}));
  } else {
    int need_reduce = 0;
    tma_copy =
        Evaluate(Call(DataType::Handle(), tma_store(),
                      {global_addr, shared_addr, total_bytes, need_reduce,
                       GetEvictionPolicy(op), GetInnerCachePolicy(op),
                       GetOuterCachePolicy(op)}));
  }

  if (!is_load) {
    Array<Stmt> seq;
    seq.reserve(3);
    seq.push_back(tma_copy);
    seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_arrive(), {})));
    if (!GetIsTmaCopy(op)) {
      seq.push_back(Evaluate(Call(DataType::Handle(), tma_store_wait(),
                                  {IntImm(DataType::Int(32), 0)})));
    }
    tma_copy = SeqStmt(std::move(seq));
  }

  // For 1D TMA loads with inline mbarrier: emit expect_tx + tma_load
  // (inside thread-gated block), and wait_parity after (all threads).
  if (is_load && barrier_base_id >= 0) {
    Stmt barrier_before_tma_stmt;
    Optional<Stmt> barrier_after_tma_stmt = std::nullopt;
    if (GetIsTmaCopy(op)) {
      // T.tma_copy(): only expect_tx (no arrive). User must call
      // T.barrier_arrive() explicitly. This allows multiple tma_copy operations
      // to share a single arrive.
      barrier_before_tma_stmt =
          Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
    } else {
      // T.copy() with TMA: keep expect_tx and arrive as separate control ops.
      barrier_before_tma_stmt =
          Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                        {mbar_handle, total_bytes}));
      barrier_after_tma_stmt = Evaluate(Call(
          DataType::Handle(), builtin::ptx_arrive_barrier(), {mbar_handle}));
    }

    Array<Stmt> producer_seq{barrier_before_tma_stmt, tma_copy};
    if (barrier_after_tma_stmt.defined()) {
      producer_seq.push_back(barrier_after_tma_stmt.value());
    }

    Stmt producer = IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent),
                               SeqStmt(producer_seq));

    // tma_copy (from T.tma_copy()) is fire-and-forget: only emit the
    // producer (expect_tx + tma_load). The user manages synchronization
    // (arrive + wait) explicitly.
    if (GetIsTmaCopy(op)) {
      return producer;
    }

    // For T.copy() with TMA: emit producer + wait pair so the pipeline/WS
    // passes can split them into different stages.
    Stmt wait_stmt =
        Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                      {mbar_handle, GetCopyMbarPhaseExpr(annotations, T)}));

    return SeqStmt({producer, wait_stmt});
  }

  tma_copy =
      IfThenElse(MakeTmaLeaderCondition(T.thread_bounds->extent), tma_copy);
  return tma_copy;
}

} // namespace musa

namespace {

bool MatchMUSACopyTarget(Target target) { return TargetIsMusa(target); }

bool RegisterMUSACopy() {
  RegisterCopyImpl(CopyImpl{
      "musa.Copy",
      MatchMUSACopyTarget,
      100,
      musa::Copy::InferLayout,
      musa::Copy::Lower,
  });
  return true;
}

const bool musa_copy_registered = RegisterMUSACopy();

} // namespace

} // namespace tl
} // namespace tvm
