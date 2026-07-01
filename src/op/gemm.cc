/*!
 * \file tl/op/gemm.cc
 * \brief Implementation of General Matrix Multiplication (GEMM) operators
 */

#include "gemm.h"

#include "../backend/musa/op/gemm.h"
#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <vector>

#include "../target/utils.h"
#include "tcgen5_meta.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

namespace {

std::vector<GemmImpl> &GemmImplRegistry() {
  static std::vector<GemmImpl> registry;
  return registry;
}

const GemmImpl &ResolveGemmImpl(Target target) {
  const auto &registry = GemmImplRegistry();
  const GemmImpl *matched_impl = nullptr;
  for (const GemmImpl &impl : registry) {
    if (impl.match_target(target)) {
      ICHECK(matched_impl == nullptr)
          << "tl.gemm found multiple target-specific implementations for "
          << target->ToDebugString() << ": " << matched_impl->name << " and "
          << impl.name;
      matched_impl = &impl;
    }
  }
  ICHECK(matched_impl != nullptr)
      << "tl.gemm requires a target-specific implementation, but no gemm "
         "implementation is registered for "
      << target->ToDebugString();
  return *matched_impl;
}

} // namespace

void RegisterGemmImpl(GemmImpl impl) {
  ICHECK(impl.name != nullptr);
  ICHECK(impl.match_target != nullptr);
  ICHECK(impl.select_inst != nullptr);
  ICHECK(impl.compute_warp_partition != nullptr);
  ICHECK(impl.reuse_existing_shared_layout != nullptr);
  ICHECK(impl.instruction_kind != nullptr);
  GemmImplRegistry().push_back(impl);
}

/**
 * @brief Construct a Gemm operator from serialized TL arguments and a buffer
 * map.
 *
 * This constructor deserializes operator parameters from `args` and resolves
 * buffer references via `vmap`, populating an internal GemmNode with:
 * - device pointers for A, B, C and their corresponding Buffer objects,
 * - transpose flags for A and B,
 * - matrix dimensions M, N, K,
 * - warp allocation policy and clear_accum flag,
 * - strides and memory offsets for A and B,
 * - optional kPack (must be 1 or 2) and optional wg_wait.
 *
 * The populated GemmNode is stored into the wrapper's internal `data_`.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      M (Int), N (Int), K (Int), policy (Int), clear_accum (Bool),
 *      stride_A (Int), stride_B (Int), offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) wg_wait (Int)]
 *
 * @note If `kPack` is provided it must be 1; otherwise the constructor
 *       fails with an ICHECK (runtime assertion). No other validation is
 *       performed here.
 */
// NormalizeToBufferRegion moved to src/op/utils.{h,cc}

// MakeAccessPtrFromRegion moved to src/op/utils.{h,cc}

static Array<PrimExpr> MakeRegionMatrixIndices(const BufferRegion &region,
                                               PrimExpr row, PrimExpr col) {
  ICHECK(region.defined());
  ICHECK_GE(region->region.size(), 2U)
      << "GEMM buffer region must have at least 2 dimensions";

  Array<PrimExpr> indices;
  size_t ndim = region->region.size();
  for (size_t i = 0; i + 2 < ndim; ++i) {
    indices.push_back(region->region[i]->min);
  }
  indices.push_back(region->region[ndim - 2]->min + row);
  indices.push_back(region->region[ndim - 1]->min + col);
  return indices;
}

Gemm::Gemm(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GemmNode> node = tvm::ffi::make_object<GemmNode>();

  auto a_access = NormalizeToAccessRegion(args[0], kAccessRead);
  auto b_access = NormalizeToAccessRegion(args[1], kAccessRead);
  auto c_access = NormalizeToAccessRegion(args[2], kAccessReadWrite);

  node->aRegion_ = a_access.region;
  node->bRegion_ = b_access.region;
  node->cRegion_ = c_access.region;
  node->SetAccessRegions({a_access, b_access, c_access});

  node->a_ = node->aRegion_->buffer;
  node->b_ = node->bRegion_->buffer;
  node->c_ = node->cRegion_->buffer;
  node->transA_ = args[3].as<Bool>().value();
  node->transB_ = args[4].as<Bool>().value();
  node->m_ = args[5].as<IntImm>().value()->value;
  node->n_ = args[6].as<IntImm>().value()->value;
  node->k_ = args[7].as<IntImm>().value()->value;
  node->policy_ = GemmWarpPolicy(args[8].as<IntImm>().value()->value);
  node->clearAccum_ = args[9].as<PrimExpr>().value();
  node->strideA_ = args[10].as<IntImm>().value()->value;
  node->strideB_ = args[11].as<IntImm>().value()->value;
  node->offsetA_ = args[12].as<IntImm>().value()->value;
  node->offsetB_ = args[13].as<IntImm>().value()->value;
  if (args.size() > 14) {
    node->kPack_ = args[14].as<IntImm>().value()->value;
    if (node->kPack_ != 1 && node->kPack_ != 2) {
      ICHECK(false) << "kPack must be 1 or 2";
    }
  }
  if (args.size() > 15) {
    node->wgWait_ = args[15].as<IntImm>().value()->value;
  }
  if (args.size() > 16) {
    if (const auto *load = args[16].as<BufferLoadNode>()) {
      node->mbar_ = Downcast<BufferLoad>(args[16]);
    }
  }
  node->cCoords_ = Array<PrimExpr>(
      {args[17].as<PrimExpr>().value(), args[18].as<PrimExpr>().value()});
  node->annotations_ = annotations;
  data_ = std::move(node);
}

AccessRegions GemmNode::GetAccessRegions() const {
  AccessRegions result;
  result.reads.push_back(aRegion_);
  result.reads.push_back(bRegion_);
  if (!is_one(clearAccum_)) {
    result.reads.push_back(cRegion_);
  }
  result.writes.push_back(cRegion_);
  return result;
}

/**
 * @brief Create a copy of this GemmNode as a TileOperator.
 *
 * Constructs a new GemmNode by copying the current node state and returns it
 * wrapped in a Gemm TileOperator.
 *
 * @return TileOperator A Gemm operator that owns a copy of this node.
 */
TileOperator GemmNode::Clone() const {
  auto op = tvm::ffi::make_object<GemmNode>(*this);
  return Gemm(op);
}

String GemmNode::getGemmInstructionKey(int block_size, Target target) const {
  return ResolveGemmImpl(target).select_inst(*this, block_size, target);
}

String GemmNode::getGemmInstructionKind(int block_size, Target target) const {
  const GemmImpl &impl = ResolveGemmImpl(target);
  return impl.instruction_kind(impl.select_inst(*this, block_size, target));
}

std::optional<std::array<int, 3>>
GemmNode::getGemmInstructionShape(int block_size, Target target,
                                  String gemm_inst) const {
  const GemmImpl &impl = ResolveGemmImpl(target);
  if (impl.select_inst_shape == nullptr) {
    return std::nullopt;
  }
  return impl.select_inst_shape(*this, block_size, target, gemm_inst);
}

std::pair<int, int> GemmWarpPolicyNode::computeWarpPartition(
    int M, int N, int block_size, Target target, String gemm_inst,
    std::optional<std::array<int, 3>> mma_inst_shape) const {
  return ResolveGemmImpl(target).compute_warp_partition(
      *this, M, N, block_size, target, gemm_inst, mma_inst_shape);
}

/**
 * @brief Parse and return the numeric GPU architecture from a Target's "arch"
 * attribute.
 *
 * Examines the target's "arch" string and, if it matches the pattern
 * "sm_<num>", returns <num> as an int. If the attribute is present but does not
 * match that pattern, returns 0.
 *
 * Preconditions: the target must have an "arch" attribute (this is checked via
 * ICHECK).
 *
 * @return int The parsed architecture number (e.g., 80 for "sm_80"), or 0 if
 * the arch string does not match "sm_<num>".
 */
static int GetArchInt(Target target) {
  int arch_int = 0;
  auto s = target->GetAttr<tvm::ffi::String>("arch");
  ICHECK(s.has_value());
  std::string arch = s.value();
  if (arch.rfind("sm_", 0) == 0) {
    arch_int = std::stoi(arch.substr(3));
  } else {
    arch_int = 0;
  }
  return arch_int;
}

/**
 * @brief Lower the GEMM operator to a TL TIR call expression.
 *
 * Constructs a tl::gemm call string parameterized by M, N, K, warp partition,
 * transpose flags, accumulation clearing, target-specific stride/offset/kPack
 * and optional workgroup wait value, then returns an Evaluate(call) node
 * invoking tl::tl_gemm with the composed string and the A/B/C buffer handles.
 *
 * @param T Contains lowering context including thread bounds and target.
 * @param analyzer Optional arithmetic analyzer used by lowering (may be
 * nullptr).
 * @return Stmt A TIR statement representing the evaluated TL GEMM call.
 */
Stmt GemmNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  auto block_size = *as_const_int(T.thread_bounds->extent);
  String gemm_inst = getGemmInstructionKey(block_size, T.target);
  if (TargetIsCPU(T.target) && gemm_inst == kGemmInstCPUScalar) {
    auto clear_accum_bool = clearAccum_.as<Bool>();
    ICHECK(clear_accum_bool.has_value())
        << "clear_accum must be a constant Bool type, got " << clearAccum_;

    auto A_buffer = aRegion_->buffer;
    auto B_buffer = bRegion_->buffer;
    auto C_buffer = cRegion_->buffer;

    Var i("i", DataType::Int(32));
    Var j("j", DataType::Int(32));
    Var k("k", DataType::Int(32));

    Buffer accum_buf = decl_buffer({IntImm(DataType::Int(32), 1)}, c_->dtype,
                                   "accum", "local");
    PrimExpr init_val =
        clear_accum_bool.value()
            ? make_const(c_->dtype, 0)
            : BufferLoad(C_buffer, MakeRegionMatrixIndices(cRegion_, i, j));
    Stmt init = BufferStore(accum_buf, init_val, {0});

    Array<PrimExpr> a_indices =
        MakeRegionMatrixIndices(aRegion_, transA_ ? PrimExpr(k) : PrimExpr(i),
                                transA_ ? PrimExpr(i) : PrimExpr(k));
    Array<PrimExpr> b_indices =
        MakeRegionMatrixIndices(bRegion_, transB_ ? PrimExpr(j) : PrimExpr(k),
                                transB_ ? PrimExpr(k) : PrimExpr(j));
    PrimExpr a_val = Cast(c_->dtype, BufferLoad(A_buffer, a_indices));
    PrimExpr b_val = Cast(c_->dtype, BufferLoad(B_buffer, b_indices));
    PrimExpr update_val = BufferLoad(accum_buf, {0}) + a_val * b_val;
    Stmt update = BufferStore(accum_buf, update_val, {0});
    Stmt k_loop =
        For(k, 0, IntImm(DataType::Int(32), k_), ForKind::kSerial, update);

    Stmt store_c = BufferStore(C_buffer, BufferLoad(accum_buf, {0}),
                               MakeRegionMatrixIndices(cRegion_, i, j));
    Stmt body = SeqStmt({init, k_loop, store_c});
    Stmt j_loop =
        For(j, 0, IntImm(DataType::Int(32), n_), ForKind::kSerial, body);
    Stmt i_loop =
        For(i, 0, IntImm(DataType::Int(32), m_), ForKind::kSerial, j_loop);
    return Allocate(accum_buf->data, accum_buf->dtype, accum_buf->shape,
                    const_true(), i_loop);
  }

  std::optional<std::array<int, 3>> qy2_mma_inst = std::nullopt;
  if (TargetIsQY2(T.target) && gemm_inst == kGemmInstMusaQY2MMA) {
    qy2_mma_inst =
        getGemmInstructionShape(block_size, T.target, kGemmInstMusaQY2MMA);
    ICHECK(qy2_mma_inst.has_value())
        << "QY2 extended MMA is selected but no valid MMA instruction is "
           "found.";
  }

  auto [warp_m, warp_n] = policy_->computeWarpPartition(
      m_, n_, block_size, T.target, gemm_inst, qy2_mma_inst);
  std::optional<std::array<int, 3>> ph1_mma_inst = std::nullopt;
  if (TargetIsPH1(T.target) &&
      (gemm_inst == kGemmInstMusaSQMMA || gemm_inst == kGemmInstMusaPH1WMMA)) {
    ph1_mma_inst = getGemmInstructionShape(block_size, T.target, gemm_inst);
    ICHECK(ph1_mma_inst.has_value())
        << (gemm_inst == kGemmInstMusaSQMMA
                ? "PH1 SQMMA is selected but no "
                  "valid SQMMA instruction is found."
                : "PH1 WMMA is selected but no valid "
                  "WMMA instruction is found.");
  }

  if (TargetIsPH1(T.target) && gemm_inst == kGemmInstMusaFMA) {
    auto A_buffer = a_;
    auto B_buffer = b_;
    auto C_buffer = c_;

    auto clear_accum_bool = clearAccum_.as<Bool>();
    ICHECK(clear_accum_bool.has_value())
        << "clear_accum must be a constant Bool type, got " << clearAccum_;

    Var idx_iter("idx_iter", DataType::Int(32));
    Var k_iter("k", DataType::Int(32));

    PrimExpr num_threads = T.thread_bounds->extent;
    PrimExpr local_thread_var = T.thread_var - T.thread_bounds->min;
    PrimExpr total = IntImm(DataType::Int(32), m_ * n_);
    PrimExpr trip = FloorDiv(total + num_threads - 1, num_threads);

    PrimExpr linear = idx_iter * num_threads + local_thread_var;
    PrimExpr guard = linear < total;
    PrimExpr i = FloorDiv(linear, n_);
    PrimExpr j = linear - i * n_;

    auto a_indices = [&]() {
      if (transA_) {
        return MakeRegionMatrixIndices(aRegion_, k_iter, i);
      } else {
        return MakeRegionMatrixIndices(aRegion_, i, k_iter);
      }
    };
    auto b_indices = [&]() {
      if (transB_) {
        return MakeRegionMatrixIndices(bRegion_, j, k_iter);
      } else {
        return MakeRegionMatrixIndices(bRegion_, k_iter, j);
      }
    };

    Buffer accum_buf = decl_buffer({IntImm(DataType::Int(32), 1)}, c_->dtype,
                                   "accum", "local");
    PrimExpr init_val =
        clear_accum_bool.value()
            ? make_const(c_->dtype, 0)
            : BufferLoad(C_buffer, MakeRegionMatrixIndices(cRegion_, i, j));
    Stmt init = BufferStore(accum_buf, init_val, {0});

    PrimExpr a_val = Cast(c_->dtype, BufferLoad(A_buffer, a_indices()));
    PrimExpr b_val = Cast(c_->dtype, BufferLoad(B_buffer, b_indices()));
    PrimExpr update_val = BufferLoad(accum_buf, {0}) + a_val * b_val;
    Stmt update = BufferStore(accum_buf, update_val, {0});
    Stmt k_loop =
        For(k_iter, 0, IntImm(DataType::Int(32), k_), ForKind::kSerial, update);

    Stmt store_c = BufferStore(C_buffer, BufferLoad(accum_buf, {0}),
                               MakeRegionMatrixIndices(cRegion_, i, j));
    Stmt body = SeqStmt({init, k_loop, store_c});
    Stmt guarded = IfThenElse(guard, body);
    Stmt idx_loop = For(idx_iter, 0, trip, ForKind::kSerial, guarded);

    return Allocate(accum_buf->data, accum_buf->dtype, accum_buf->shape,
                    const_true(), idx_loop);
  }

  // Build access pointers from regions locally
  PrimExpr Aptr =
      MakeAccessPtrFromRegion(aRegion_, /*r*/ 1, /*require_2d*/ true);
  PrimExpr Bptr =
      MakeAccessPtrFromRegion(bRegion_, /*r*/ 1, /*require_2d*/ true);
  PrimExpr Cptr =
      MakeAccessPtrFromRegion(cRegion_, /*rw*/ 3, /*require_2d*/ true);

  std::stringstream ss;
  std::string op_name;

  if (gemm_inst == kGemmInstCudaTCGEN05) {
    auto [can_use_tcgen5mma, meta] =
        GetTCGEN5MMAMeta(m_, n_, k_, a_->dtype, c_->dtype);
    ICHECK(can_use_tcgen5mma);
    ICHECK(IsSharedBuffer(b_));
    ICHECK(c_.scope() == "shared.tmem");
    ICHECK(mbar_.defined()) << "mbar must be provided for TCGEN5MMA";
    if (a_.scope() == "shared.tmem") {
      op_name = "tl::tcgen5mma_gemm_ts";
    } else if (IsSharedBuffer(a_)) {
      op_name = "tl::tcgen5mma_gemm_ss";
    } else {
      ICHECK(0)
          << "Unsupported A scope for TCGEN5MMA: "
          << a_.scope(); // If this is triggered, it means Tilelang has bugs.
    }
    ICHECK(wgWait_ == -1)
        << "Currently only wg_wait == -1 is supported for TCGEN5MMA. Please "
           "use "
           "wg_wait = -1 and manually synchronize with mbarrier.";

    std::string accum_dtype = "";
    if (c_->dtype.is_float()) {
      if (c_->dtype.bits() == 32) {
        accum_dtype = "float";
      }
    }
    ICHECK(!accum_dtype.empty())
        << "Unsupported C dtype for TCGEN5MMA: " << c_->dtype;
    ss << op_name << "<" << m_ << ", " << n_ << ", " << k_ << ", ";
    ss << meta.atom_m << ", " << meta.atom_n << ", " << meta.atom_k << ", ";
    ss << transA_ << ", " << transB_ << ", ";
    ss << accum_dtype;
    ss << ">";

    auto C_buffer = T.buffer_remap.count(c_) ? T.buffer_remap[c_] : c_;
    Array<PrimExpr> new_args;
    auto mbarPtr = MakeAccessPtrFromBufferLoad(mbar_, /*rw*/ 3);
    new_args.push_back(StringImm(ss.str()));
    new_args.push_back(Aptr);
    new_args.push_back(Bptr);
    new_args.push_back(BufferLoad(C_buffer, cCoords_));
    new_args.push_back(mbarPtr);
    new_args.push_back(clearAccum_);
    auto new_call = Call(DataType::Handle(), builtin::call_extern(), new_args);

    // Since TCGEN5MMA atoms provided by CUTLASS always have an internal
    // `elect_one_sync()`, we check if we are calling it using full warps
    constexpr int warp_size = 32;
    ICHECK(
        analyzer->CanProveEqual(FloorMod(T.thread_bounds->min, warp_size), 0) &&
        analyzer->CanProveEqual(FloorMod(T.thread_bounds->extent, warp_size),
                                0))
        << "TCGEN5MMA requires thread bounds to be multiples of warp size (32) "
           "and aligned to warps.";
    Stmt tcgen5mma_call;
    if (analyzer->CanProveEqual(T.thread_bounds->extent, warp_size)) {
      // If the thread bounds is exactly one warp, we can use the original call
      tcgen5mma_call = Evaluate(new_call);
    } else {
      // Add an if-else clause
      tcgen5mma_call = IfThenElse(EQ(FloorDiv(T.thread_var, warp_size),
                                     FloorDiv(T.thread_bounds->min, warp_size)),
                                  Evaluate(new_call));
    }
    PrimExpr mbar_phase = T.mbar_phase_expr;
    if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations_)) {
      mbar_phase = explicit_phase.value();
    }
    Stmt wait_stmt = Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(), {mbar_, mbar_phase}));
    return SeqStmt(Array<Stmt>{tcgen5mma_call, wait_stmt});
  }

  if (IsFragmentBuffer(a_)) {
    if (IsFragmentBuffer(b_)) {
      ICHECK(TargetIsPH1(T.target) && gemm_inst == kGemmInstMusaPH1WMMA ||
             TargetIsQY2(T.target))
          << "gemm_rr is currently only implemented for PH1 WMMA and QY2.";
      op_name = "tl::gemm_rr";
    } else {
      op_name = "tl::gemm_rs";
    }
  } else if (IsFragmentBuffer(b_)) {
    op_name = "tl::gemm_sr";
  } else {
    op_name = "tl::gemm_ss";
  }
  ICHECK(IsFragmentBuffer(c_));

  ss << op_name << "<" << m_ << ", " << n_ << ", " << k_ << ", ";
  ss << warp_m << ", " << warp_n << ", ";
  ss << transA_ << ", " << transB_;
  auto clear_accum_bool = clearAccum_.as<Bool>();
  ICHECK(clear_accum_bool.has_value())
      << "clear_accum must be a constant Bool type, got " << clearAccum_;
  ss << ", " << bool(clear_accum_bool.value());
  if ((TargetIsCuda(T.target) && (GetArchInt(T.target) >= 75)) ||
      (TargetIsPH1(T.target)) || (TargetIsQY2(T.target))) {
    ss << ", " << strideA_ << ", " << strideB_;
    ss << ", " << offsetA_ << ", " << offsetB_;
  }
  if (TargetIsCDNA(T.target)) {
    // for cdna gemm, we need to specify kPack
    ss << ", " << kPack_;
  } else if (TargetIsHopper(T.target)) {
    ss << ", " << (gemm_inst == kGemmInstCudaWGMMA ? "true" : "false");
  } else if (TargetIsPH1(T.target)) {
    if (gemm_inst == kGemmInstMusaSQMMA) {
      ss << ", true";
      ss << ", " << (*ph1_mma_inst)[0] << ", " << (*ph1_mma_inst)[1] << ", "
         << (*ph1_mma_inst)[2];
    } else if (gemm_inst == kGemmInstMusaPH1WMMA) {
      ss << ", false";
      ss << ", " << (*ph1_mma_inst)[0] << ", " << (*ph1_mma_inst)[1] << ", "
         << (*ph1_mma_inst)[2];
    } else {
      ICHECK(gemm_inst == kGemmInstMusaSQMMA ||
             gemm_inst == kGemmInstMusaPH1WMMA)
          << "Unexpected PH1 GEMM instruction kind in templated lowering: "
          << gemm_inst;
    }
  } else if (TargetIsQY2(T.target) && gemm_inst == kGemmInstMusaQY2MMA) {
    ICHECK(qy2_mma_inst.has_value())
        << "QY2 extended MMA instruction shape is required for " << gemm_inst;
    ss << ", " << (*qy2_mma_inst)[0] << ", " << (*qy2_mma_inst)[1] << ", "
       << (*qy2_mma_inst)[2];
  }

  if ((TargetIsHopper(T.target) || TargetIsPH1(T.target)) && wgWait_ != 0) {
    ss << ", " << wgWait_;
  } else if (TargetIsSm100(T.target)) {
    // NOTE On sm100, only the leading thread issues the TCGEN5MMA instruction
    // but all threads need to wait, so we emit another statement for cases
    // where wg_wait == 0.
    ICHECK(wgWait_ == 0 || wgWait_ == -1)
        << "wg_wait must be 0 or -1 for Sm100";
  } else {
    ICHECK(wgWait_ == 0)
        << "wg_wait must be 0 for non-Hopper and non-Sm100 targets";
  }
  ss << ">"; // tl::gemm op end

  Array<PrimExpr> gemm_args{StringImm(ss.str()), Aptr, Bptr, Cptr};
  if (TargetIsPH1(T.target) && op_name == "tl::gemm_ss") {
    gemm_args.push_back(T.thread_bounds->min);
  }
  auto new_call = Call(DataType::Handle(), tl::tl_gemm(), gemm_args);
  return AttrStmt(Integer(0), tl::kGemmInst, StringImm(gemm_inst),
                  Evaluate(new_call));
}

/**
 * @brief Infer and bind target-specific memory/layout mappings for A, B, and C.
 *
 * Infers per-buffer layouts (fragment or shared-memory layouts) for this GEMM
 * operator according to the target architecture, thread bounds, warp
 * partitioning, data types, and transpose flags, then binds fragment layouts
 * to the thread range when required.
 *
 * Preconditions:
 * - C.scope() == "local.fragment"
 *
 * Side effects:
 * - Marks layout inference as completed (sets completed_ = true).
 * - May abort via ICHECK on unsupported targets, invalid buffer scopes, or
 *   incompatible shape constraints.
 *
 * @param T Input layout-inference context (provides thread bounds and target).
 * @return LayoutMap mapping A, B, and C to their inferred layouts.
 */
LayoutMap GemmNode::InferLayout(const LayoutInferArgs &T,
                                InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;
  auto thread_range = T.thread_bounds;
  auto block_size = *as_const_int(thread_range->extent);
  String gemm_inst = getGemmInstructionKey(block_size, T.target);
  if (TargetIsCPU(T.target) && gemm_inst == kGemmInstCPUScalar) {
    completed_ = true;
    return {};
  }
  std::optional<std::array<int, 3>> qy2_mma_inst = std::nullopt;
  if (TargetIsQY2(T.target) && gemm_inst == kGemmInstMusaQY2MMA) {
    qy2_mma_inst =
        getGemmInstructionShape(block_size, T.target, kGemmInstMusaQY2MMA);
    ICHECK(qy2_mma_inst.has_value())
        << "QY2 extended MMA is selected but no valid MMA instruction is "
           "found.";
  }
  auto [warp_m, warp_n] = policy_->computeWarpPartition(
      m_, n_, block_size, T.target, gemm_inst, qy2_mma_inst);
  if (TargetIsVolta(T.target)) {
    ICHECK(IsFragmentBuffer(c_))
        << "Volta gemm only supports C in local.fragment scope, got "
        << c_.scope();
    auto fragment = makeGemmVoltaFragmentC(m_, n_, m_ / warp_m, n_ / warp_n,
                                           c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));
    if (IsSharedBuffer(a_)) {
      int dim_A = a_->shape.size();
      auto layout = makeGemmVoltaABLayout(*as_const_int(a_->shape[dim_A - 2]),
                                          *as_const_int(a_->shape[dim_A - 1]),
                                          true, !transA_);
      results.Set(a_, ExpandLayoutToMatchBuffer(layout, a_));
    } else if (IsFragmentBuffer(a_)) {
      ICHECK(transA_ == false);
      auto fragment =
          makeGemmVoltaFragmentA(m_, n_, k_, m_ / warp_m, n_ / warp_n);
      results.Set(a_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }

    ICHECK(IsSharedBuffer(b_));
    int dim_B = b_->shape.size();
    auto layout = makeGemmVoltaABLayout(*as_const_int(b_->shape[dim_B - 2]),
                                        *as_const_int(b_->shape[dim_B - 1]),
                                        false, transB_);
    results.Set(b_, ExpandLayoutToMatchBuffer(layout, b_));
  } else if (TargetIsAmpere(T.target) || TargetIsTuring(T.target) ||
             TargetIsSM120(T.target) ||
             (TargetIsSm100(T.target) && gemm_inst == kGemmInstCudaMMA)) {
    ICHECK(IsFragmentBuffer(c_))
        << "MMA only supports C in local.fragment scope, got " << c_.scope();

    auto fragment =
        makeGemmFragmentC(m_, n_, m_ / warp_m, n_ / warp_n, c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));

    if (IsSharedBuffer(a_)) {
      int dim_A = a_->shape.size();
      const int64_t mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      auto layout = makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                     a_->dtype.bits(), !transA_);
      results.Set(a_, ExpandLayoutToMatchBuffer(layout, a_));
    } else if (IsFragmentBuffer(a_)) {
      auto fragment = makeGemmFragmentA(m_, n_, k_, m_ / warp_m, n_ / warp_n,
                                        a_->dtype.bits(), transA_);
      results.Set(a_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (IsSharedBuffer(b_)) {
      int dim_B = b_->shape.size();
      const int64_t mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      auto layout = makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                     b_->dtype.bits(), transB_);
      results.Set(b_, ExpandLayoutToMatchBuffer(layout, b_));
    } else if (IsFragmentBuffer(b_)) {
      auto fragment =
          makeGemmFragmentB(m_, n_, k_, m_ / warp_m, n_ / warp_n, transB_);
      results.Set(b_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else if (TargetIsHopper(T.target)) {
    ICHECK(IsFragmentBuffer(c_))
        << (gemm_inst == kGemmInstCudaWGMMA ? "WGMMA " : "MMA ")
        << "only supports C in local.fragment scope, got " << c_.scope();
    auto fragment = gemm_inst == kGemmInstCudaWGMMA
                        ? makeGemmFragmentCHopper(m_, n_, m_ / warp_m,
                                                  n_ / warp_n, c_->dtype.bits())
                        : makeGemmFragmentC(m_, n_, m_ / warp_m, n_ / warp_n,
                                            c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));
    if (IsSharedBuffer(a_)) {
      int dim_A = a_->shape.size();
      const int64_t mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      const int64_t continuity =
          transA_ ? 4 * mat_continuous / warp_m : mat_continuous;
      auto ABLayout =
          gemm_inst == kGemmInstCudaWGMMA
              ? makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                       a_->dtype.bits(), !transA_)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 a_->dtype.bits(), !transA_);
      results.Set(a_, ExpandLayoutToMatchBuffer(ABLayout, a_));
    } else {
      auto fragment = makeGemmFragmentA(m_, n_, k_, m_ / warp_m, n_ / warp_n,
                                        a_->dtype.bits(), transA_);
      results.Set(a_, fragment->BindThreadRange(thread_range));
    }
    if (IsSharedBuffer(b_)) {
      int dim_B = b_->shape.size();
      const int64_t mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      const int64_t continuity =
          transB_ ? mat_continuous : mat_continuous / warp_n;

      auto ABLayout =
          gemm_inst == kGemmInstCudaWGMMA
              ? makeGemmABLayoutHopper(mat_stride, mat_continuous, continuity,
                                       b_->dtype.bits(), transB_)
              : makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                 b_->dtype.bits(), transB_);
      results.Set(b_, ExpandLayoutToMatchBuffer(ABLayout, b_));
    } else {
      auto fragment =
          makeGemmFragmentB(m_, n_, k_, m_ / warp_m, n_ / warp_n, transB_);
      results.Set(b_, fragment->BindThreadRange(thread_range));
    }
  } else if (TargetIsPH1(T.target)) {
    if (gemm_inst == kGemmInstMusaSQMMA) {
      ICHECK(a_.scope() == "shared" || a_.scope() == "shared.dyn");
      ICHECK(b_.scope() == "shared" || b_.scope() == "shared.dyn");
      ICHECK(c_.scope() == "local.fragment");
      auto sqmma_inst =
          getGemmInstructionShape(block_size, T.target, kGemmInstMusaSQMMA);
      ICHECK(sqmma_inst.has_value())
          << "SQMMA is selected but no valid SQMMA instruction is found.";
      auto fragment = makePHSqmmaFragmentC(m_, n_, warp_m, warp_n,
                                           c_->dtype.bits(), *sqmma_inst);
      results.Set(c_, fragment->BindThreadRange(thread_range));

      int dim_A = a_->shape.size();
      const int64_t a_mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t a_mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      const int64_t a_continuity = a_mat_continuous;
      const bool a_is_fp8 = musa::IsPH1SupportedFp8(a_->dtype);
      auto ALayout =
          (transA_ && a_is_fp8)
              ? musa::MakeTransposedPH1SqmmaOperandLayout(
                    a_mat_stride, a_mat_continuous, m_, k_, a_->dtype.bits(),
                    /*k_inner=*/true)
              : makeGemmABLayoutPH1(a_mat_stride, a_mat_continuous,
                                    a_continuity, a_->dtype.bits(), !transA_);
      // PH1 32x32x32 SQMMA kernels require the legacy K-inner expansion to
      // match tilelang_musa_6 shared-memory staging.
      const bool a_need_legacy_repeat =
          (a_mat_stride == 32 && a_mat_continuous == 32);
      if (a_need_legacy_repeat) {
        const int a_bits = a_->dtype.bits();
        const int a_repeat_factor =
            (a_bits > 0 && a_bits <= 32) ? (32 / a_bits) : 1;
        results.Set(a_, ALayout->Repeat(/*dim=*/1, /*factor=*/a_repeat_factor));
      } else {
        results.Set(a_, ALayout);
      }

      int dim_B = b_->shape.size();
      const int64_t b_mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t b_mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      int64_t b_continuity = b_mat_continuous;
      if (!transB_) {
        b_continuity = (*sqmma_inst)[1];
      }
      const bool b_is_fp8 = musa::IsPH1SupportedFp8(b_->dtype);
      auto BLayout =
          (!transB_ && b_is_fp8)
              ? musa::MakeTransposedPH1SqmmaOperandLayout(
                    b_mat_stride, b_mat_continuous, n_, k_, b_->dtype.bits(),
                    /*k_inner=*/true)
              : makeGemmABLayoutPH1(b_mat_stride, b_mat_continuous,
                                    b_continuity, b_->dtype.bits(), transB_);
      const bool b_need_legacy_repeat =
          (b_mat_stride == 32 && b_mat_continuous == 32);
      if (b_need_legacy_repeat) {
        const int b_bits = b_->dtype.bits();
        const int b_repeat_factor =
            (b_bits > 0 && b_bits <= 32) ? (32 / b_bits) : 1;
        results.Set(b_, BLayout->Repeat(/*dim=*/1, /*factor=*/b_repeat_factor));
      } else {
        results.Set(b_, BLayout);
      }
    } else if (gemm_inst == kGemmInstMusaPH1WMMA) {
      const bool a_is_shared =
          a_.scope() == "shared" || a_.scope() == "shared.dyn";
      const bool b_is_shared =
          b_.scope() == "shared" || b_.scope() == "shared.dyn";
      const bool a_is_fragment = IsFragmentBuffer(a_);
      const bool b_is_fragment = IsFragmentBuffer(b_);
      ICHECK((a_is_shared && b_is_shared) || (a_is_fragment && b_is_fragment))
          << "PH1 WMMA requires A/B to both be in shared/shared.dyn or both "
             "be in local.fragment scope, got A="
          << a_.scope() << ", B=" << b_.scope();
      ICHECK(c_.scope() == "local.fragment")
          << "PH1 WMMA requires C in local.fragment scope, got " << c_.scope();
      auto wmma_inst =
          getGemmInstructionShape(block_size, T.target, kGemmInstMusaPH1WMMA);
      ICHECK(wmma_inst.has_value())
          << "PH1 WMMA is selected but no valid WMMA instruction is found.";

      auto fragment = makePH1WmmaCLayout(m_, n_, warp_m, warp_n,
                                         c_->dtype.bits(), *wmma_inst);
      results.Set(c_, fragment->BindThreadRange(thread_range));

      if (a_is_shared && b_is_shared) {
        int dim_A = a_->shape.size();
        const int64_t a_mat_stride = *as_const_int(a_->shape[dim_A - 2]);
        const int64_t a_mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
        auto ALayout =
            makePH1WmmaABLayout(a_mat_stride, a_mat_continuous,
                                a_mat_continuous, a_->dtype.bits(), !transA_);
        results.Set(a_, ExpandLayoutToMatchBuffer(ALayout, a_));

        int dim_B = b_->shape.size();
        const int64_t b_mat_stride = *as_const_int(b_->shape[dim_B - 2]);
        const int64_t b_mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
        auto BLayout =
            makePH1WmmaABLayout(b_mat_stride, b_mat_continuous,
                                b_mat_continuous, b_->dtype.bits(), transB_);
        results.Set(b_, ExpandLayoutToMatchBuffer(BLayout, b_));
      } else {
        auto fragment_a = makePH1WmmaFragmentA(
            m_, n_, k_, warp_m, warp_n, a_->dtype.bits(), transA_, *wmma_inst);
        results.Set(a_, fragment_a->BindThreadRange(thread_range));

        auto fragment_b = makePH1WmmaFragmentB(
            m_, n_, k_, warp_m, warp_n, b_->dtype.bits(), transB_, *wmma_inst);
        results.Set(b_, fragment_b->BindThreadRange(thread_range));
      }
    } else {
      auto fragment = makeGemmFragmentCLinear(m_, n_, block_size);
      results.Set(c_, fragment->BindThreadRange(thread_range));

      int dim_A = a_->shape.size();
      const int64_t a_mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t a_mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      results.Set(a_, makeLinearLayout(Array<PrimExpr>{
                          Integer(a_mat_stride), Integer(a_mat_continuous)}));

      int dim_B = b_->shape.size();
      const int64_t b_mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t b_mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      results.Set(b_, makeLinearLayout(Array<PrimExpr>{
                          Integer(b_mat_stride), Integer(b_mat_continuous)}));
    }
  } else if (TargetIsQY2(T.target)) {
    ICHECK(c_.scope() == "local.fragment")
        << "QY2 MMA only supports C in local.fragment scope, got "
        << c_.scope();
    const bool use_qy2_wmma = qy2_mma_inst.has_value();

    auto fragment =
        use_qy2_wmma ? makeGemmQY2WmmaCLayout(m_, n_, m_ / warp_m, n_ / warp_n,
                                              c_->dtype.bits(), *qy2_mma_inst)
                     : makeGemmQY2FragmentC(m_, n_, m_ / warp_m, n_ / warp_n,
                                            c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));

    if (IsSharedBuffer(a_)) {
      int dim_A = a_->shape.size();
      const int64_t mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      auto layout = makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                     a_->dtype.bits(), !transA_);
      results.Set(a_, ExpandLayoutToMatchBuffer(layout, a_));
    } else if (IsFragmentBuffer(a_)) {
      auto fragment =
          use_qy2_wmma
              ? makeGemmQY2WmmaFragmentA(m_, n_, k_, m_ / warp_m, n_ / warp_n,
                                         a_->dtype.bits(), transA_,
                                         *qy2_mma_inst)
              : (transA_
                     ? makeGemmQY2FragmentACol(m_, n_, k_, m_ / warp_m,
                                               n_ / warp_n, a_->dtype.bits())
                     : makeGemmQY2FragmentARow(m_, n_, k_, m_ / warp_m,
                                               n_ / warp_n, a_->dtype.bits()));
      results.Set(a_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (IsSharedBuffer(b_)) {
      int dim_B = b_->shape.size();
      const int64_t mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      auto layout = makeGemmABLayout(mat_stride, mat_continuous, mat_continuous,
                                     b_->dtype.bits(), transB_);
      results.Set(b_, ExpandLayoutToMatchBuffer(layout, b_));
    } else if (IsFragmentBuffer(b_)) {
      auto fragment =
          use_qy2_wmma
              ? makeGemmQY2WmmaFragmentB(m_, n_, k_, m_ / warp_m, n_ / warp_n,
                                         b_->dtype.bits(), transB_,
                                         *qy2_mma_inst)
              : (transB_
                     ? makeGemmQY2FragmentBRow(m_, n_, k_, m_ / warp_m,
                                               n_ / warp_n, b_->dtype.bits())
                     : makeGemmQY2FragmentBCol(m_, n_, k_, m_ / warp_m,
                                               n_ / warp_n, b_->dtype.bits()));
      results.Set(b_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else if (gemm_inst == kGemmInstCudaTCGEN05) {
    ICHECK(c_.scope() == "shared.tmem")
        << "TCGEN5MMA only supports C in shared.tmem scope, got " << c_.scope();
    ICHECK(IsSharedBuffer(a_))
        << "Current TCGEN5MMA only supports A in shared.dyn scope";
    auto [can_use_tcgen5mma, meta] =
        GetTCGEN5MMAMeta(m_, n_, k_, a_->dtype, c_->dtype);
    ICHECK(can_use_tcgen5mma);
    {
      int dim_A = a_->shape.size();
      const int64_t mat_stride = *as_const_int(a_->shape[dim_A - 2]);
      const int64_t mat_continuous = *as_const_int(a_->shape[dim_A - 1]);
      auto layout =
          makeGemmABLayoutSm100(mat_stride, mat_continuous, mat_continuous,
                                a_->dtype.bits(), transA_ ? 1 : 2);
      results.Set(a_, ExpandLayoutToMatchBuffer(layout, a_));
    }
    {
      int dim_B = b_->shape.size();
      const int64_t mat_stride = *as_const_int(b_->shape[dim_B - 2]);
      const int64_t mat_continuous = *as_const_int(b_->shape[dim_B - 1]);
      const int64_t continuity = mat_continuous;
      auto layout =
          makeGemmABLayoutSm100(mat_stride, mat_continuous, continuity,
                                b_->dtype.bits(), transB_ ? 2 : 1);
      results.Set(b_, ExpandLayoutToMatchBuffer(layout, b_));
    }
    {
      Layout res;
      IterVar i = make_itervar("i", m_);
      IterVar j = make_itervar("j", n_);
      ICHECK(m_ % meta.atom_m == 0);
      PrimExpr atom_idx = FloorDiv(i, meta.atom_m) +
                          FloorDiv(j, meta.atom_n) * (m_ / meta.atom_m);
      PrimExpr ai = FloorMod(i, meta.atom_m); // "ai" means "atom_i"
      PrimExpr aj = FloorMod(j, meta.atom_n);
      if (meta.atom_m == 128) {
        // Layout D
        // (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-d)
        res = Layout(Array{i, j}, {ai, aj + atom_idx * meta.atom_n});
      } else if (meta.atom_m == 64) {
        // Layout E
        // (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-e)
        // since .ws variant is used About why we use .ws variant here, please
        // refer to gemm_sm100.h
        res = Layout(Array{i, j}, {FloorDiv(ai, 32) * 32 + FloorMod(ai, 32) +
                                       FloorDiv(aj, meta.atom_n / 2) * 64,
                                   FloorMod(aj, meta.atom_n / 2) +
                                       atom_idx * (meta.atom_n / 2)});
      } else if (meta.atom_m == 32) {
        // Layout G
        // (https://docs.nvidia.com/cuda/parallel-thread-execution/#tcgen05-data-path-layout-g)
        res = Layout(
            Array{i, j},
            {FloorMod(ai, 32) + FloorDiv(aj, meta.atom_n / 4) * 32,
             FloorMod(aj, meta.atom_n / 4) + atom_idx * (meta.atom_n / 4)});
      } else {
        ICHECK(0);
      }
      results.Set(c_, res);
    }
  } else if (TargetIsCDNA(T.target)) {
    ICHECK(IsFragmentBuffer(c_))
        << "CDNA gemm (FMMA) only supports C in local.fragment scope, got "
        << c_.scope();
    auto fragment = makeGemmFragmentCCDNA(m_, n_, m_ / warp_m, n_ / warp_n,
                                          c_->dtype.bits());
    results.Set(c_, fragment->BindThreadRange(thread_range));

    if (IsSharedBuffer(a_)) {
      int dim_A = a_->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(a_->shape[dim_A - 2]),
          *as_const_int(a_->shape[dim_A - 1]), a_->dtype.bits(), kPack_);
      results.Set(a_, ExpandLayoutToMatchBuffer(shared_layout, a_));
    } else if (IsFragmentBuffer(a_)) {
      auto fragment =
          makeGemmFragmentACDNA(m_, n_, k_, m_ / warp_m, n_ / warp_n,
                                a_->dtype.bits(), kPack_, transA_);
      results.Set(a_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
    if (IsSharedBuffer(b_)) {
      int dim_B = b_->shape.size();
      auto shared_layout = makeGemmABLayoutCDNA(
          *as_const_int(b_->shape[dim_B - 2]),
          *as_const_int(b_->shape[dim_B - 1]), b_->dtype.bits(), kPack_);

      results.Set(b_, ExpandLayoutToMatchBuffer(shared_layout, b_));
    } else if (IsFragmentBuffer(b_)) {
      auto fragment =
          makeGemmFragmentB(m_, n_, k_, m_ / warp_m, n_ / warp_n, transB_);
      results.Set(b_, fragment->BindThreadRange(thread_range));
    } else {
      ICHECK(0);
    }
  } else {
    ICHECK(0) << "Not supported " << T.target->str();
  }
  completed_ = true;
  return results;
}

TIR_REGISTER_TL_TILE_OP(Gemm, gemm)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_REGISTER_OP("tl.GemmWarpPolicy")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "GemmWarpPolicy");

TVM_FFI_STATIC_INIT_BLOCK() {
  GemmNode::RegisterReflection();
  GemmWarpPolicyNode::RegisterReflection();
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.GemmWarpPolicyComputeWarpPartition",
                        [](GemmWarpPolicy policy, int M, int N, int block_size,
                           Target target, String gemm_inst) {
                          policy->computeWarpPartition(M, N, block_size, target,
                                                       gemm_inst);
                        });
}

} // namespace tl
} // namespace tvm
