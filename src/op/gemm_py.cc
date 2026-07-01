/*!
 * \file tl/op/gemm_py.cc
 * \brief Implementation of General Matrix Multiplication (GEMM) operators
 */

#include "gemm_py.h"

#include "builtin.h"
#include <tvm/tir/builtin.h>
#include <tvm/tir/op.h>
#include <tvm/tir/op_attr_types.h>
#include <tvm/tir/transform.h>
#include <vector>

#include "../target/utils.h"
#include "tcgen5_meta.h"
#include "utils.h"

namespace tvm {
namespace tl {

using namespace tir;

// NormalizeToBufferRegion moved to src/op/utils.{h,cc}

// MakeAccessPtrFromRegion moved to src/op/utils.{h,cc}

/**
 * @brief Construct a Gemm operator from serialized TL arguments and a buffer
 * map.
 *
 * This constructor deserializes operator parameters from `args` and resolves
 * buffer references via `vmap`, populating an internal GemmPyNode with:
 * - device pointers for A, B, C and their corresponding Buffer objects,
 * - transpose flags for A and B,
 * - matrix dimensions M, N, K,
 * - warp allocation policy and clear_accum flag,
 * - strides and memory offsets for A and B,
 * - optional kPack (must be 1 or 2) and optional wg_wait.
 *
 * The populated GemmPyNode is stored into the wrapper's internal `data_`.
 *
 * @param args Positional serialized arguments produced by the TL frontend:
 *   expected layout is:
 *     [Aptr, Bptr, Cptr, trans_A (Bool), trans_B (Bool),
 *      M (Int), N (Int), K (Int), policy (Int), clear_accum (Bool),
 *      stride_A (Int), stride_B (Int), offset_A (Int), offset_B (Int),
 *      (optional) kPack (Int), (optional) wg_wait (Int)]
 *
 * @note If `kPack` is provided it must be 1 or 2; otherwise the constructor
 *       fails with an ICHECK (runtime assertion). No other validation is
 *       performed here.
 */
GemmPy::GemmPy(Array<PrimExpr> args, Map<String, ObjectRef> annotations) {
  ObjectPtr<GemmPyNode> node = tvm::ffi::make_object<GemmPyNode>();

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

AccessRegions GemmPyNode::GetAccessRegions() const {
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
 * @brief Create a copy of this GemmPyNode as a TileOperator.
 *
 * Constructs a new GemmPyNode by copying the current node state and returns it
 * wrapped in a Gemm TileOperator.
 *
 * @return TileOperator A Gemm operator that owns a copy of this node.
 */
TileOperator GemmPyNode::Clone() const {
  auto op = tvm::ffi::make_object<GemmPyNode>(*this);
  return GemmPy(op);
}

bool GemmPyNode::allowTcgen5Mma(Target target) const {
  bool scope_ok = (IsSharedBuffer(a_) || a_.scope() == "shared.tmem") &&
                  IsSharedBuffer(b_) && c_.scope() == "shared.tmem";
  if (!TargetIsSm100(target) || !scope_ok)
    return false;
  // For TS variant (A from TMEM), use B's dtype as the input dtype
  DataType ab_dtype = (a_.scope() == "shared.tmem") ? b_->dtype : a_->dtype;
  return GetTCGEN5MMAMeta(m_, n_, k_, ab_dtype, c_->dtype).first;
}

bool GemmPyNode::allowWgmma(int block_size, Target target) const {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();

  int warp_size = TargetGetWarpSize(target);
  int num_warps = block_size / warp_size;
  return !ctxt->GetConfig(kDisableWGMMA, Optional<Bool>()).value_or(false) &&
         TargetIsHopper(target) && (this->m_ >= 64) && (num_warps % 4 == 0) &&
         checkWgmma();
}

String GemmPyNode::getGemmInstructionKey(int block_size, Target target) const {
  bool allow_sqmma = AllowSQMMA(block_size, target);
  if (allowTcgen5Mma(target)) {
    return kGemmInstCudaTCGEN05;
  } else if (allowWgmma(block_size, target)) {
    return kGemmInstCudaWGMMA;
  } else if (TargetIsCDNA(target)) {
    return kGemmInstROCmMFMA;
  } else if (TargetIsQY2(target)) {
    return kGemmInstMusaMMA;
  } else if (TargetIsCuda(target)) {
    return kGemmInstCudaMMA;
  } else if (TargetIsPH1(target)) {
    if (allow_sqmma) {
      return kGemmInstMusaSQMMA;
    }
    if (AllowPH1Wmma(block_size, target)) {
      return kGemmInstMusaPH1WMMA;
    }
    return kGemmInstMusaFMA;
  } else if (TargetIsCPU(target)) {
    return kGemmInstCPUScalar;
  } else {
    ICHECK(0) << "Unsupported target for gemm: " << target->str();
    return kGemmInstCudaMMA;
  }
}

bool GemmPyNode::AllowSQMMA(int block_size, Target target) const {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  if (ctxt->GetConfig(kDisableSQMMA, Optional<Bool>()).value_or(false)) {
    return false;
  }
  return SelectSQMMAInstShape(block_size, target).has_value();
}

std::optional<std::array<int, 3>>
GemmPyNode::SelectSQMMAInstShape(int block_size, Target target) const {
  if (!TargetIsPH1(target)) {
    return std::nullopt;
  }
  if (a_.scope() != "shared.dyn" && a_.scope() != "shared") {
    return std::nullopt;
  }
  if (b_.scope() != "shared.dyn" && b_.scope() != "shared") {
    return std::nullopt;
  }
  if (c_.scope() != "local.fragment") {
    return std::nullopt;
  }
  int warp_size = TargetGetWarpSize(target);
  if (block_size % warp_size != 0) {
    return std::nullopt;
  }
  int num_warps = block_size / warp_size;
  if (num_warps % 4 != 0) {
    return std::nullopt;
  }
  auto warp_parts = policy_->computeWarpPartition(m_, n_, block_size, target,
                                                  kGemmInstMusaSQMMA);
  int warp_m = warp_parts.first;
  int warp_n = warp_parts.second;
  if (warp_m <= 0 || warp_n <= 0) {
    return std::nullopt;
  }
  if (warp_m % 4 != 0) {
    return std::nullopt;
  }
  int warp_groups_m = warp_m / 4;
  if (warp_groups_m <= 0) {
    return std::nullopt;
  }
  if (m_ % (warp_m * 4) != 0) {
    return std::nullopt;
  }
  if (n_ % (warp_n * 8) != 0) {
    return std::nullopt;
  }
  int64_t atom_m = m_ / warp_groups_m;
  int64_t atom_n = n_ / warp_n;

  const auto &a_dtype = a_->dtype;
  const auto &b_dtype = b_->dtype;
  const auto &c_dtype = c_->dtype;
  const bool major_a_is_k = !transA_;
  const bool major_b_is_k = transB_;

  enum class SqmmaTypeClass : uint8_t {
    kInt8,
    kUInt8,
    kFP16,
    kBF16,
    kTF32,
    kFP8
  };
  std::optional<SqmmaTypeClass> type_class = std::nullopt;

  if (a_dtype == DataType::Float(16) && b_dtype == DataType::Float(16) &&
      c_dtype == DataType::Float(32)) {
    type_class = SqmmaTypeClass::kFP16;
  } else if (a_dtype == DataType::BFloat(16) &&
             b_dtype == DataType::BFloat(16) &&
             c_dtype == DataType::Float(32)) {
    type_class = SqmmaTypeClass::kBF16;
  } else if (a_dtype == DataType::Float(32) && b_dtype == DataType::Float(32) &&
             c_dtype == DataType::Float(32)) {
    type_class = SqmmaTypeClass::kTF32;
  } else if (((a_dtype.is_float8_e4m3() && b_dtype.is_float8_e4m3()) ||
              (a_dtype.is_float8_e5m2() && b_dtype.is_float8_e5m2()) ||
              (a_dtype.is_float8_e4m3() && b_dtype.is_float8_e5m2()) ||
              (a_dtype.is_float8_e5m2() && b_dtype.is_float8_e4m3())) &&
             c_dtype == DataType::Float(32)) {
    type_class = SqmmaTypeClass::kFP8;
  } else if (a_dtype == DataType::Int(8) && b_dtype == DataType::Int(8) &&
             c_dtype == DataType::Int(32)) {
    type_class = SqmmaTypeClass::kInt8;
  } else if (a_dtype == DataType::UInt(8) && b_dtype == DataType::UInt(8) &&
             c_dtype == DataType::UInt(32)) {
    type_class = SqmmaTypeClass::kUInt8;
  } else {
    return std::nullopt;
  }

  if ((*type_class == SqmmaTypeClass::kFP16 ||
       *type_class == SqmmaTypeClass::kBF16) &&
      transA_ && warp_groups_m > 1) {
    return std::nullopt;
  }
  // PH1 TF32 SQMMA is reliable for the basic tile range covered by the
  // dedicated fp32 SQMMA tests; larger multi-inst tiles fall back to WMMA.
  if (*type_class == SqmmaTypeClass::kTF32 &&
      (m_ > 128 || n_ > 128 || k_ > 32)) {
    return std::nullopt;
  }

  auto select_inst_n = [](int inst_m, SqmmaTypeClass tc) -> std::vector<int> {
    if (tc == SqmmaTypeClass::kTF32) {
      if (inst_m == 128)
        return {128, 64};
      if (inst_m == 64)
        return {64, 32, 16};
      if (inst_m == 32)
        return {64, 32};
      if (inst_m == 16)
        return {64};
      return {};
    }
    if (inst_m == 128)
      return {128, 64, 32};
    if (inst_m == 64)
      return {128, 64, 32, 16};
    if (inst_m == 32)
      return {128, 64, 32};
    if (inst_m == 16)
      return {64};
    return {};
  };

  std::vector<int> inst_k_candidates;
  if (*type_class == SqmmaTypeClass::kFP16 ||
      *type_class == SqmmaTypeClass::kBF16) {
    inst_k_candidates = {64, 32, 16};
  } else if (*type_class == SqmmaTypeClass::kTF32) {
    inst_k_candidates = {32, 16, 8};
  } else {
    inst_k_candidates = {128, 64, 32};
  }

  for (int inst_m : {128, 64, 32, 16}) {
    if (atom_m % inst_m != 0)
      continue;
    if (*type_class == SqmmaTypeClass::kTF32 && inst_m == 128 &&
        (!major_a_is_k || !major_b_is_k)) {
      continue;
    }
    for (int inst_n : select_inst_n(inst_m, *type_class)) {
      if (atom_n % inst_n != 0)
        continue;
      for (int inst_k : inst_k_candidates) {
        if (k_ % inst_k == 0) {
          return std::array<int, 3>{inst_m, inst_n, inst_k};
        }
      }
    }
  }

  return std::nullopt;
}

std::optional<std::array<int, 3>>
GemmPyNode::SelectPH1WmmaInstShape(int block_size, Target target) const {
  if (!TargetIsPH1(target)) {
    return std::nullopt;
  }
  const bool a_is_shared = a_.scope() == "shared.dyn" || a_.scope() == "shared";
  const bool b_is_shared = b_.scope() == "shared.dyn" || b_.scope() == "shared";
  const bool a_is_fragment = IsFragmentBuffer(a_);
  const bool b_is_fragment = IsFragmentBuffer(b_);
  if (!((a_is_shared && b_is_shared) || (a_is_fragment && b_is_fragment))) {
    return std::nullopt;
  }
  if (c_.scope() != "local.fragment") {
    return std::nullopt;
  }

  int warp_size = TargetGetWarpSize(target);
  if (block_size % warp_size != 0) {
    return std::nullopt;
  }
  if (m_ % 4 != 0 || n_ % 8 != 0) {
    return std::nullopt;
  }

  auto warp_parts = policy_->computeWarpPartition(m_, n_, block_size, target,
                                                  kGemmInstMusaPH1WMMA);
  int warp_m = warp_parts.first;
  int warp_n = warp_parts.second;
  if (warp_m <= 0 || warp_n <= 0) {
    return std::nullopt;
  }
  if (m_ % warp_m != 0 || n_ % warp_n != 0) {
    return std::nullopt;
  }

  int warp_tile_m = m_ / warp_m;
  int warp_tile_n = n_ / warp_n;
  const auto &a_dtype = a_->dtype;
  const auto &b_dtype = b_->dtype;
  const auto &c_dtype = c_->dtype;

  enum class Ph1WmmaTypeClass : uint8_t {
    kF16F16F32,
    kBF16BF16F32,
    kTF32TF32F32,
    kS8S8S32,
    kU8U8U32,
    kF16S8F32,
    kBF16S8F32,
    kS8F16F32,
    kS8BF16F32,
    kFP8F32,
  };
  std::optional<Ph1WmmaTypeClass> type_class = std::nullopt;

  const bool a_is_ph1_fp8 = a_dtype.is_float8_e4m3() ||
                            a_dtype.is_float8_e4m3fn() ||
                            a_dtype.is_float8_e5m2();
  const bool b_is_ph1_fp8 = b_dtype.is_float8_e4m3() ||
                            b_dtype.is_float8_e4m3fn() ||
                            b_dtype.is_float8_e5m2();
  if (a_dtype == DataType::Float(16) && b_dtype == DataType::Float(16) &&
      c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kF16F16F32;
  } else if (a_dtype == DataType::BFloat(16) &&
             b_dtype == DataType::BFloat(16) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kBF16BF16F32;
  } else if (a_dtype == DataType::Float(32) && b_dtype == DataType::Float(32) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kTF32TF32F32;
  } else if (a_dtype == DataType::Int(8) && b_dtype == DataType::Int(8) &&
             c_dtype == DataType::Int(32)) {
    type_class = Ph1WmmaTypeClass::kS8S8S32;
  } else if (a_dtype == DataType::UInt(8) && b_dtype == DataType::UInt(8) &&
             c_dtype == DataType::UInt(32)) {
    type_class = Ph1WmmaTypeClass::kU8U8U32;
  } else if (a_dtype == DataType::Float(16) && b_dtype == DataType::Int(8) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kF16S8F32;
  } else if (a_dtype == DataType::BFloat(16) && b_dtype == DataType::Int(8) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kBF16S8F32;
  } else if (a_dtype == DataType::Int(8) && b_dtype == DataType::Float(16) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kS8F16F32;
  } else if (a_dtype == DataType::Int(8) && b_dtype == DataType::BFloat(16) &&
             c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kS8BF16F32;
  } else if (a_is_ph1_fp8 && b_is_ph1_fp8 && c_dtype == DataType::Float(32)) {
    type_class = Ph1WmmaTypeClass::kFP8F32;
  } else {
    return std::nullopt;
  }

  auto select_inst = [&](const std::vector<std::array<int, 3>> &candidates)
      -> std::optional<std::array<int, 3>> {
    for (const auto &inst : candidates) {
      if (warp_tile_m % inst[0] == 0 && warp_tile_n % inst[1] == 0 &&
          k_ % inst[2] == 0) {
        return inst;
      }
    }
    return std::nullopt;
  };

  if (*type_class == Ph1WmmaTypeClass::kF16F16F32 ||
      *type_class == Ph1WmmaTypeClass::kBF16BF16F32) {
    return select_inst(
        {{16, 16, 32}, {16, 16, 16}, {16, 8, 16}, {8, 16, 16}, {16, 8, 8}});
  }
  if (*type_class == Ph1WmmaTypeClass::kTF32TF32F32) {
    return select_inst({{16, 16, 16}, {16, 8, 8}, {16, 8, 4}});
  }
  if (*type_class == Ph1WmmaTypeClass::kS8S8S32 ||
      *type_class == Ph1WmmaTypeClass::kU8U8U32 ||
      *type_class == Ph1WmmaTypeClass::kFP8F32) {
    return select_inst(
        {{16, 16, 64}, {16, 16, 32}, {16, 16, 16}, {16, 8, 16}, {8, 16, 16}});
  }
  return select_inst({{16, 16, 32}, {16, 16, 16}, {16, 8, 16}, {8, 16, 16}});
}

bool GemmPyNode::AllowPH1Wmma(int block_size, Target target) const {
  tvm::transform::PassContext ctxt = tvm::transform::PassContext::Current();
  if (ctxt->GetConfig(kDisablePH1WMMA, Optional<Bool>()).value_or(false)) {
    return false;
  }
  return SelectPH1WmmaInstShape(block_size, target).has_value();
}

/**
 * @brief Checks whether WGMMA (warp-group MMA) can be used for this GEMM.
 *
 * Evaluates device-memory placement, data-type combinations, transpose flags,
 * and K divisibility constraints required for the Hopper WGMMA code path.
 *
 * The check returns true only when:
 * - B resides in shared memory ("shared" or "shared.dyn"); and
 * - (C, A, B) dtypes match one of the supported combinations below and K
 *   satisfies the required alignment; and
 * - for combinations that require specific orientations, A is not transposed
 *   and B is transposed.
 *
 * Supported combinations and constraints:
 * - C=float16:
 *   - A=float16, B=float16: K % 16 == 0
 *   - Various float8 mixes (e4m3/e5m2): require (!trans_A && trans_B) and K %
 * 32 == 0
 * - C=float32:
 *   - A=float16, B=float16: K % 16 == 0
 *   - A=bfloat16, B=bfloat16: K % 16 == 0
 *   - A=float32, B=float32: require (!trans_A && trans_B) and K % 8 == 0
 *   - Various float8 mixes: require (!trans_A && trans_B) and K % 32 == 0
 * - C=int32:
 *   - 8-bit integer combinations (Int8/UInt8): require (!trans_A && trans_B)
 * and K % 32 == 0
 *
 * @return true if WGMMA is supported for the current buffers, dtypes, and
 *         transpose/shape constraints; false otherwise.
 */
bool GemmPyNode::checkWgmma() const {
  if (b_.scope() != "shared.dyn" && b_.scope() != "shared") {
    return false;
  }

  if (c_->dtype == DataType::Float(16)) {
    if (a_->dtype == DataType::Float(16) && b_->dtype == DataType::Float(16))
      return k_ % 16 == 0;
    else if (a_->dtype.is_float8() && b_->dtype.is_float8())
      return (!transA_) && transB_ && k_ % 32 == 0;
    else
      return false;
  } else if (c_->dtype == DataType::Float(32)) {
    if (a_->dtype == DataType::Float(16) && b_->dtype == DataType::Float(16))
      return k_ % 16 == 0;
    else if (a_->dtype == DataType::BFloat(16) &&
             b_->dtype == DataType::BFloat(16))
      return k_ % 16 == 0;
    else if (a_->dtype == DataType::Float(32) &&
             b_->dtype == DataType::Float(32))
      return (!transA_) && transB_ && k_ % 8 == 0;
    else if (a_->dtype.is_float8() && b_->dtype.is_float8())
      return (!transA_) && transB_ && k_ % 32 == 0;
    else
      return false;
  } else if (c_->dtype == DataType::Int(32)) {
    if (a_->dtype == DataType::Int(8) && b_->dtype == DataType::Int(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else if (a_->dtype == DataType::Int(8) && b_->dtype == DataType::UInt(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else if (a_->dtype == DataType::UInt(8) && b_->dtype == DataType::Int(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else if (a_->dtype == DataType::UInt(8) && b_->dtype == DataType::UInt(8))
      return (!transA_) && transB_ && k_ % 32 == 0;
    else
      return false;
  } else {
    return false;
  }
}

Stmt GemmPyNode::Lower(const LowerArgs &T, arith::Analyzer *analyzer) const {
  (void)analyzer;
  if (const auto f = ffi::Function::GetGlobal("tl.gemm_py.lower")) {
    PrimExpr mbar_phase = T.mbar_phase_expr;
    if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(annotations_)) {
      mbar_phase = explicit_phase.value();
    }
    // NOTE(wt): Decide instruction key and compute warp partition on Python
    // side.
    auto prim_func = Downcast<PrimFunc>(
        (*f)(tvm::ffi::GetRef<GemmPy>(this), T.layout_map, T.target,
             T.thread_bounds, T.thread_var, mbar_phase));
    ICHECK(prim_func->attrs.defined());
    auto global_symbol =
        prim_func->attrs.GetAttr<tvm::ffi::String>("global_symbol");
    ICHECK(global_symbol.has_value());
    if (prim_func->body.as<BlockRealizeNode>()) {
      BlockRealize block_realize = Downcast<BlockRealize>(prim_func->body);
      auto block = block_realize->block;
      {
        BlockNode *n = block.CopyOnWrite();
        n->name_hint = global_symbol.value();
        n->annotations.Set(tl::attr::kLexicalAllocScope,
                           IntImm(DataType::Int(32), 1));
      }
      return BlockRealize(block_realize->iter_values, block_realize->predicate,
                          block);
    }
    // warp with block realize node
    Map<String, ObjectRef> block_annotations;
    block_annotations.Set(tl::attr::kLexicalAllocScope,
                          IntImm(DataType::Int(32), 1));
    return BlockRealize(
        /*iter_values=*/Array<PrimExpr>(),
        /*predicate=*/const_true(),
        /*block=*/
        Block(/*iter_vars=*/{}, /*reads=*/{}, /*writes=*/{},
              /*name_hint=*/global_symbol.value(), prim_func->body,
              /*init=*/Optional<Stmt>(), /*alloc_buffers=*/{},
              /*match_buffers=*/{}, /*annotations=*/block_annotations));
  } else {
    LOG(FATAL) << "No lower function found for gemm_py";
    return Stmt(); // This line will never be reached due to LOG(FATAL), but
                   // satisfies compiler
  }
}

LayoutMap GemmPyNode::InferLayout(const LayoutInferArgs &T,
                                  InferLevel level) const {
  if (completed_)
    return {};
  LayoutMap results;

  if (const auto f = ffi::Function::GetGlobal("tl.gemm_py.infer_layout")) {
    results = Downcast<LayoutMap>(
        (*f)(tvm::ffi::GetRef<GemmPy>(this), T.target, T.thread_bounds));
    // Bind all fragment layouts with the provided thread range
    for (auto kv : results) {
      const Buffer &buf = kv.first;
      const Layout &layout = kv.second;
      if (auto frag = layout.as<Fragment>()) {
        results.Set(buf, frag.value()->BindThreadRange(T.thread_bounds));
      }
    }
  } else {
    LOG(FATAL) << "No infer layout function found for gemm_py";
  }

  completed_ = true;
  return results;
}

TIR_REGISTER_TL_TILE_OP(GemmPy, gemm_py)
    .set_num_inputs(5)
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

TVM_FFI_STATIC_INIT_BLOCK() { GemmPyNode::RegisterReflection(); }

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.GemmPyGemmInstructionKey",
                        [](GemmPy gemm_py, int block_size, Target target) {
                          return gemm_py->getGemmInstructionKey(block_size,
                                                                target);
                        });
  refl::GlobalDef().def("tl.GemmPyAllowSQMMA",
                        [](GemmPy gemm_py, int block_size, Target target) {
                          return gemm_py->AllowSQMMA(block_size, target);
                        });
  refl::GlobalDef().def("tl.GemmPyAllowPH1Wmma",
                        [](GemmPy gemm_py, int block_size, Target target) {
                          return gemm_py->AllowPH1Wmma(block_size, target);
                        });
  refl::GlobalDef().def("tl.GemmPySelectSQMMAInstShape",
                        [](GemmPy gemm_py, int block_size, Target target) {
                          Array<Integer> result;
                          auto inst_shape =
                              gemm_py->SelectSQMMAInstShape(block_size, target);
                          if (inst_shape.has_value()) {
                            result.push_back(Integer((*inst_shape)[0]));
                            result.push_back(Integer((*inst_shape)[1]));
                            result.push_back(Integer((*inst_shape)[2]));
                          }
                          return result;
                        });
  refl::GlobalDef().def("tl.GemmPySelectPH1WmmaInstShape",
                        [](GemmPy gemm_py, int block_size, Target target) {
                          Array<Integer> result;
                          auto inst_shape = gemm_py->SelectPH1WmmaInstShape(
                              block_size, target);
                          if (inst_shape.has_value()) {
                            result.push_back(Integer((*inst_shape)[0]));
                            result.push_back(Integer((*inst_shape)[1]));
                            result.push_back(Integer((*inst_shape)[2]));
                          }
                          return result;
                        });
}

} // namespace tl
} // namespace tvm
