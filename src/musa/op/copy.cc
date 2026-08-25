/*!
 * \file tl/musa/op/copy.cc
 * \brief MUSA implementation for tl.copy lowering.
 */

#include "musa/op/copy.h"

#include "backend/common/target_utils.h"
#include "cuda/transform/ptx_async_copy_injector.h"
#include "musa/target_utils.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;

namespace musa {

namespace {

bool IsExplicitAsyncCopy(const CopyNode &op) {
  if (auto value = op.annotations.Get("is_async_copy")) {
    if (const auto *flag = value.value().as<IntImmNode>()) {
      return flag->value != 0;
    }
  }
  return false;
}

bool IsExplicitTmaCopy(const CopyNode &op) {
  if (auto value = op.annotations.Get("is_tma_copy")) {
    if (const auto *flag = value->as<IntImmNode>()) {
      return flag->value != 0;
    }
  }
  return false;
}

// Values are part of the stable MUSA descriptor ABI. Including musa.h in
// this host compiler translation unit conflicts with TileLang's CUDA stubs;
// the MP31 device template includes the authoritative toolkit header instead.
constexpr int kDescInt8 = 0;
constexpr int kDescUInt8 = 1;
constexpr int kDescInt16 = 2;
constexpr int kDescUInt16 = 3;
constexpr int kDescFloat16 = 4;
constexpr int kDescBFloat16 = 5;
constexpr int kDescInt32 = 6;
constexpr int kDescUInt32 = 7;
constexpr int kDescFloat32 = 8;
constexpr int kDescInt64 = 10;
constexpr int kDescUInt64 = 11;
constexpr int kDescFloat64 = 12;

PrimExpr GetTmaBarrier(const CopyNode &op) {
  auto value = op.annotations.Get("barrier");
  ICHECK(value) << "T.tma_copy() load requires a barrier argument on MP31";
  auto barrier = value->as<PrimExprNode>();
  ICHECK(barrier) << "T.tma_copy() barrier annotation must be a PrimExpr";
  return GetRef<PrimExpr>(barrier);
}

int GetTmaDescriptorDataType(DataType dtype) {
  if (dtype.is_bfloat16()) {
    return kDescBFloat16;
  }
  if (dtype.is_float()) {
    switch (dtype.bits()) {
    case 16:
      return kDescFloat16;
    case 32:
      return kDescFloat32;
    case 64:
      return kDescFloat64;
    default:
      break;
    }
  }
  if (dtype.is_int()) {
    switch (dtype.bits()) {
    case 8:
      return kDescInt8;
    case 16:
      return kDescInt16;
    case 32:
      return kDescInt32;
    case 64:
      return kDescInt64;
    default:
      break;
    }
  }
  if (dtype.is_uint()) {
    switch (dtype.bits()) {
    case 8:
      return kDescUInt8;
    case 16:
      return kDescUInt16;
    case 32:
      return kDescUInt32;
    case 64:
      return kDescUInt64;
    default:
      break;
    }
  }
  LOG(FATAL) << "MP31 TME load does not support descriptor dtype " << dtype;
  return 0;
}

Array<PrimExpr> Reverse(const Array<PrimExpr> &values) {
  Array<PrimExpr> result;
  result.reserve(values.size());
  for (auto it = values.rbegin(); it != values.rend(); ++it) {
    result.push_back(*it);
  }
  return result;
}

Array<PrimExpr> ReverseRanges(const Array<Range> &ranges, bool extent) {
  Array<PrimExpr> values;
  values.reserve(ranges.size());
  for (auto it = ranges.rbegin(); it != ranges.rend(); ++it) {
    values.push_back(extent ? (*it)->extent : (*it)->min);
  }
  return values;
}

Stmt LowerTmaLoad(const CopyNode &op, const LowerArgs &args,
                  arith::Analyzer *analyzer) {
  ICHECK(TargetIsMP31(args.target))
      << "MP31 TME load requires an MP31 MUSA target, got " << args.target;
  ICHECK(IsExplicitTmaCopy(op))
      << "MUSA MP31 TME lowering requires T.tma_copy";
  ICHECK(IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst))
      << "MP31 TME load only supports global-to-shared copies, got src="
      << op.src.scope() << ", dst=" << op.dst.scope();
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "MP31 TME load requires matching source and destination dtypes";

  const size_t rank = op.src_range.size();
  ICHECK_GE(rank, 1U);
  ICHECK_LE(rank, 5U);
  ICHECK_EQ(op.dst_range.size(), rank);

  // TME descriptors are expressed in bytes and use the innermost dimension
  // first. Restrict the first vertical slice to a row-major contiguous tile;
  // swizzle and split-box support will be added in later MP31 commits.
  Array<PrimExpr> global_shape = Reverse(op.src->shape);
  Array<PrimExpr> global_coords = ReverseRanges(op.src_range, false);
  Array<PrimExpr> box_dims = ReverseRanges(op.dst_range, true);
  Array<PrimExpr> global_stride;
  if (!op.src->strides.empty()) {
    global_stride = Reverse(op.src->strides);
  } else {
    PrimExpr stride = 1;
    Array<PrimExpr> row_major;
    for (auto shape : global_shape) {
      row_major.push_back(stride);
      stride *= shape;
    }
    global_stride = row_major;
  }
  ICHECK_EQ(global_stride.size(), rank);
  ICHECK(is_one(global_stride[0]))
      << "MP31 TME load requires a contiguous innermost global dimension";

  Array<PrimExpr> global_stride_bytes;
  for (size_t i = 0; i < rank; ++i) {
    PrimExpr stride =
        analyzer->Simplify(global_stride[i] * op.src->dtype.bytes());
    if (i != 0) {
      if (auto imm = stride.as<IntImmNode>()) {
        ICHECK_EQ(imm->value % 16, 0)
            << "MP31 TME global stride must be 16-byte aligned, got "
            << stride;
      }
    }
    global_stride_bytes.push_back(cast(DataType::UInt(64), stride));
  }

  for (size_t i = 0; i < rank; ++i) {
    ICHECK(analyzer->CanProveEqual(op.src_range[rank - i - 1]->extent,
                                   op.dst_range[rank - i - 1]->extent))
        << "MP31 TME load requires matching global/shared tile extents";
  }

  Array<PrimExpr> smem_stride;
  for (size_t i = 0; i < rank; ++i) {
    smem_stride.push_back(1);
  }
  TMEDesc desc{
      rank,
      GetTmaDescriptorDataType(op.src->dtype),
      global_shape,
      global_stride_bytes,
      box_dims,
      smem_stride,
      op.src->data,
      /*interleave=*/0,
      /*swizzle=*/0,
      /*l2_promotion=*/0,
      /*oob_fill=*/0,
  };
  PrimExpr descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());

  PrimExpr shared_offset = 0;
  PrimExpr stride = 1;
  for (auto it = op.dst_range.rbegin(); it != op.dst_range.rend(); ++it) {
    shared_offset += (*it)->min * stride;
    stride *= (*it)->extent;
  }
  PrimExpr shared_ptr = op.dst.access_ptr(
      /*rw_mask=*/2, DataType::Handle(), 1, shared_offset, stride);

  Array<PrimExpr> tma_args;
  tma_args.push_back(descriptor);
  tma_args.push_back(GetTmaBarrier(op));
  tma_args.push_back(shared_ptr);
  tma_args.insert(tma_args.end(), global_coords.begin(), global_coords.end());
  tma_args.insert(tma_args.end(), box_dims.begin(), box_dims.end());
  Stmt load = Evaluate(Call(DataType::Handle(), tma_load(), tma_args));

  PrimExpr bytes = 1;
  for (auto dim : box_dims) {
    bytes *= dim;
  }
  bytes = analyzer->Simplify(bytes * op.dst->dtype.bytes());
  Stmt expect = Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                              {GetTmaBarrier(op), bytes}));
  load = SeqStmt({expect, load});
  return IfThenElse(EQ(args.thread_index, args.thread_bounds->min), load);
}

Stmt LowerAsyncCopy(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
  ICHECK(TargetMUSAHasAsyncCopy(lower_args.target))
      << "T.async_copy requires a MUSA target with async-copy support "
         "(mp_21 or newer). Got target="
      << lower_args.target;
  ICHECK(IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst))
      << "T.async_copy only supports global-to-shared copies. Got src scope="
      << op.src.scope() << ", dst scope=" << op.dst.scope();
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "T.async_copy requires matching source and destination dtypes.";

  auto simt_loop = op.MakeSIMTLoop(analyzer);
  auto fused_loop = Downcast<For>(ParallelLoopFuser::Fuse(simt_loop));
  auto par_op = ParallelOp(fused_loop);
  for (auto level :
       {InferLevel::kCommon, InferLevel::kStrict, InferLevel::kFree}) {
    par_op->InferLayout({lower_args.target,
                         lower_args.thread_bounds,
                         lower_args.layout_map,
                         analyzer,
                         lower_args.buffer_remap,
                         {}},
                        level);
  }

  auto loop_layout = par_op->GetLoopLayout();
  Stmt lowered_loop = LowerParallelLoop(
      par_op->GetRoot(), loop_layout, lower_args.thread_index, analyzer,
      lower_args.layout_map, par_op->GetPredicate(lower_args.thread_index),
      /*parallel_loop=*/true, /*should_vectorize=*/true,
      par_op->LoopLayoutRequiresPaddingGuard());

  auto injected = InjectPTXAsyncCopy(lowered_loop,
                                     /*async_without_async_commit_wait=*/true);
  ICHECK(injected.injected_ptx_async_copy)
      << "T.async_copy requires an eligible global-to-shared vectorized copy.";

  Stmt commit_group =
      Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
  return SeqStmt({injected.stmt, commit_group});
}

} // namespace

const char *CopyInstToString(CopyInst inst) {
  switch (inst) {
  case CopyInst::kNormal:
    return "normal";
  case CopyInst::kAsync:
    return "async";
  case CopyInst::kTMELoad:
    return "tme_load";
  case CopyInst::kInvalid:
    return "invalid";
  }
  return "unknown";
}

bool CopyInstIsTME(CopyInst inst) { return inst == CopyInst::kTMELoad; }

bool CopyInstIsAsync(CopyInst inst) { return inst == CopyInst::kAsync; }

CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx) {
  if (IsExplicitTmaCopy(op)) {
    if (!TargetIsMP31(ctx.target)) {
      return {CopyInst::kInvalid, false,
              "T.tma_copy() is currently supported only on MP31 MUSA "
              "targets"};
    }
    return {CopyInst::kTMELoad, true, {}};
  }
  if (IsExplicitAsyncCopy(op)) {
    return {CopyInst::kAsync, true, {}};
  }
  return {CopyInst::kNormal, true, {}};
}

struct Copy {
  static LayoutMap InferLayout(const CopyNode &op,
                               const LayoutInferArgs &layout_args,
                               InferLevel level) {
    CopyAnalysisContext ctx{layout_args.target, &layout_args.layout_map,
                            layout_args.analyzer, true};
    CopyInstSelection selection = SelectCopyInstForLowering(op, ctx);
    ICHECK(selection.supported) << selection.reason;
    return op.InferSIMTLayout(layout_args, level);
  }

  static Stmt Lower(const CopyNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    CopyAnalysisContext ctx{lower_args.target, &lower_args.layout_map, analyzer,
                            true};
    CopyInstSelection selection = SelectCopyInstForLowering(op, ctx);
    ICHECK(selection.supported) << selection.reason;
    if (CopyInstIsTME(selection.inst)) {
      return LowerTmaLoad(op, lower_args, analyzer);
    }
    if (CopyInstIsAsync(selection.inst)) {
      return LowerAsyncCopy(op, lower_args, analyzer);
    }
    return LowerNormalCopy(op, lower_args, analyzer);
  }
};

} // namespace musa

namespace {

bool MatchMUSACopyTarget(Target target) { return TargetIsMUSA(target); }

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
