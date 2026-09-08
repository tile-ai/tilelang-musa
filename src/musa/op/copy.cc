/*!
 * \file tl/musa/op/copy.cc
 * \brief MUSA implementation for tl.copy lowering.
 */

#include "musa/op/copy.h"

#include "backend/common/target_utils.h"
#include "musa/layout/swizzle_layout.h"
#include "musa/op/builtin.h"
#include "musa/target_utils.h"
#include "musa/transform/async_copy_injector.h"
#include "op/builtin.h"
#include "op/utils.h"
#include "transform/common/loop_fusion_utils.h"
#include "transform/loop_partition.h"

#include "support/check.h"

#include <algorithm>
#include <optional>
#include <vector>

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

bool HasPipelineManagedAsyncAnnotation(const CopyNode &op) {
  if (auto value = op.annotations.Get(attr::kAsyncCopyNoImplicitCommitWait)) {
    if (const auto *flag = value.value().as<IntImmNode>()) {
      return flag->value != 0;
    }
  }
  return false;
}

std::string GetCopyPreference(const CopyNode &op) {
  auto value = op.annotations.Get("prefer_instruction");
  if (!value) {
    return "";
  }
  const auto *prefer = value->as<StringImmNode>();
  ICHECK(prefer) << "T.copy prefer_instruction annotation must be a string";
  return prefer->value;
}

bool PreferAsyncCopy(const CopyNode &op) {
  return GetCopyPreference(op) == "cp_async";
}

bool PreferTmaCopy(const CopyNode &op) {
  return GetCopyPreference(op) == "tma";
}

bool PreferSyncCopy(const CopyNode &op) {
  return GetCopyPreference(op) == "sync";
}

bool DisableTmaCopy(const CopyNode &op) {
  if (auto value = op.annotations.Get("disable_tma")) {
    const auto *flag = value->as<IntImmNode>();
    ICHECK(flag) << "T.copy disable_tma annotation must be a boolean";
    return flag->value != 0;
  }
  return false;
}

bool CanLowerTmeTensorCopy(const Buffer &global_buffer,
                           const Array<Range> &global_range,
                           const Array<Range> &shared_range,
                           arith::Analyzer *analyzer);

bool CanWarpSpecializeTmeLoad(const CopyNode &op, Target target) {
  if (!TargetIsMP31(target) || !IsGlobalBuffer(op.src) ||
      !IsSharedBuffer(op.dst) || op.src->dtype != op.dst->dtype ||
      DisableTmaCopy(op) || PreferSyncCopy(op)) {
    return false;
  }
  Array<Range> versioned_shared_range;
  versioned_shared_range.push_back(Range::FromMinExtent(0, 1));
  for (const Range &range : op.dst_range) {
    versioned_shared_range.push_back(range);
  }
  arith::Analyzer analyzer;
  return CanLowerTmeTensorCopy(op.src, op.src_range, versioned_shared_range,
                               &analyzer);
}

bool IsExplicitTmaCopy(const CopyNode &op) {
  if (auto value = op.annotations.Get("is_tma_copy")) {
    if (const auto *flag = value->as<IntImmNode>()) {
      return flag->value != 0;
    }
  }
  return false;
}

bool GetBoolAnnotation(const CopyNode &op, const char *key) {
  if (auto value = op.annotations.Get(key)) {
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

SwizzleLayout GetSwizzleParams(const Buffer &shared_buffer,
                               const LowerArgs &args) {
  const SwizzleLayout no_swizzle{SwizzleGranularity::kNone,
                                 SwizzleStride::k256B, SwizzleLine::k256B};
  if (shared_buffer->shape.size() < 2) {
    return no_swizzle;
  }
  auto layout_it = args.layout_map.find(shared_buffer);
  if (layout_it == args.layout_map.end()) {
    return no_swizzle;
  }

  return AnalyzeSwizzleLayout(shared_buffer, (*layout_it).second);
}

constexpr int kSQMMAMaxLeadingStrideBytes = 256;

std::optional<int> GetPanelizedCols(const Buffer &shared_buffer,
                                    const LowerArgs &args) {
  if (!IsSharedBuffer(shared_buffer) || shared_buffer->shape.size() < 2) {
    return std::nullopt;
  }
  const SwizzleLayout swizzle = GetSwizzleParams(shared_buffer, args);
  if (swizzle.swizzle_granularity == SwizzleGranularity::kNone) {
    return std::nullopt;
  }
  const int64_t *cols = as_const_int(shared_buffer->shape.back());
  ICHECK(cols != nullptr)
      << "MP31 SQMMA panelized layout requires a constant continuous extent";
  const int element_bytes = shared_buffer->dtype.bytes();
  ICHECK_GT(element_bytes, 0);
  if (*cols * element_bytes <= kSQMMAMaxLeadingStrideBytes) {
    return std::nullopt;
  }
  const int panel_cols = kSQMMAMaxLeadingStrideBytes / element_bytes;
  ICHECK_GT(panel_cols, 0);
  ICHECK_EQ(*cols % panel_cols, 0)
      << "MP31 SQMMA continuous extent must be divisible by its 256-byte "
         "panel width, got cols="
      << *cols << ", panel_cols=" << panel_cols;
  return panel_cols;
}

std::vector<ObjectPtr<CopyNode>> SplitPanelizedCopy(const CopyNode &op,
                                                    const LowerArgs &args,
                                                    arith::Analyzer *analyzer) {
  const bool shared_is_dst = IsSharedBuffer(op.dst);
  const bool shared_is_src = IsSharedBuffer(op.src);
  if (shared_is_dst == shared_is_src) {
    return {};
  }
  const Buffer &shared_buffer = shared_is_dst ? op.dst : op.src;
  const auto panel_cols = GetPanelizedCols(shared_buffer, args);
  if (!panel_cols.has_value()) {
    return {};
  }

  const Array<Range> &shared_range =
      shared_is_dst ? op.dst_range : op.src_range;
  const Array<Range> &other_range = shared_is_dst ? op.src_range : op.dst_range;
  ICHECK(!shared_range.empty() && !other_range.empty());
  const int64_t *start = as_const_int(shared_range.back()->min);
  const int64_t *extent = as_const_int(shared_range.back()->extent);
  ICHECK(start != nullptr && extent != nullptr)
      << "MP31 SQMMA panel split requires a constant shared-memory "
         "continuous range";
  ICHECK(analyzer->CanProveEqual(shared_range.back()->extent,
                                 other_range.back()->extent))
      << "MP31 SQMMA panel split requires matching source and destination "
         "continuous extents";

  std::vector<ObjectPtr<CopyNode>> splits;
  int64_t consumed = 0;
  while (consumed < *extent) {
    const int64_t col = *start + consumed;
    const int64_t chunk =
        std::min<int64_t>(*panel_cols - col % *panel_cols, *extent - consumed);
    ObjectPtr<CopyNode> split = make_object<CopyNode>(op);
    Array<Range> src_range = split->src_range;
    Array<Range> dst_range = split->dst_range;
    const PrimExpr delta = Integer(consumed);
    src_range.Set(
        src_range.size() - 1,
        Range::FromMinExtent(src_range.back()->min + delta, Integer(chunk)));
    dst_range.Set(
        dst_range.size() - 1,
        Range::FromMinExtent(dst_range.back()->min + delta, Integer(chunk)));
    split->src_range = src_range;
    split->dst_range = dst_range;
    splits.push_back(std::move(split));
    consumed += chunk;
  }
  return splits;
}

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
  // MUSA TME descriptors have no native FP8 data type. FP8 values are
  // byte-preserving for TME, so describe them as UINT8 in the descriptor.
  if (dtype.is_float8()) {
    return kDescUInt8;
  }
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
  LOG(FATAL) << "MP31 TME does not support descriptor dtype " << dtype;
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

bool AlignTmeSharedRangeToGlobal(const Array<Range> &global_range,
                                 const Array<Range> &shared_range,
                                 arith::Analyzer *analyzer,
                                 Array<Range> *aligned_shared_range) {
  aligned_shared_range->clear();
  aligned_shared_range->reserve(global_range.size());

  size_t global_dim = 0;
  size_t shared_dim = 0;
  while (global_dim < global_range.size() &&
         shared_dim < shared_range.size()) {
    const Range &global = global_range[global_dim];
    const Range &shared = shared_range[shared_dim];
    if (analyzer->CanProveEqual(shared->extent, Integer(1))) {
      ++shared_dim;
      continue;
    }
    if (analyzer->CanProveEqual(global->extent, Integer(1))) {
      aligned_shared_range->push_back(Range::FromMinExtent(0, 1));
      ++global_dim;
      continue;
    }
    if (!analyzer->CanProveEqual(global->extent, shared->extent)) {
      return false;
    }
    aligned_shared_range->push_back(shared);
    ++global_dim;
    ++shared_dim;
  }

  while (shared_dim < shared_range.size()) {
    if (!analyzer->CanProveEqual(shared_range[shared_dim]->extent,
                                 Integer(1))) {
      return false;
    }
    ++shared_dim;
  }
  while (global_dim < global_range.size()) {
    if (!analyzer->CanProveEqual(global_range[global_dim]->extent,
                                 Integer(1))) {
      return false;
    }
    aligned_shared_range->push_back(Range::FromMinExtent(0, 1));
    ++global_dim;
  }
  return aligned_shared_range->size() == global_range.size();
}

struct LoweredTMEDesc {
  PrimExpr descriptor;
  Array<PrimExpr> global_coords;
  Array<PrimExpr> box_dims;
};

bool TmaDescriptorSupportsDataType(DataType dtype) {
  if (dtype.is_bfloat16()) {
    return true;
  }
  if (dtype.is_float()) {
    return dtype.bits() == 16 || dtype.bits() == 32 || dtype.bits() == 64;
  }
  if (dtype.is_int() || dtype.is_uint()) {
    return dtype.bits() == 8 || dtype.bits() == 16 || dtype.bits() == 32 ||
           dtype.bits() == 64;
  }
  return false;
}

std::optional<Array<PrimExpr>>
GetTmaGlobalStrideBytes(const Buffer &global_buffer,
                        arith::Analyzer *analyzer) {
  const size_t rank = global_buffer->shape.size();
  Array<PrimExpr> global_shape = Reverse(global_buffer->shape);
  Array<PrimExpr> global_stride;
  if (!global_buffer->strides.empty()) {
    global_stride = Reverse(global_buffer->strides);
  } else {
    PrimExpr stride = Integer(1);
    for (const PrimExpr &shape : global_shape) {
      global_stride.push_back(stride);
      stride = analyzer->Simplify(stride * shape);
    }
  }
  if (global_stride.size() != rank ||
      !analyzer->CanProveEqual(global_stride[0], Integer(1))) {
    return std::nullopt;
  }

  Array<PrimExpr> result;
  result.reserve(rank);
  for (size_t i = 0; i < rank; ++i) {
    PrimExpr stride =
        analyzer->Simplify(global_stride[i] * global_buffer->dtype.bytes());
    if (i != 0) {
      if (const auto *imm = stride.as<IntImmNode>()) {
        if (imm->value % 16 != 0) {
          return std::nullopt;
        }
      }
    }
    result.push_back(cast(DataType::UInt(64), stride));
  }
  return result;
}

bool CanLowerTmeTensorCopy(const Buffer &global_buffer,
                           const Array<Range> &global_range,
                           const Array<Range> &shared_range,
                           arith::Analyzer *analyzer) {
  const size_t rank = global_range.size();
  if (rank < 1 || rank > 5 || global_buffer->shape.size() != rank ||
      !TmaDescriptorSupportsDataType(global_buffer->dtype) ||
      !GetTmaGlobalStrideBytes(global_buffer, analyzer).has_value()) {
    return false;
  }

  Array<Range> aligned_shared_range;
  return AlignTmeSharedRangeToGlobal(global_range, shared_range, analyzer,
                                     &aligned_shared_range);
}

bool CanAutoSelectTmeStore(const CopyNode &op, const CopyAnalysisContext &ctx) {
  return TargetIsMP31(ctx.target) && IsSharedBuffer(op.src) &&
         IsGlobalBuffer(op.dst) && op.src->dtype == op.dst->dtype &&
         CanLowerTmeTensorCopy(op.dst, op.dst_range, op.src_range,
                               ctx.analyzer);
}

PrimExpr MakeIm2ColTmaDescriptor(const Buffer &global_buffer,
                                 const Array<PrimExpr> &box_dims,
                                 const Array<PrimExpr> &global_stride_bytes) {
  size_t rank = global_buffer->shape.size();
  ICHECK_EQ(rank, 4) << "MP31 TME im2col requires an NHWC 4D input";
  Array<PrimExpr> global_shape = Reverse(global_buffer->shape);
  TMEDesc desc{/*rank=*/rank,
               /*data_type=*/GetTmaDescriptorDataType(global_buffer->dtype),
               /*global_shape=*/global_shape,
               /*global_stride=*/global_stride_bytes,
               /*smem_box=*/box_dims,
               /*smem_stride=*/Array<PrimExpr>(rank, Integer(1)),
               /*global_addr=*/global_buffer->data,
               /*interleave=*/0,
               /*swizzle=*/0,
               /*l2_promotion=*/0,
               /*oob_fill=*/0};
  return Call(DataType::Handle(), create_tma_descriptor(),
              desc.EncodeCallArgs());
}

LoweredTMEDesc MakeTmaDescriptor(const Buffer &global_buffer,
                                 const Array<Range> &global_range,
                                 const Array<Range> &shared_range,
                                 arith::Analyzer *analyzer) {
  const size_t rank = global_range.size();
  ICHECK_GE(rank, 1U);
  ICHECK_LE(rank, 5U);
  Array<Range> logical_shared_range;
  ICHECK(AlignTmeSharedRangeToGlobal(global_range, shared_range, analyzer,
                                     &logical_shared_range))
      << "MP31 TME requires matching non-unit global/shared tile extents; "
         "unmatched dimensions must have extent one";

  // TME descriptors are expressed in bytes and use the innermost dimension
  // first. Restrict the first vertical slice to a row-major contiguous tile.
  // MP31 shared-memory swizzle is selected by the device TME instruction and
  // therefore does not change this global tensor descriptor.
  Array<PrimExpr> global_shape = Reverse(global_buffer->shape);
  Array<PrimExpr> global_coords = ReverseRanges(global_range, false);
  // Keep the descriptor's true global shape while the TME box follows the
  // normalized shared tile. Coordinates plus box dimensions may cross the
  // descriptor boundary; MP31 then fills those load elements with zero.
  Array<PrimExpr> box_dims = ReverseRanges(logical_shared_range, true);
  Array<PrimExpr> global_stride;
  if (!global_buffer->strides.empty()) {
    global_stride = Reverse(global_buffer->strides);
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
        analyzer->Simplify(global_stride[i] * global_buffer->dtype.bytes());
    if (i != 0) {
      if (auto imm = stride.as<IntImmNode>()) {
        ICHECK_EQ(imm->value % 16, 0)
            << "MP31 TME global stride must be 16-byte aligned, got " << stride;
      }
    }
    global_stride_bytes.push_back(cast(DataType::UInt(64), stride));
  }

  Array<PrimExpr> smem_stride;
  for (size_t i = 0; i < rank; ++i) {
    smem_stride.push_back(1);
  }
  TMEDesc desc{
      rank,
      GetTmaDescriptorDataType(global_buffer->dtype),
      global_shape,
      global_stride_bytes,
      box_dims,
      smem_stride,
      global_buffer->data,
      /*interleave=*/0,
      /*swizzle=*/0,
      /*l2_promotion=*/0,
      /*oob_fill=*/0,
  };
  PrimExpr descriptor =
      Call(DataType::Handle(), create_tma_descriptor(), desc.EncodeCallArgs());
  return {descriptor, global_coords, box_dims};
}

PrimExpr MakeTmaSharedPtr(const Buffer &shared_buffer,
                          const Array<Range> &shared_range,
                          const LowerArgs &args, arith::Analyzer *analyzer,
                          int rw_mask) {
  if (const auto panel_cols = GetPanelizedCols(shared_buffer, args)) {
    ICHECK(shared_buffer->strides.empty())
        << "MP31 SQMMA panelized TME copy does not support explicit shared "
           "strides";
    const size_t rank = shared_buffer->shape.size();
    ICHECK_EQ(rank, shared_range.size());
    ICHECK_GE(rank, 2U);
    const int64_t *rows = as_const_int(shared_buffer->shape[rank - 2]);
    const int64_t *cols = as_const_int(shared_buffer->shape[rank - 1]);
    ICHECK(rows != nullptr && cols != nullptr);
    const PrimExpr rows_expr = Integer(*rows);
    const PrimExpr cols_expr = Integer(*cols);
    const PrimExpr panel_cols_expr = Integer(*panel_cols);
    ICHECK(analyzer->CanProveEqual(shared_range[rank - 2]->min, 0))
        << "MP31 SQMMA panelized TME copy must start at matrix row zero";
    ICHECK(analyzer->CanProveEqual(shared_range[rank - 2]->extent, rows_expr))
        << "MP31 SQMMA panelized TME copy must cover the complete matrix row "
           "dimension";
    ICHECK(analyzer->CanProveEqual(
        FloorMod(shared_range[rank - 1]->min, panel_cols_expr), 0))
        << "MP31 SQMMA panelized TME copy must start at a 256-byte panel "
           "boundary";
    ICHECK(analyzer->CanProveEqual(
        FloorMod(shared_range[rank - 1]->extent, panel_cols_expr), 0))
        << "MP31 SQMMA panelized TME copy must cover complete 256-byte panels";

    PrimExpr leading_index = 0;
    for (size_t i = 0; i + 2 < rank; ++i) {
      leading_index =
          leading_index * shared_buffer->shape[i] + shared_range[i]->min;
    }
    const PrimExpr row = shared_range[rank - 2]->min;
    const PrimExpr col = shared_range[rank - 1]->min;
    const PrimExpr panel = FloorDiv(col, panel_cols_expr);
    const PrimExpr in_panel = FloorMod(col, panel_cols_expr);
    PrimExpr shared_offset = leading_index * rows_expr * cols_expr +
                             panel * rows_expr * panel_cols_expr +
                             row * panel_cols_expr + in_panel;
    PrimExpr shared_elements = 1;
    for (const Range &range : shared_range) {
      shared_elements *= range->extent;
    }
    return shared_buffer.access_ptr(rw_mask, DataType::Handle(), 1,
                                    analyzer->Simplify(shared_offset),
                                    shared_elements);
  }

  std::vector<PrimExpr> shared_strides;
  if (!shared_buffer->strides.empty()) {
    shared_strides.assign(shared_buffer->strides.begin(),
                          shared_buffer->strides.end());
  } else {
    PrimExpr stride = 1;
    for (auto it = shared_buffer->shape.rbegin();
         it != shared_buffer->shape.rend(); ++it) {
      shared_strides.insert(shared_strides.begin(), stride);
      stride *= *it;
    }
  }
  ICHECK_EQ(shared_strides.size(), shared_range.size());
  PrimExpr shared_offset = 0;
  PrimExpr shared_elements = 1;
  for (size_t i = 0; i < shared_range.size(); ++i) {
    shared_offset += shared_range[i]->min * shared_strides[i];
    shared_elements *= shared_range[i]->extent;
  }
  return shared_buffer.access_ptr(rw_mask, DataType::Handle(), 1, shared_offset,
                                  shared_elements);
}

struct ContiguousTmaRegion {
  PrimExpr offset;
  PrimExpr elements;
};

ContiguousTmaRegion AnalyzeContiguousTmaRegion(const Buffer &buffer,
                                               const Array<Range> &ranges,
                                               arith::Analyzer *analyzer,
                                               const char *label) {
  const size_t rank = buffer->shape.size();
  ICHECK_EQ(ranges.size(), rank) << "Runtime-pointer TME " << label
                                 << " region rank must match its buffer rank";

  int first_non_unit = -1;
  PrimExpr elements = 1;
  Array<PrimExpr> indices;
  for (size_t i = 0; i < rank; ++i) {
    elements *= ranges[i]->extent;
    indices.push_back(ranges[i]->min);
    if (first_non_unit < 0 && !analyzer->CanProveEqual(ranges[i]->extent, 1)) {
      first_non_unit = static_cast<int>(i);
    }
  }

  if (first_non_unit >= 0) {
    for (size_t i = static_cast<size_t>(first_non_unit + 1); i < rank; ++i) {
      ICHECK(analyzer->CanProveEqual(ranges[i]->extent, buffer->shape[i]))
          << "Runtime-pointer TME " << label
          << " region must be contiguous; trailing dimension " << i
          << " is not fully covered";
    }

    if (!buffer->strides.empty()) {
      PrimExpr expected_stride = 1;
      for (size_t i = rank; i-- > static_cast<size_t>(first_non_unit);) {
        ICHECK(analyzer->CanProveEqual(buffer->strides[i], expected_stride))
            << "Runtime-pointer TME " << label
            << " region requires compact trailing strides";
        expected_stride *= buffer->shape[i];
      }
    }
  }

  Array<PrimExpr> physical_indices = buffer.OffsetOf(indices);
  ICHECK(!physical_indices.empty())
      << "Runtime-pointer TME " << label << " region has no physical offset";
  return {analyzer->Simplify(physical_indices.back()),
          analyzer->Simplify(elements)};
}

Stmt LowerTmaLoad(const CopyNode &op, const LowerArgs &args,
                  arith::Analyzer *analyzer) {
  ICHECK(TargetIsMP31(args.target))
      << "MP31 TME load requires an MP31 MUSA target, got " << args.target;
  ICHECK(IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst))
      << "MP31 TME load only supports global-to-shared copies, got src="
      << op.src.scope() << ", dst=" << op.dst.scope();
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "MP31 TME load requires matching source and destination dtypes";

  if (GetBoolAnnotation(op, "musa_runtime_pointer_tme")) {
    ICHECK(IsExplicitTmaCopy(op))
        << "Runtime-pointer TME loads require explicit T.tma_copy";
    ICHECK_EQ(GetSwizzleParams(op.dst, args).swizzle_granularity,
              SwizzleGranularity::kNone)
        << "Runtime-pointer TME loads do not support swizzled shared layouts";
    ICHECK(!GetPanelizedCols(op.dst, args).has_value())
        << "Runtime-pointer TME loads do not support panelized shared layouts";
    ContiguousTmaRegion global_region =
        AnalyzeContiguousTmaRegion(op.src, op.src_range, analyzer, "source");
    ContiguousTmaRegion shared_region = AnalyzeContiguousTmaRegion(
        op.dst, op.dst_range, analyzer, "destination");
    ICHECK(
        analyzer->CanProveEqual(global_region.elements, shared_region.elements))
        << "Runtime-pointer TME requires matching source/shared extents";
    PrimExpr bytes =
        analyzer->Simplify(global_region.elements * op.src->dtype.bytes());
    PrimExpr global_ptr = op.src.access_ptr(
        /*access_mask=*/1, DataType::Handle(), /*content_lanes=*/1,
        global_region.offset, global_region.elements);
    PrimExpr shared_ptr = MakeTmaSharedPtr(op.dst, op.dst_range, args, analyzer,
                                           /*rw_mask=*/2);
    PrimExpr barrier = GetTmaBarrier(op);
    Stmt load = Evaluate(Call(DataType::Handle(), tma_load(),
                              {shared_ptr, global_ptr, barrier, bytes}));
    return IfThenElse(EQ(args.thread_index, args.thread_bounds->min), load);
  }

  SwizzleLayout swizzle = GetSwizzleParams(op.dst, args);
  if (swizzle.swizzle_granularity != SwizzleGranularity::kNone &&
      args.require_smem_alignment) {
    args.require_smem_alignment(op.dst->data, 256);
  }

  PrimExpr barrier;
  if (IsExplicitTmaCopy(op)) {
    barrier = GetTmaBarrier(op);
  } else {
    ICHECK(args.alloc_mbarrier)
        << "T.copy TME load requires an mbarrier allocator";
    int barrier_index = args.alloc_mbarrier(1, "tme_copy_mbarrier");
    barrier =
        BufferLoad((*args.mbarrier_buffer).value(), {Integer(barrier_index)});
  }

  auto emit_load = [&](const CopyNode &part) -> Stmt {
    LoweredTMEDesc lowered =
        MakeTmaDescriptor(part.src, part.src_range, part.dst_range, analyzer);
    PrimExpr shared_ptr =
        MakeTmaSharedPtr(part.dst, part.dst_range, args, analyzer,
                         /*rw_mask=*/2);
    Array<PrimExpr> tma_args;
    tma_args.push_back(lowered.descriptor);
    tma_args.push_back(barrier);
    tma_args.push_back(shared_ptr);
    tma_args.insert(tma_args.end(), lowered.global_coords.begin(),
                    lowered.global_coords.end());
    tma_args.insert(tma_args.end(), lowered.box_dims.begin(),
                    lowered.box_dims.end());
    tma_args.push_back(Integer(static_cast<int>(swizzle.swizzle_granularity)));
    tma_args.push_back(Integer(static_cast<int>(swizzle.swizzle_stride)));
    tma_args.push_back(Integer(static_cast<int>(swizzle.swizzle_line)));
    return Evaluate(Call(DataType::Handle(), tma_load(), tma_args));
  };

  const std::vector<ObjectPtr<CopyNode>> splits =
      SplitPanelizedCopy(op, args, analyzer);
  Array<Stmt> load_parts;
  if (splits.empty()) {
    load_parts.push_back(emit_load(op));
  } else {
    for (const ObjectPtr<CopyNode> &part : splits) {
      load_parts.push_back(emit_load(*part));
    }
  }
  Stmt load = load_parts.size() == 1 ? load_parts[0] : SeqStmt(load_parts);

  PrimExpr bytes = 1;
  for (const Range &range : op.dst_range) {
    bytes *= range->extent;
  }
  bytes = analyzer->Simplify(bytes * op.dst->dtype.bytes());
  Stmt expect = Evaluate(
      Call(DataType::Handle(), mbarrier_expect_tx(), {barrier, bytes}));
  load = SeqStmt({expect, load});
  PrimExpr leader = EQ(args.thread_index, args.thread_bounds->min);
  if (IsExplicitTmaCopy(op)) {
    return IfThenElse(leader, load);
  }
  Stmt arrive = Evaluate(
      Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {barrier}));
  Stmt issue = IfThenElse(leader, SeqStmt({load, arrive}));
  Stmt wait = Evaluate(Call(DataType::Handle(), mbarrier_wait_parity(),
                            {barrier, args.mbar_phase_expr}));
  return SeqStmt({issue, wait});
}

Stmt LowerTmaStore(const CopyNode &op, const LowerArgs &args,
                   arith::Analyzer *analyzer) {
  ICHECK(TargetIsMP31(args.target))
      << "MP31 TME store requires an MP31 MUSA target, got " << args.target;
  ICHECK(IsSharedBuffer(op.src) && IsGlobalBuffer(op.dst))
      << "MP31 TME store only supports shared-to-global copies, got src="
      << op.src.scope() << ", dst=" << op.dst.scope();
  ICHECK_EQ(op.src->dtype, op.dst->dtype)
      << "MP31 TME store requires matching source and destination dtypes";

  if (GetBoolAnnotation(op, "musa_runtime_pointer_tme")) {
    ICHECK(IsExplicitTmaCopy(op))
        << "Runtime-pointer TME stores require explicit T.tma_copy";
    ICHECK_EQ(GetSwizzleParams(op.src, args).swizzle_granularity,
              SwizzleGranularity::kNone)
        << "Runtime-pointer TME stores do not support swizzled shared layouts";
    ICHECK(!GetPanelizedCols(op.src, args).has_value())
        << "Runtime-pointer TME stores do not support panelized shared layouts";
    ContiguousTmaRegion shared_region =
        AnalyzeContiguousTmaRegion(op.src, op.src_range, analyzer, "source");
    ContiguousTmaRegion global_region = AnalyzeContiguousTmaRegion(
        op.dst, op.dst_range, analyzer, "destination");
    ICHECK(
        analyzer->CanProveEqual(shared_region.elements, global_region.elements))
        << "Runtime-pointer TME requires matching shared/destination extents";
    PrimExpr bytes =
        analyzer->Simplify(global_region.elements * op.dst->dtype.bytes());
    PrimExpr global_ptr = op.dst.access_ptr(
        /*access_mask=*/2, DataType::Handle(), /*content_lanes=*/1,
        global_region.offset, global_region.elements);
    PrimExpr shared_ptr = MakeTmaSharedPtr(op.src, op.src_range, args, analyzer,
                                           /*rw_mask=*/1);
    Stmt store = Evaluate(
        Call(DataType::Handle(), tma_store(), {global_ptr, shared_ptr, bytes}));
    Stmt commit = Evaluate(Call(DataType::Handle(), tma_store_arrive(), {}));
    return IfThenElse(EQ(args.thread_index, args.thread_bounds->min),
                      SeqStmt({store, commit}));
  }

  SwizzleLayout swizzle = GetSwizzleParams(op.src, args);
  if (swizzle.swizzle_granularity != SwizzleGranularity::kNone &&
      args.require_smem_alignment) {
    args.require_smem_alignment(op.src->data, 256);
  }

  auto emit_store = [&](const CopyNode &part) -> Stmt {
    LoweredTMEDesc lowered =
        MakeTmaDescriptor(part.dst, part.dst_range, part.src_range, analyzer);
    PrimExpr shared_ptr =
        MakeTmaSharedPtr(part.src, part.src_range, args, analyzer,
                         /*rw_mask=*/1);
    Array<PrimExpr> tma_args;
    tma_args.push_back(lowered.descriptor);
    tma_args.push_back(shared_ptr);
    tma_args.insert(tma_args.end(), lowered.global_coords.begin(),
                    lowered.global_coords.end());
    tma_args.insert(tma_args.end(), lowered.box_dims.begin(),
                    lowered.box_dims.end());
    tma_args.push_back(Integer(static_cast<int>(swizzle.swizzle_granularity)));
    tma_args.push_back(Integer(static_cast<int>(swizzle.swizzle_stride)));
    tma_args.push_back(Integer(static_cast<int>(swizzle.swizzle_line)));
    return Evaluate(Call(DataType::Handle(), tma_store(), tma_args));
  };

  const std::vector<ObjectPtr<CopyNode>> splits =
      SplitPanelizedCopy(op, args, analyzer);
  Array<Stmt> store_parts;
  if (splits.empty()) {
    store_parts.push_back(emit_store(op));
  } else {
    for (const ObjectPtr<CopyNode> &part : splits) {
      store_parts.push_back(emit_store(*part));
    }
  }
  Stmt store = store_parts.size() == 1 ? store_parts[0] : SeqStmt(store_parts);
  Stmt commit = Evaluate(Call(DataType::Handle(), tma_store_arrive(), {}));
  Stmt issue = IfThenElse(EQ(args.thread_index, args.thread_bounds->min),
                          SeqStmt({store, commit}));
  if (IsExplicitTmaCopy(op)) {
    return issue;
  }
  Stmt wait = Evaluate(
      Call(DataType::Handle(), tl::tma_store_wait(), {Integer(0), Integer(1)}));
  return SeqStmt({issue, wait});
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
      /*parallel_loop=*/true, par_op->LoopLayoutRequiresPaddingGuard());

  const bool explicit_async = IsExplicitAsyncCopy(op);
  const bool pipeline_managed = HasPipelineManagedAsyncAnnotation(op);
  auto injected =
      InjectMUSAAsyncCopy(lowered_loop, /*async_without_async_commit_wait=*/
                          explicit_async || pipeline_managed);
  if (!injected.injected_ptx_async_copy) {
    ICHECK(!explicit_async)
        << "T.async_copy requires an eligible global-to-shared vectorized "
           "copy; no SIMT fallback is allowed.";
    ICHECK(!PreferAsyncCopy(op))
        << "T.copy(prefer_instruction=\"cp_async\") requires an eligible "
           "global-to-shared vectorized copy; no SIMT fallback is allowed.";
    return lowered_loop;
  }

  if (explicit_async) {
    Stmt commit_group =
        Evaluate(Call(DataType::Handle(), builtin::ptx_commit_group(), {}));
    return SeqStmt({injected.stmt, commit_group});
  }

  // A pipeline-managed copy leaves commit/wait ownership to the enclosing
  // software pipeline.  A regular T.copy preference keeps synchronous copy
  // semantics; InjectMUSAAsyncCopy(false) has already appended commit+wait.
  return injected.stmt;
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
  case CopyInst::kTMEStore:
    return "tme_store";
  case CopyInst::kInvalid:
    return "invalid";
  }
  return "unknown";
}

bool CopyInstIsTME(CopyInst inst) {
  return inst == CopyInst::kTMELoad || inst == CopyInst::kTMEStore;
}

bool CopyInstIsTMEStore(CopyInst inst) { return inst == CopyInst::kTMEStore; }

bool CopyInstIsAsync(CopyInst inst) { return inst == CopyInst::kAsync; }

CopyInstSelection SelectCopyInstForLowering(const CopyNode &op,
                                            const CopyAnalysisContext &ctx) {
  const std::string preference = GetCopyPreference(op);
  if (!preference.empty() && preference != "tma" && preference != "cp_async" &&
      preference != "sync") {
    return {CopyInst::kInvalid, false,
            "Unknown copy prefer_instruction=\"" + preference + "\""};
  }
  if (IsExplicitTmaCopy(op)) {
    if (!TargetIsMP31(ctx.target)) {
      return {CopyInst::kInvalid, false,
              "T.tma_copy() is currently supported only on MP31 MUSA "
              "targets"};
    }
    if (IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst)) {
      return {CopyInst::kTMELoad, true, {}};
    }
    if (IsSharedBuffer(op.src) && IsGlobalBuffer(op.dst)) {
      return {CopyInst::kTMEStore, true, {}};
    }
    return {CopyInst::kInvalid, false,
            "T.tma_copy() only supports global-to-shared loads or "
            "shared-to-global stores on MP31"};
  }
  if (PreferTmaCopy(op)) {
    if (DisableTmaCopy(op)) {
      return {CopyInst::kInvalid, false,
              "T.copy prefer_instruction=\"tma\" conflicts with "
              "disable_tma=True"};
    }
    if (!TargetIsMP31(ctx.target)) {
      return {CopyInst::kInvalid, false,
              "T.copy prefer_instruction=\"tma\" is supported only on MP31 "
              "MUSA targets"};
    }
    if (IsGlobalBuffer(op.src) && IsSharedBuffer(op.dst)) {
      return {CopyInst::kTMELoad, true, {}};
    }
    if (IsSharedBuffer(op.src) && IsGlobalBuffer(op.dst)) {
      return {CopyInst::kTMEStore, true, {}};
    }
    return {CopyInst::kInvalid, false,
            "T.copy prefer_instruction=\"tma\" only supports "
            "global-to-shared loads or shared-to-global stores on MP31"};
  }
  if (IsExplicitAsyncCopy(op) || PreferAsyncCopy(op)) {
    return {CopyInst::kAsync, true, {}};
  }
  if (PreferSyncCopy(op)) {
    return {CopyInst::kNormal, true, {}};
  }
  if (HasPipelineManagedAsyncAnnotation(op)) {
    return {CopyInst::kAsync, true, {}};
  }
  if (!DisableTmaCopy(op) && CanAutoSelectTmeStore(op, ctx)) {
    return {CopyInst::kTMEStore, true, {}};
  }
  return {CopyInst::kNormal, true, {}};
}

CopyInstSelection ClassifyWarpSpecializedProducerCopy(const CopyNode &op,
                                                      Target target) {
  const std::string preference = GetCopyPreference(op);
  if (!preference.empty() && preference != "tma" && preference != "cp_async" &&
      preference != "sync") {
    return {CopyInst::kInvalid, false,
            "Unknown copy prefer_instruction=\"" + preference + "\""};
  }
  if (IsExplicitTmaCopy(op) || PreferTmaCopy(op)) {
    if (!IsExplicitTmaCopy(op) && DisableTmaCopy(op)) {
      return {CopyInst::kInvalid, false,
              "T.copy prefer_instruction=\"tma\" conflicts with "
              "disable_tma=True"};
    }
    if (TargetIsMP31(target) && IsGlobalBuffer(op.src) &&
        IsSharedBuffer(op.dst)) {
      return {CopyInst::kTMELoad, true, {}};
    }
    return {CopyInst::kInvalid, false,
            "MP31 warp-specialized TME producer must copy global-to-shared"};
  }
  if (IsExplicitAsyncCopy(op) || PreferAsyncCopy(op)) {
    if (TargetMUSAHasAsyncCopy(target) && IsGlobalBuffer(op.src) &&
        IsSharedBuffer(op.dst) && op.src->dtype == op.dst->dtype) {
      return {CopyInst::kAsync, true, {}};
    }
    return {CopyInst::kInvalid, false,
            "MP31 warp-specialized async producer must be an eligible "
            "global-to-shared copy"};
  }
  if (PreferSyncCopy(op)) {
    return {CopyInst::kNormal, true, {}};
  }
  if (CanWarpSpecializeTmeLoad(op, target)) {
    return {CopyInst::kTMELoad, true, {}};
  }
  if (HasPipelineManagedAsyncAnnotation(op)) {
    return {CopyInst::kAsync, true, {}};
  }
  return {CopyInst::kNormal, true, {}};
}

bool IsPipelineManagedAsyncCopy(const CopyNode &op, Target target) {
  const std::string preference = GetCopyPreference(op);
  return !IsExplicitTmaCopy(op) && !IsExplicitAsyncCopy(op) &&
         preference.empty() && IsGlobalBuffer(op.src) &&
         IsSharedBuffer(op.dst) && op.src->dtype == op.dst->dtype &&
         TargetMUSAHasAsyncCopy(target);
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
    if (CopyInstIsTMEStore(selection.inst)) {
      return LowerTmaStore(op, lower_args, analyzer);
    }
    if (CopyInstIsTME(selection.inst)) {
      return LowerTmaLoad(op, lower_args, analyzer);
    }
    const std::vector<ObjectPtr<CopyNode>> splits =
        SplitPanelizedCopy(op, lower_args, analyzer);
    if (splits.empty()) {
      if (CopyInstIsAsync(selection.inst)) {
        return LowerAsyncCopy(op, lower_args, analyzer);
      }
      return LowerNormalCopy(op, lower_args, analyzer);
    }
    Array<Stmt> lowered;
    for (const ObjectPtr<CopyNode> &part : splits) {
      lowered.push_back(CopyInstIsAsync(selection.inst)
                            ? LowerAsyncCopy(*part, lower_args, analyzer)
                            : LowerNormalCopy(*part, lower_args, analyzer));
    }
    return lowered.size() == 1 ? lowered[0] : SeqStmt(lowered);
  }
};

struct Im2Col {
  static Stmt Lower(const Im2ColOpNode &op, const LowerArgs &lower_args,
                    arith::Analyzer *analyzer) {
    ICHECK(TargetIsMP31(lower_args.target))
        << "T.im2col is supported only on MP31 MUSA targets";
    const Buffer &src = op.src_;
    const Buffer &dst_unmapped = op.dst_;
    const BufferRegion &dst_region = op.dstRegion_;
    ICHECK(src.scope() == "global")
        << "MP31 TME im2col requires a global input buffer";
    ICHECK(dst_unmapped.scope() == "shared" ||
           dst_unmapped.scope() == "shared.dyn")
        << "MP31 TME im2col requires a shared-memory destination";
    ICHECK_EQ(src->shape.size(), 4)
        << "MP31 TME im2col requires a 4D NHWC input";
    ICHECK_GE(dst_region->region.size(), 2)
        << "MP31 TME im2col destination must have at least two dimensions";
    ICHECK_EQ(dst_region->region.size(), dst_unmapped->shape.size())
        << "MP31 TME im2col requires a complete destination region rank";
    ICHECK(src->dtype == dst_unmapped->dtype)
        << "MP31 TME im2col requires matching source/destination dtypes";
    ICHECK(TmaDescriptorSupportsDataType(src->dtype))
        << "MP31 TME im2col does not support dtype " << src->dtype;

    int64_t kernel = op.kernel_;
    int64_t stride = op.stride_;
    int64_t dilation = op.dilation_;
    int64_t padding = op.padding_;
    ICHECK_GT(kernel, 0) << "MP31 TME im2col requires kernel > 0";
    ICHECK_GT(stride, 0) << "MP31 TME im2col requires stride > 0";
    ICHECK_GT(dilation, 0) << "MP31 TME im2col requires dilation > 0";
    ICHECK_GE(padding, 0) << "MP31 TME im2col requires padding >= 0";

    const size_t dst_rank = dst_region->region.size();
    PrimExpr block_m = dst_region->region[dst_rank - 2]->extent;
    PrimExpr block_k = dst_region->region[dst_rank - 1]->extent;
    PrimExpr transaction_bytes =
        analyzer->Simplify(block_m * block_k * dst_unmapped->dtype.bytes());
    PrimExpr channel_bytes =
        analyzer->Simplify(block_k * dst_unmapped->dtype.bytes());
    ICHECK(analyzer->CanProveEqual(floormod(channel_bytes, Integer(16)),
                                   Integer(0)))
        << "MP31 TME im2col requires block_K * dtype_bytes to be 16-byte "
           "aligned, got "
        << channel_bytes << " bytes";

    PrimExpr h = src->shape[1];
    PrimExpr w = src->shape[2];
    PrimExpr c = src->shape[3];
    // One MP31 im2col transaction cannot cross a kernel-position channel
    // slice. TileLang's c_step is a tile index, so C divisible by block_k
    // guarantees every legal tile starts and ends within one such slice.
    ICHECK(analyzer->CanProveEqual(floormod(c, block_k), Integer(0)))
        << "MP31 TME im2col requires input channels to be divisible by "
           "block_K";
    auto global_stride_bytes = GetTmaGlobalStrideBytes(src, analyzer);
    ICHECK(global_stride_bytes.has_value())
        << "MP31 TME im2col requires a contiguous innermost dimension and "
           "16-byte aligned outer strides";

    PrimExpr pad_term = Integer(2 * padding - dilation * (kernel - 1) - 1);
    PrimExpr p = analyzer->Simplify(floordiv(h + pad_term, stride) + 1);
    PrimExpr q = analyzer->Simplify(floordiv(w + pad_term, stride) + 1);
    PrimExpr m_start = analyzer->Simplify(op.nhw_step_ * block_m);
    PrimExpr k_start = analyzer->Simplify(op.c_step_ * block_k);
    PrimExpr rs = analyzer->Simplify(floordiv(k_start, c));
    PrimExpr kernel_r = analyzer->Simplify(floordiv(rs, kernel));
    PrimExpr kernel_s = analyzer->Simplify(floormod(rs, kernel));
    // MP31 packs the W/S kernel position in the low byte and H/R in the next
    // byte, matching the old architecture instruction contract.
    PrimExpr weight_pos =
        analyzer->Simplify(kernel_s + kernel_r * Integer(256));

    Array<PrimExpr> box_dims = {block_k, block_m, Integer(1), Integer(1)};
    PrimExpr descriptor =
        MakeIm2ColTmaDescriptor(src, box_dims, global_stride_bytes.value());
    Buffer shared = dst_unmapped;
    if (lower_args.buffer_remap.count(shared)) {
      shared = lower_args.buffer_remap.at(shared);
    }
    if (lower_args.require_smem_alignment) {
      lower_args.require_smem_alignment(shared->data, 128);
    }

    PrimExpr shared_offset = Integer(0);
    PrimExpr shared_stride = Integer(1);
    for (size_t i = 0; i < dst_unmapped->shape.size(); ++i) {
      size_t axis = dst_unmapped->shape.size() - i - 1;
      shared_offset += dst_region->region[axis]->min * shared_stride;
      shared_stride *= dst_unmapped->shape[axis];
    }
    shared_offset = analyzer->Simplify(shared_offset);
    PrimExpr shared_ptr = shared.access_ptr(2, DataType::Handle(), 1,
                                            shared_offset, block_m * block_k);

    PrimExpr output_volume = analyzer->Simplify(p * q);
    Array<PrimExpr> block_pos = {
        analyzer->Simplify(floormod(k_start, c)),
        analyzer->Simplify(floormod(floormod(m_start, output_volume), q)),
        analyzer->Simplify(floordiv(floormod(m_start, output_volume), q)),
        analyzer->Simplify(floordiv(m_start, output_volume))};

    PrimExpr barrier;
    bool user_managed_barrier = false;
    if (auto user_barrier = op.annotations_.Get("barrier")) {
      barrier = Downcast<PrimExpr>(user_barrier.value());
      user_managed_barrier = true;
    } else {
      ICHECK(lower_args.alloc_mbarrier && lower_args.mbarrier_buffer != nullptr)
          << "MP31 TME im2col requires mbarrier allocation support";
      int barrier_slot =
          lower_args.alloc_mbarrier(1, std::string("im2col_mbarrier"));
      ICHECK(lower_args.mbarrier_buffer->defined());
      barrier = BufferLoad(lower_args.mbarrier_buffer->value(),
                           {IntImm(DataType::Int(32), barrier_slot)});
    }

    Array<PrimExpr> args = {descriptor,
                            barrier,
                            shared_ptr,
                            block_k,
                            block_m,
                            block_pos[0],
                            block_pos[1],
                            block_pos[2],
                            block_pos[3],
                            weight_pos,
                            p,
                            q,
                            Integer(padding * 257),
                            Integer(65536 + stride * 257),
                            Integer(65536 + dilation * 257)};
    Stmt load =
        Evaluate(Call(DataType::Handle(), tma_load_im2col(), std::move(args)));
    Stmt expect = Evaluate(Call(DataType::Handle(), mbarrier_expect_tx(),
                                {barrier, transaction_bytes}));

    Array<Stmt> producer_seq{expect, load};
    if (user_managed_barrier) {
      if (auto emit_arrive = op.annotations_.Get("emit_arrive")) {
        if (Downcast<IntImm>(emit_arrive.value())->value != 0) {
          producer_seq.push_back(Evaluate(Call(
              DataType::Handle(), builtin::ptx_arrive_barrier(), {barrier})));
        }
      }
    } else {
      producer_seq.push_back(Evaluate(
          Call(DataType::Handle(), builtin::ptx_arrive_barrier(), {barrier})));
    }

    Stmt producer = IfThenElse(Call(DataType::Bool(), tl_shuffle_elect(),
                                    {lower_args.thread_bounds->extent}),
                               SeqStmt(producer_seq));
    if (user_managed_barrier) {
      return producer;
    }
    PrimExpr phase = lower_args.mbar_phase_expr;
    if (auto explicit_phase = GetAnnotatedMbarPhaseExpr(op.annotations_)) {
      phase = explicit_phase.value();
    }
    Stmt wait = Evaluate(
        Call(DataType::Handle(), mbarrier_wait_parity(), {barrier, phase}));
    return SeqStmt({producer, wait});
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

bool RegisterMUSAIm2Col() {
  RegisterIm2ColImpl(Im2ColImpl{
      "musa.Im2Col",
      MatchMUSACopyTarget,
      100,
      musa::Im2Col::Lower,
  });
  return true;
}

const bool musa_im2col_registered = RegisterMUSAIm2Col();

} // namespace

} // namespace tl
} // namespace tvm
