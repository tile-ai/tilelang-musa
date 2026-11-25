/*!
 * \file explicit_alias_layout.cc
 * \brief Layout propagation helpers for explicit TileLang alias buffers.
 */

#include "explicit_alias_layout.h"

#include <tvm/tirx/index_map.h>
#include <tvm/tirx/stmt_functor.h>

#include "support/check.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace {

bool HasCompactAddressing(const Buffer &buffer, arith::Analyzer *analyzer) {
  if (!buffer->axis_separators.empty()) {
    return false;
  }
  if (buffer->strides.empty()) {
    return true;
  }

  PrimExpr expected_stride = Integer(1);
  for (size_t i = buffer->shape.size(); i > 0; --i) {
    size_t dim = i - 1;
    if (!analyzer->CanProveEqual(buffer->strides[dim], expected_stride)) {
      return false;
    }
    expected_stride = analyzer->Simplify(expected_stride * buffer->shape[dim]);
  }
  return true;
}

bool HaveEquivalentAddressing(const Buffer &lhs, const Buffer &rhs,
                              arith::Analyzer *analyzer) {
  if (GetElementStorageBits(lhs->dtype) != GetElementStorageBits(rhs->dtype) ||
      !ShapesEqual(lhs->shape, rhs->shape, analyzer)) {
    return false;
  }

  Array<PrimExpr> indices;
  for (size_t i = 0; i < lhs->shape.size(); ++i) {
    indices.push_back(InputPlaceholder(i));
  }
  auto lhs_offsets = lhs.OffsetOf(indices);
  auto rhs_offsets = rhs.OffsetOf(indices);
  if (lhs_offsets.size() != rhs_offsets.size()) {
    return false;
  }
  for (size_t i = 0; i < lhs_offsets.size(); ++i) {
    if (!analyzer->CanProveEqual(lhs_offsets[i], rhs_offsets[i])) {
      return false;
    }
  }
  return true;
}

} // namespace

int64_t GetElementStorageBits(DataType dtype) {
  // Layout aliasing must be reasoned about in logical storage bits per element,
  // not in bytes.  For sub-byte dtypes such as fp4, `dtype.bytes()` rounds up
  // to 1 and loses the "two fp4 values share one byte" relationship that
  // subtype-changing views rely on.
  return static_cast<int64_t>(dtype.bits()) * dtype.lanes();
}

bool ShapesEqual(const Array<PrimExpr> &lhs, const Array<PrimExpr> &rhs,
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

bool IsExplicitAliasPair(const Buffer &source, const Buffer &alias) {
  return source->is_explicit_alias || alias->is_explicit_alias;
}

Layout ReshapeLayoutForAlias(const Layout &layout, const Buffer &source,
                             const Buffer &alias, arith::Analyzer *analyzer) {
  const int64_t source_bits = GetElementStorageBits(source->dtype);
  const int64_t alias_bits = GetElementStorageBits(alias->dtype);

  // Preserve the pre-alias behavior for ordinary views such as T.reshape and
  // T.view. The IR groups all buffers sharing a data Var, but only
  // T.alias_buffer is marked as an explicit physical alias.
  if (!IsExplicitAliasPair(source, alias)) {
    if (ShapesEqual(layout->InputShape(), alias->shape, analyzer)) {
      return layout;
    }
    return layout->Reshape(alias->shape, analyzer, Integer(source_bits),
                           Integer(alias_bits));
  }

  if (source_bits != alias_bits) {
    return layout->Reshape(alias->shape, analyzer, Integer(source_bits),
                           Integer(alias_bits));
  }
  if (HaveEquivalentAddressing(source, alias, analyzer)) {
    return layout;
  }
  if (HasCompactAddressing(source, analyzer) &&
      HasCompactAddressing(alias, analyzer) &&
      analyzer->CanProveEqual(source->elem_offset, alias->elem_offset)) {
    return layout->Reshape(alias->shape, analyzer);
  }
  if (!HasCompactAddressing(source, analyzer)) {
    TVM_FFI_THROW(ValueError)
        << "Cannot propagate layout from non-compact alias " << source->name
        << " to " << alias->name
        << "; annotate the compact backing buffer instead";
  }

  if (layout.as<Fragment>()) {
    TVM_FFI_THROW(ValueError)
        << "Cannot propagate a non-compact explicit alias from fragment "
        << source->name << " to " << alias->name
        << "; annotate the compact fragment view instead";
  }

  // An alias is a storage view, not necessarily a compact reshape. Recover
  // the source's logical coordinates from the alias's element address before
  // applying the source layout (e.g. [slice,row,d] -> [row,slice*64+d]).
  Array<Var> source_vars;
  Array<PrimExpr> source_indices;
  Array<Range> source_ranges;
  for (const auto &extent : source->shape) {
    Var var("alias_source_index", extent.dtype());
    source_vars.push_back(var);
    source_indices.push_back(var);
    source_ranges.push_back(Range::FromMinExtent(0, extent));
  }
  IndexMap source_address(source_vars, source.OffsetOf(source_indices));
  if (source_address->final_indices.size() != 1U) {
    TVM_FFI_THROW(ValueError)
        << "Cannot propagate layout through alias " << alias->name
        << ": source " << source->name
        << " does not have a single physical address axis";
  }
  auto [inverse, padding_predicate] =
      source_address.NonSurjectiveInverse(source_ranges, analyzer);

  Array<PrimExpr> alias_indices;
  for (size_t i = 0; i < alias->shape.size(); ++i) {
    alias_indices.push_back(InputPlaceholder(i));
  }
  auto alias_offsets = alias.OffsetOf(alias_indices);
  if (alias_offsets.size() != 1U) {
    TVM_FFI_THROW(ValueError)
        << "Cannot propagate layout through alias " << alias->name
        << ": it does not have a single physical address axis";
  }

  ICHECK_EQ(inverse->initial_indices.size(), alias_offsets.size());
  Map<Var, PrimExpr> address_substitution;
  for (size_t i = 0; i < alias_offsets.size(); ++i) {
    address_substitution.Set(inverse->initial_indices[i], alias_offsets[i]);
  }
  PrimExpr invalid_alias_address =
      tirx::Substitute(padding_predicate, address_substitution);
  auto mapped_indices = inverse->MapIndices(alias_offsets, analyzer);
  ICHECK_EQ(mapped_indices.size(), source->shape.size());
  for (size_t i = 0; i < mapped_indices.size(); ++i) {
    invalid_alias_address = invalid_alias_address || mapped_indices[i] < 0 ||
                            mapped_indices[i] >= source->shape[i];
  }
  auto round_trip_offsets = source.OffsetOf(mapped_indices);
  ICHECK_EQ(round_trip_offsets.size(), alias_offsets.size());
  for (size_t i = 0; i < alias_offsets.size(); ++i) {
    invalid_alias_address =
        invalid_alias_address || round_trip_offsets[i] != alias_offsets[i];
  }

  auto proof_analyzer = analyzer->Clone();
  for (size_t i = 0; i < alias->shape.size(); ++i) {
    proof_analyzer->Bind(InputPlaceholder(i),
                         Range::FromMinExtent(Integer(0), alias->shape[i]),
                         /*allow_override=*/true);
  }
  if (!proof_analyzer->CanProve(!invalid_alias_address)) {
    TVM_FFI_THROW(ValueError)
        << "Cannot prove that every address of alias " << alias->name
        << " maps to compact source buffer " << source->name
        << "; alias shape=" << alias->shape << ", strides=" << alias->strides
        << ", elem_offset=" << alias->elem_offset;
  }

  return layout->CopyMetadataTo(
      Layout(alias->shape, layout->Forward(mapped_indices)));
}

Optional<Buffer> FindLayoutAnchorBuffer(const Array<Buffer> &buffers,
                                        const Layout &layout,
                                        arith::Analyzer *analyzer) {
  const bool has_explicit_alias =
      std::any_of(buffers.begin(), buffers.end(), [](const Buffer &buffer) {
        return buffer->is_explicit_alias;
      });
  Optional<Buffer> anchor;
  for (const auto &buffer : buffers) {
    if (ShapesEqual(layout->InputShape(), buffer->shape, analyzer)) {
      if (!anchor.defined()) {
        anchor = buffer;
      } else if (has_explicit_alias &&
                 !HaveEquivalentAddressing(anchor.value(), buffer, analyzer)) {
        TVM_FFI_THROW(ValueError)
            << "Ambiguous layout annotation for data-shared buffers "
            << anchor.value()->name << " and " << buffer->name
            << ": both match shape " << layout->InputShape()
            << " but use different addressing";
      }
    }
  }
  return anchor;
}

} // namespace tl
} // namespace tvm
