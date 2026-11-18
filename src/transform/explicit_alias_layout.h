/*!
 * \file explicit_alias_layout.h
 * \brief Layout propagation helpers for explicit TileLang alias buffers.
 */

#ifndef TVM_TL_TRANSFORM_EXPLICIT_ALIAS_LAYOUT_H_
#define TVM_TL_TRANSFORM_EXPLICIT_ALIAS_LAYOUT_H_

#include <tvm/arith/analyzer.h>
#include <tvm/tirx/buffer.h>
#include <tvm/tirx/expr.h>

#include "../layout/layout.h"

namespace tvm {
namespace tl {

int64_t GetElementStorageBits(DataType dtype);

bool ShapesEqual(const ffi::Array<PrimExpr> &lhs,
                 const ffi::Array<PrimExpr> &rhs, arith::Analyzer *analyzer);

bool IsExplicitAliasPair(const tirx::Buffer &source, const tirx::Buffer &alias);

Layout ReshapeLayoutForAlias(const Layout &layout, const tirx::Buffer &source,
                             const tirx::Buffer &alias,
                             arith::Analyzer *analyzer);

ffi::Optional<tirx::Buffer>
FindLayoutAnchorBuffer(const ffi::Array<tirx::Buffer> &buffers,
                       const Layout &layout, arith::Analyzer *analyzer);

} // namespace tl
} // namespace tvm

#endif // TVM_TL_TRANSFORM_EXPLICIT_ALIAS_LAYOUT_H_
