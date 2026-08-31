#include "swizzle_layout.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {
namespace musa {
using namespace ffi;
using namespace tirx;

Layout MakeSwizzleLayout(const tirx::Buffer &buffer,
                         const SwizzleLayout &swizzle) {
  ICHECK(buffer.defined()) << "Swizzle layout expects a defined buffer";
  ICHECK_EQ(buffer->shape.size(), 2)
      << "Swizzle layout expects a 2D buffer, got rank="
      << buffer->shape.size();

  const int64_t *rows_ptr = as_const_int(buffer->shape[0]);
  const int64_t *cols_ptr = as_const_int(buffer->shape[1]);
  ICHECK(rows_ptr && cols_ptr)
      << "Swizzle layout requires constant buffer shape";
  const int rows = static_cast<int>(*rows_ptr);
  const int cols = static_cast<int>(*cols_ptr);
  const int element_size = buffer->dtype.bits();
  const int sg = static_cast<int>(swizzle.swizzle_granularity);
  const int ss = static_cast<int>(swizzle.swizzle_stride);
  const int sl = static_cast<int>(swizzle.swizzle_line);
  ICHECK_GT(rows, 0);
  ICHECK_GT(cols, 0);
  ICHECK_GT(element_size, 0);
  ICHECK_EQ(element_size % 8, 0)
      << "Swizzle layout element size must be byte-addressable";
  ICHECK_GT(sg, 0);
  ICHECK_GT(ss, 0);
  ICHECK_GT(sl, 0);
  ICHECK_EQ(ss % sg, 0) << "Swizzle stride must be divisible by granularity";

  Var row = InputPlaceholder(0), col = InputPlaceholder(1);
  PrimExpr addr = (row * cols + col) * (element_size / 8);
  PrimExpr line_id = FloorDiv(addr, sl);
  PrimExpr line_offset = FloorMod(addr, sl);
  PrimExpr cycle_line_id = FloorMod(line_id, ss / sg);
  PrimExpr granule_id = FloorDiv(line_offset, sg);
  PrimExpr granule_offset = FloorMod(line_offset, sg);
  PrimExpr target_granule_id = granule_id ^ cycle_line_id;
  PrimExpr target_addr = line_id * sl + target_granule_id * sg + granule_offset;
  PrimExpr target_linear = FloorDiv(target_addr, element_size / 8);
  return Layout(Array<PrimExpr>{rows, cols},
                {FloorDiv(target_linear, cols), FloorMod(target_linear, cols)});
}

SwizzleLayout AnalyzeSwizzleLayout(const tirx::Buffer &buffer,
                                   const Layout &layout) {
  if (!buffer.defined() || !layout.defined() || buffer->shape.size() != 2) {
    return {SwizzleGranularity::kNone, SwizzleStride::k256B,
            SwizzleLine::k256B};
  }
  const int64_t *rows_ptr = as_const_int(buffer->shape[0]);
  const int64_t *cols_ptr = as_const_int(buffer->shape[1]);
  if (rows_ptr == nullptr || cols_ptr == nullptr) {
    return {SwizzleGranularity::kNone, SwizzleStride::k256B,
            SwizzleLine::k256B};
  }
  const int rows = static_cast<int>(*rows_ptr);
  const int cols = static_cast<int>(*cols_ptr);
  const int element_size = buffer->dtype.bits();
  if (rows <= 0 || cols <= 0 || element_size <= 0 || element_size % 8 != 0) {
    return {SwizzleGranularity::kNone, SwizzleStride::k256B,
            SwizzleLine::k256B};
  }
  for (SwizzleGranularity sg :
       {SwizzleGranularity::k16B, SwizzleGranularity::k32B,
        SwizzleGranularity::k64B, SwizzleGranularity::k128B}) {
    for (SwizzleStride ss : {SwizzleStride::k256B, SwizzleStride::k128B,
                             SwizzleStride::k64B, SwizzleStride::k32B}) {
      for (SwizzleLine sl : {SwizzleLine::k256B, SwizzleLine::k128B}) {
        const int sg_bytes = static_cast<int>(sg);
        const int ss_bytes = static_cast<int>(ss);
        if (ss_bytes < sg_bytes || ss_bytes % sg_bytes != 0) {
          continue;
        }
        SwizzleLayout candidate_swizzle{sg, ss, sl};
        Layout candidate = MakeSwizzleLayout(buffer, candidate_swizzle);
        if (StructuralEqual()(layout, candidate)) {
          return candidate_swizzle;
        }
      }
    }
  }
  return {SwizzleGranularity::kNone, SwizzleStride::k256B, SwizzleLine::k256B};
}

} // namespace musa
} // namespace tl
} // namespace tvm
