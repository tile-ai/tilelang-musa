#ifndef TVM_TILELANG_MUSA_LAYOUT_SWIZZLE_LAYOUT_H_
#define TVM_TILELANG_MUSA_LAYOUT_SWIZZLE_LAYOUT_H_

#include "layout/layout.h"

namespace tvm {
namespace tl {
namespace musa {

// Byte-level swizzle parameters used to construct or identify a layout.
enum class SwizzleGranularity : int {
  kNone = 0,
  k16B = 16,
  k32B = 32,
  k64B = 64,
  k128B = 128,
};

enum class SwizzleStride : int {
  k32B = 32,
  k64B = 64,
  k128B = 128,
  k256B = 256,
};

enum class SwizzleLine : int {
  k128B = 128,
  k256B = 256,
};

struct SwizzleLayout {
  SwizzleGranularity swizzle_granularity;
  SwizzleStride swizzle_stride;
  SwizzleLine swizzle_line;
};

// Construct a 2D row-major layout with the requested byte-level swizzle.
Layout MakeSwizzleLayout(const tirx::Buffer &buffer,
                         const SwizzleLayout &swizzle);

// Identify the supported swizzle parameters represented by a 2D layout.
SwizzleLayout AnalyzeSwizzleLayout(const tirx::Buffer &buffer,
                                   const Layout &layout);

} // namespace musa
} // namespace tl
} // namespace tvm

#endif
