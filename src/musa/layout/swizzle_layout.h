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

// Swizzle the final two dimensions of a row-major buffer. MP31 SQMMA operands
// whose continuous dimension exceeds 256 bytes are stored as independent
// 256-byte panels so every hardware descriptor has a representable stride.
Layout MakeSwizzleLayout(const tirx::Buffer &buffer,
                         const SwizzleLayout &swizzle);

// Identify the supported swizzle parameters represented by a layout.
SwizzleLayout AnalyzeSwizzleLayout(const tirx::Buffer &buffer,
                                   const Layout &layout);

} // namespace musa
} // namespace tl
} // namespace tvm

#endif
