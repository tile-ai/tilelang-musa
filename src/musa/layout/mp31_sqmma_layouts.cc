/*!
 * \file tl/musa/layout/mp31_sqmma_layouts.cc
 * \brief MP31 SQMMA layout helpers.
 */

#include "layout/layout.h"
#include "musa/layout/swizzle_layout.h"

#include <tvm/ffi/reflection/registry.h>
#include <tvm/runtime/logging.h>
#include <tvm/tirx/op.h>

namespace tvm {
namespace tl {
namespace musa {
namespace {

using namespace ffi;
using namespace tirx;

int GetConstInt(const PrimExpr &value, const char *name) {
  const int64_t *const_value = as_const_int(value);
  ICHECK(const_value != nullptr) << name << " must be a constant integer";
  return static_cast<int>(*const_value);
}

Layout MakeMP31SQMMASharedAB(const tirx::Buffer &buffer, bool k_major) {
  ICHECK(buffer.defined()) << "MP31 SQMMA shared layout expects a buffer";
  ICHECK_EQ(buffer->shape.size(), 2)
      << "MP31 SQMMA shared layout expects a 2D buffer, got rank="
      << buffer->shape.size();

  const int element_size = buffer->dtype.bits();
  ICHECK(element_size == 8 || element_size == 16 || element_size == 32)
      << "Unsupported MP31 SQMMA shared layout with element_size="
      << element_size;

  const SwizzleGranularity sg =
      k_major ? SwizzleGranularity::k16B
              : static_cast<SwizzleGranularity>(element_size * 2);
  const SwizzleLayout swizzle{sg, SwizzleStride::k256B, SwizzleLine::k256B};
  return MakeSwizzleLayout(buffer, swizzle);
}

Fragment MakeMP31SQMMAFragmentC(const Array<PrimExpr> &buffer_shape, int warp_m,
                                int warp_n, const Array<PrimExpr> &inst_shape) {
  ICHECK_EQ(buffer_shape.size(), 2)
      << "MP31 SQMMA accumulator layout expects a 2D shape, got rank="
      << buffer_shape.size();
  ICHECK_EQ(inst_shape.size(), 3)
      << "MP31 SQMMA inst_shape must contain [M, N, K]";
  ICHECK_GT(warp_m, 0);
  ICHECK_GT(warp_n, 0);
  ICHECK_EQ(warp_m % 4, 0)
      << "MP31 SQMMA requires warp_m to be a multiple of 4";

  const int block_m = GetConstInt(buffer_shape[0], "block_m");
  const int block_n = GetConstInt(buffer_shape[1], "block_n");
  const int inst_m = GetConstInt(inst_shape[0], "inst_m");
  const int inst_n = GetConstInt(inst_shape[1], "inst_n");
  const int inst_k = GetConstInt(inst_shape[2], "inst_k");
  ICHECK_GT(inst_m, 0);
  ICHECK_GT(inst_n, 0);
  ICHECK_GT(inst_k, 0);
  ICHECK_EQ(block_m % warp_m, 0);
  ICHECK_EQ(block_n % warp_n, 0);

  const int warp_tile_m = block_m / warp_m;
  const int warp_tile_n = block_n / warp_n;
  ICHECK_EQ((warp_tile_m * 4) % inst_m, 0)
      << "MP31 SQMMA squad M tile must align to inst_m";
  ICHECK_EQ(warp_tile_n % inst_n, 0)
      << "MP31 SQMMA squad N tile must align to inst_n";

  IterVar i = MakeIterVar("i", 4);
  IterVar j = MakeIterVar("j", 8);
  IterVar rep = MakeIterVar("rep", 1);
  auto atom =
      Fragment({i, j}, {FloorMod(j->var, 1)}, FloorDiv(j->var, 1) + 8 * i, rep);
  auto base_layout = atom->Repeat({4, 1}, true, false);
  auto instruction_layout =
      base_layout->Repeat({inst_m / 16, inst_n / 8}, false, true);
  auto squad_layout = instruction_layout->Repeat(
      {warp_tile_m * 4 / inst_m, warp_tile_n / inst_n}, false, false);
  auto block_layout = squad_layout->Repeat({warp_m / 4, warp_n}, true, false);

  return block_layout;
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.make_mp31_sqmma_shared_ab",
           [](tirx::Buffer buffer, bool k_major) {
             return MakeMP31SQMMASharedAB(buffer, k_major);
           })
      .def("tl.make_mp31_sqmma_fragment_c",
           [](Array<PrimExpr> shape, int warp_m, int warp_n,
              Array<PrimExpr> inst_shape) {
             return MakeMP31SQMMAFragmentC(shape, warp_m, warp_n, inst_shape);
           });
}

} // namespace musa
} // namespace tl
} // namespace tvm
