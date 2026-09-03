/*!
 * \file tl/musa/layout/mp31_wmma_layouts.cc
 * \brief MP31 WMMA fragment layout helpers.
 */

#include "layout/layout.h"

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

Fragment MakeMP31WMMAFragmentC(const Array<PrimExpr> &buffer_shape, int warp_m,
                               int warp_n, const Array<PrimExpr> &inst_shape) {
  ICHECK_EQ(buffer_shape.size(), 2)
      << "MP31 WMMA C fragment expects a 2D buffer shape";
  const int block_m = GetConstInt(buffer_shape[0], "block_m");
  const int block_n = GetConstInt(buffer_shape[1], "block_n");
  ICHECK_EQ(inst_shape.size(), 3)
      << "MP31 WMMA inst_shape must contain [M, N, K]";
  const int inst_m = GetConstInt(inst_shape[0], "inst_m");
  const int inst_n = GetConstInt(inst_shape[1], "inst_n");
  const int inst_k = GetConstInt(inst_shape[2], "inst_k");
  ICHECK(inst_m > 0 && inst_n > 0 && inst_k > 0);
  ICHECK(block_m % warp_m == 0);
  ICHECK(block_n % warp_n == 0);
  ICHECK((block_m / warp_m) % inst_m == 0);
  ICHECK((block_n / warp_n) % inst_n == 0);
  IterVar i = MakeIterVar("i", 4);
  IterVar j = MakeIterVar("j", 8);
  IterVar rep = MakeIterVar("rep", 1);
  PrimExpr forward_thread = FloorDiv(j->var, 1) + 8 * i;
  PrimExpr index = FloorMod(j->var, 1);
  auto base_layout = Fragment({i, j}, {index}, forward_thread, rep);
  auto inst_layout = base_layout->Repeat({inst_m / 4, inst_n / 8}, false, true);
  auto warp_layout = inst_layout->Repeat({warp_m, warp_n}, true, false);
  return warp_layout->Repeat(
      {block_m / warp_m / inst_m, block_n / warp_n / inst_n}, false, false);
}

Fragment MakeMP31WMMAFragmentA(const Array<PrimExpr> &buffer_shape, int warp_m,
                               int warp_n, int element_size, bool transposed,
                               const Array<PrimExpr> &inst_shape) {
  ICHECK_EQ(buffer_shape.size(), 2)
      << "MP31 WMMA A fragment expects a 2D buffer shape";
  const int block_m = GetConstInt(buffer_shape[transposed ? 1 : 0], "block_m");
  const int block_k = GetConstInt(buffer_shape[transposed ? 0 : 1], "block_k");
  ICHECK_EQ(inst_shape.size(), 3)
      << "MP31 WMMA inst_shape must contain [M, N, K]";
  const int inst_m = GetConstInt(inst_shape[0], "inst_m");
  const int inst_k = GetConstInt(inst_shape[2], "inst_k");
  ICHECK(inst_m > 0 && inst_k > 0);
  ICHECK_EQ(32 % element_size, 0);
  const int num_reg = 32 / element_size;
  IterVar i = MakeIterVar("i", 1);
  IterVar j = MakeIterVar("j", 1);
  IterVar rep = MakeIterVar("rep", 1);
  auto base_layout = Fragment({i, j}, {Integer(0)}, Integer(0), rep);
  if (transposed) {
    auto reg_layout = base_layout->Repeat({1, num_reg}, false, true);
    auto tile_layout =
        reg_layout->Repeat({4 * num_reg, 8 / num_reg}, true, true);
    auto inst_layout =
        tile_layout->Repeat({inst_k / num_reg / 4, inst_m / 8}, false, true);
    auto warp_layout =
        inst_layout->Repeat({1, warp_m}, true, false)->Replicate(warp_n);
    return warp_layout->Repeat({block_k / inst_k, block_m / warp_m / inst_m},
                               false, true);
  }
  auto reg_layout = base_layout->Repeat({1, num_reg}, false, true);
  auto tile_layout = reg_layout->Repeat({8, 4}, true, true);
  auto inst_layout =
      tile_layout->Repeat({inst_m / 8, inst_k / num_reg / 4}, false, true);
  auto warp_layout =
      inst_layout->Repeat({warp_m, 1}, true, false)->Replicate(warp_n);
  return warp_layout->Repeat({block_m / warp_m / inst_m, block_k / inst_k},
                             false, false);
}

Fragment MakeMP31WMMAFragmentB(const Array<PrimExpr> &buffer_shape, int warp_m,
                               int warp_n, int element_size, bool transposed,
                               const Array<PrimExpr> &inst_shape) {
  ICHECK_EQ(buffer_shape.size(), 2)
      << "MP31 WMMA B fragment expects a 2D buffer shape";
  const int block_n = GetConstInt(buffer_shape[transposed ? 0 : 1], "block_n");
  const int block_k = GetConstInt(buffer_shape[transposed ? 1 : 0], "block_k");
  ICHECK_EQ(inst_shape.size(), 3)
      << "MP31 WMMA inst_shape must contain [M, N, K]";
  const int inst_n = GetConstInt(inst_shape[1], "inst_n");
  const int inst_k = GetConstInt(inst_shape[2], "inst_k");
  ICHECK(inst_n > 0 && inst_k > 0);
  ICHECK_EQ(32 % element_size, 0);
  const int num_reg = 32 / element_size;
  IterVar i = MakeIterVar("i", 1);
  IterVar j = MakeIterVar("j", 1);
  IterVar rep = MakeIterVar("rep", 1);
  auto base_layout = Fragment({i, j}, {Integer(0)}, Integer(0), rep);
  if (transposed) {
    auto reg_layout = base_layout->Repeat({num_reg, 1}, false, true);
    auto tile_layout =
        reg_layout->Repeat({8 / num_reg, 4 * num_reg}, true, false);
    auto inst_layout =
        tile_layout->Repeat({inst_n / 8, inst_k / num_reg / 4}, false, false);
    auto warp_layout =
        inst_layout->Replicate(warp_m)->Repeat({warp_n, 1}, true, false);
    return warp_layout->Repeat({block_n / warp_n / inst_n, block_k / inst_k},
                               false, false);
  }
  auto reg_layout = base_layout->Repeat({num_reg, 1}, false, true);
  auto tile_layout = reg_layout->Repeat({4, 8}, true, false);
  auto inst_layout =
      tile_layout->Repeat({inst_k / num_reg / 4, inst_n / 8}, false, false);
  auto warp_layout =
      inst_layout->Replicate(warp_m)->Repeat({1, warp_n}, true, false);
  return warp_layout->Repeat({block_k / inst_k, block_n / warp_n / inst_n},
                             false, true);
}

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("tl.make_mp31_wmma_fragment_a",
           [](Array<PrimExpr> buffer_shape, int warp_m, int warp_n,
              int element_size, bool transposed, Array<PrimExpr> inst_shape) {
             return MakeMP31WMMAFragmentA(buffer_shape, warp_m, warp_n,
                                          element_size, transposed, inst_shape);
           })
      .def("tl.make_mp31_wmma_fragment_b",
           [](Array<PrimExpr> buffer_shape, int warp_m, int warp_n,
              int element_size, bool transposed, Array<PrimExpr> inst_shape) {
             return MakeMP31WMMAFragmentB(buffer_shape, warp_m, warp_n,
                                          element_size, transposed, inst_shape);
           })
      .def("tl.make_mp31_wmma_fragment_c", [](Array<PrimExpr> buffer_shape,
                                              int warp_m, int warp_n,
                                              Array<PrimExpr> inst_shape) {
        return MakeMP31WMMAFragmentC(buffer_shape, warp_m, warp_n, inst_shape);
      });
}

} // namespace musa
} // namespace tl
} // namespace tvm
