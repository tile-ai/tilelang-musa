/*!
 * \file tl/backend/musa/op/gemm.h
 * \brief MUSA helpers for tl.gemm lowering and layout inference.
 */

#ifndef TVM_TL_BACKEND_MUSA_OP_GEMM_H_
#define TVM_TL_BACKEND_MUSA_OP_GEMM_H_

#include "layout/layout.h"
#include "op/gemm.h"

#include <array>
#include <optional>
#include <utility>

namespace tvm {
namespace tl {
namespace musa {

bool IsPH1SupportedFp8(DataType dtype);

int GetQY2MmaWarpSize(
    const std::optional<std::array<int, 3>> &mma_inst_shape = std::nullopt);

std::optional<std::array<int, 3>>
SelectQY2MmaInstShape(const Buffer &a, const Buffer &b, const Buffer &c, int M,
                      int N, int K, int block_size, Target target);

std::pair<int, int> ComputeQY2GemmWarpPartition(
    const GemmWarpPolicy &policy, int M, int N, int block_size, Target target,
    const std::optional<std::array<int, 3>> &mma_inst_shape = std::nullopt);

Layout MakeTransposedPH1SqmmaOperandLayout(int actual_rows, int actual_cols,
                                           int logical_rows, int logical_cols,
                                           int element_bits, bool k_inner);

} // namespace musa
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_MUSA_OP_GEMM_H_
