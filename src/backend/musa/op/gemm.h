/*!
 * \file tl/backend/musa/op/gemm.h
 * \brief MUSA helpers for tl.gemm lowering and layout inference.
 */

#ifndef TVM_TL_BACKEND_MUSA_OP_GEMM_H_
#define TVM_TL_BACKEND_MUSA_OP_GEMM_H_

#include "layout/layout.h"

namespace tvm {
namespace tl {
namespace musa {

bool IsPH1SupportedFp8(DataType dtype);

Layout MakeTransposedPH1SqmmaOperandLayout(int actual_rows, int actual_cols,
                                           int logical_rows, int logical_cols,
                                           int element_bits, bool k_inner);

} // namespace musa
} // namespace tl
} // namespace tvm

#endif // TVM_TL_BACKEND_MUSA_OP_GEMM_H_
