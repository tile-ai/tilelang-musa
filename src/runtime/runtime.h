/*!
 * \file tl/runtime/runtime.h
 * \brief Runtime functions.
 *
 */

#ifndef TVM_TL_RUNTIME_RUNTIME_H_
#define TVM_TL_RUNTIME_RUNTIME_H_

namespace tvm {
namespace tl {

#if (defined(CUDA_MAJOR_VERSION) && (CUDA_MAJOR_VERSION >= 12)) ||             \
    defined(MUSA_MAJOR_VERSION)
constexpr const char *tvm_tensormap_create_tiled =
    "__tvm_tensormap_create_tiled";
constexpr const char *tvm_tensormap_create_im2col =
    "__tvm_tensormap_create_im2col";
constexpr const char *tvm_musa_tensormap_create_tiled = "__tvm_musa_tensormap_create_tiled";
constexpr const char *tvm_musa_tensormap_create_im2col = "__tvm_musa_tensormap_create_im2col";
constexpr const char *tvm_musa_im2col_conv_param_create =
    "__tvm_musa_im2col_conv_param_create";
constexpr const char *tvm_musa_im2col_output_dim_create =
    "__tvm_musa_im2col_output_dim_create";
#endif // (defined(CUDA_MAJOR_VERSION) && (CUDA_MAJOR_VERSION >= 12)) ||
       // defined(MUSA_MAJOR_VERSION)

// CUDA stream access policy window helpers
constexpr const char *tvm_cuda_stream_set_access_policy_window =
    "__tvm_cuda_stream_set_access_policy_window";
constexpr const char *tvm_cuda_stream_reset_access_policy_window =
    "__tvm_cuda_stream_reset_access_policy_window";

// MUSA stream access policy window helpers
constexpr const char *tvm_musa_stream_set_access_policy_window =
    "__tvm_musa_stream_set_access_policy_window";
constexpr const char *tvm_musa_stream_reset_access_policy_window =
    "__tvm_musa_stream_reset_access_policy_window";

// Construct the 128-bit MUSA robust descriptor passed to device kernels.
constexpr const char *tvm_musa_robust_descriptor_create =
    "__tvm_musa_robust_descriptor_create";
} // namespace tl
} // namespace tvm

#endif //  TVM_TL_RUNTIME_RUNTIME_H_
