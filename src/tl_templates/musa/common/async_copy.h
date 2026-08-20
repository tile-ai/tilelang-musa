#pragma once

namespace tl {

TL_DEVICE void cp_async_commit() {}

template <int N> TL_DEVICE void cp_async_wait() {
  // MTCC has no corresponding MUSA C API; this must use the builtin.
  __musa_memcpy_g2s_wait();
}

template <int N>
TL_DEVICE void cp_async_gs(void const *const smem_addr,
                           void const *const global_ptr) {
  // MTCC has no corresponding MUSA C API; this must use the builtin.
  __musa_memcpy_g2s((void _AS3 *)smem_addr, (void const _AS1 *)global_ptr, N,
                    0);
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *const global_ptr,
                                       bool condition) {
  if (condition) {
    cp_async_gs<N>(smem_addr, global_ptr);
  } else {
    // MUSA's g2s builtin has no CUDA-style source-size operand for
    // zero-filling a predicated cp.async.  Preserve cp.async's semantics by
    // writing exactly the destination transfer width.  In particular, the
    // 4-byte case must not use uint4, which would clobber adjacent shared data.
    void *dst = const_cast<void *>(smem_addr);
    if constexpr (N == 16) {
      *reinterpret_cast<uint4 *>(dst) = uint4{};
    } else if constexpr (N == 8) {
      *reinterpret_cast<uint2 *>(dst) = uint2{};
    } else {
      static_assert(N == 4, "MUSA cp.async supports 4, 8, or 16 bytes");
      *reinterpret_cast<uint32_t *>(dst) = 0;
    }
  }
}

} // namespace tl
