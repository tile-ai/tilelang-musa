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
  __musa_memcpy_g2s((void _AS3 *)smem_addr,
                    (void const _AS1 *)global_ptr, N, 0);
}

template <int N>
TL_DEVICE void cp_async_gs_conditional(void const *const smem_addr,
                                       void const *const global_ptr,
                                       bool condition) {
  if (condition) {
    cp_async_gs<N>(smem_addr, global_ptr);
  }
}

} // namespace tl
