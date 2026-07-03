#pragma once

#include "common.h"

struct alignas(uint64_t) Barrier {
  uint64_t storage;

  TL_DEVICE operator uint32_t() { return cast_smem_ptr_to_uint(&storage); }
  TL_DEVICE operator uint32_t() const {
    return cast_smem_ptr_to_uint(&storage);
  }
};
static_assert(sizeof(Barrier) == sizeof(uint64_t));

namespace tl {
TL_DEVICE void
mbarrier_init(uint32_t barrier_id, // 32 bits user-managed barrier's id
              uint32_t warp_count =
                  1, // Warp count expected to arrive/wait on this barrier
              uint32_t init_phase = 0) { // Init phase on this barrier
  // #if __MUSA_ARCH__ >= 310
  __musa_async_init_arrival(barrier_id, warp_count, init_phase);
  // #endif
}

TL_DEVICE void mbarrier_wait(uint32_t barrier_id, int phase_bit) {
  // #if __MUSA_ARCH__ >= 310
  __musa_async_wait(barrier_id, phase_bit);
  // #endif
}

TL_DEVICE uint32_t mbarrier_arrive(uint32_t barrier_id) {
  // #if __MUSA_ARCH__ >= 310
  return __musa_async_arrive(barrier_id);
  // #endif
}

TL_DEVICE void mbarrier_arrive_expect_tx(uint32_t barrier_id,
                                         uint32_t transaction_bytes) {
#if defined(__MUSA_ARCH_LIST__) && (__MUSA_ARCH_LIST__ >= 420)
  __musa_async_arrive_add_trans(barrier_id, transaction_bytes);
#else
  __musa_async_add_trans(barrier_id, transaction_bytes);
  __musa_async_arrive(barrier_id);
#endif
}

TL_DEVICE void fence_proxy_async() { asm volatile("" : : : "memory"); }

TL_DEVICE void fence_barrier_init() { asm volatile("" : : : "memory"); }

TL_DEVICE void tma_store_arrive() { __musa_tme_store_commit(); }

template <int Count> TL_DEVICE void tma_store_wait() {
  static_assert(Count == 0,
                "MUSA tma_store_wait currently supports Count == 0");
  __musa_tme_store_read_wait();
}

} // namespace tl
