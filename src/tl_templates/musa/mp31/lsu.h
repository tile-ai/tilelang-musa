#pragma once

#include <cstdint>

namespace tl {

TL_DEVICE uint4 load_global_128_peer_robust(const void *ptr) {
  // MTCC has no public vector peer-load wrapper with these cache controls.
  // The PH1/MP31 instruction form is therefore kept in the MP31 template.
  i4 packed;
  asm volatile("LSU.LD.B128 %0, %1, _, 16, 1, 1, inner_persist=4, "
               "outer_persist=2, chrnt=l1, slc=new, persist=0, "
               "stride_add_first=0"
               : "=R"(packed)
               : "R"(ptr));
  return *reinterpret_cast<const uint4 *>(&packed);
}

TL_DEVICE void store_global_128_peer_streaming(void *ptr, uint4 value) {
  // MTCC has no public vector peer-store wrapper with these cache controls.
  // The PH1/MP31 instruction form is therefore kept in the MP31 template.
  i4 packed = *reinterpret_cast<const i4 *>(&value);
  asm volatile("LSU.ST.B128 %0, %1, _, 16, 1, 1, inner_persist=4, "
               "outer_persist=2, chrnt=l2_l3, slc=byp, persist=0, "
               "stride_add_first=0"
               :
               : "R"(packed), "R"(ptr)
               : "memory");
}

TL_DEVICE void peer_release_fence() {
  asm volatile("LSU.BAR.SLC.NEW" ::: "memory");
  __syncthreads();
  if (static_cast<int>(threadIdx.x) == 0) {
    const unsigned long long zero = 0;
    asm volatile("LSU.IDF.SLC.BYPASS %0" : : "R"(zero) : "memory");
  }
  __syncthreads();
}

TL_DEVICE void peer_signal_store(void *target, int value) {
  asm volatile("LSU.ST.VOLATILE.B32 %0, %1, _, 4, 1, 1, "
               "stride_add_first=0"
               :
               : "R"(value), "R"(target)
               : "memory");
}

TL_DEVICE void peer_signal_store(int64_t base, int index, int value) {
  const uint64_t address = static_cast<uint64_t>(base) +
                           static_cast<uint64_t>(index) * sizeof(int);
  peer_signal_store(reinterpret_cast<void *>(address), value);
}

} // namespace tl
