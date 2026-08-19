#pragma once

#include <cstdint>
#include <musa_runtime.h>
#include <type_traits>

#define TL_DEVICE __forceinline__ __device__

namespace tl {

template <typename T> struct normalize_atomic_type {
  using type = T;
};

template <> struct normalize_atomic_type<int64_t> {
  using type = long long;
};

template <typename T> TL_DEVICE unsigned short BitCastToU16(T value) {
  static_assert(sizeof(T) == sizeof(unsigned short),
                "invalid atomic bitcast size");
  union {
    T value;
    unsigned short bits;
  } caster;
  caster.value = value;
  return caster.bits;
}

template <typename T> TL_DEVICE T BitCastFromU16(unsigned short bits) {
  static_assert(sizeof(T) == sizeof(unsigned short),
                "invalid atomic bitcast size");
  union {
    unsigned short bits;
    T value;
  } caster;
  caster.bits = bits;
  return caster.value;
}

template <typename T> TL_DEVICE int BitCastToI32(T value) {
  static_assert(sizeof(T) == sizeof(int), "invalid atomic bitcast size");
  union {
    T value;
    int bits;
  } caster;
  caster.value = value;
  return caster.bits;
}

template <typename T> TL_DEVICE T BitCastFromI32(int bits) {
  static_assert(sizeof(T) == sizeof(int), "invalid atomic bitcast size");
  union {
    int bits;
    T value;
  } caster;
  caster.bits = bits;
  return caster.value;
}

template <typename T> TL_DEVICE unsigned long long BitCastToU64(T value) {
  static_assert(sizeof(T) == sizeof(unsigned long long),
                "invalid atomic bitcast size");
  union {
    T value;
    unsigned long long bits;
  } caster;
  caster.value = value;
  return caster.bits;
}

template <typename T> TL_DEVICE T BitCastFromU64(unsigned long long bits) {
  static_assert(sizeof(T) == sizeof(unsigned long long),
                "invalid atomic bitcast size");
  union {
    unsigned long long bits;
    T value;
  } caster;
  caster.bits = bits;
  return caster.value;
}

template <typename T> struct is_16bit_float_type : std::false_type {};

#if defined(TL_MUSA_ENABLE_FP16)
template <> struct is_16bit_float_type<half> : std::true_type {};
#endif

#if defined(TL_MUSA_ENABLE_BF16)
template <> struct is_16bit_float_type<mt_bfloat16> : std::true_type {};
#endif

TL_DEVICE void AtomicFenceForLoad(int memory_order) {
  if (memory_order == 1 || memory_order == 2 || memory_order == 4 ||
      memory_order == 5) {
    __threadfence();
  }
}

TL_DEVICE void AtomicFenceForStore(int memory_order) {
  if (memory_order == 3 || memory_order == 5) {
    __threadfence();
  }
}

template <typename T> TL_DEVICE T atomic_cast(T value) { return value; }

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAdd16BitRet(T1 *address, T2 value) {
  using RawT = std::remove_cv_t<T1>;
  RawT casted = static_cast<RawT>(value);
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310
  unsigned short *address_as_u16 = reinterpret_cast<unsigned short *>(address);
  unsigned short old = *address_as_u16;
  unsigned short assumed;
  do {
    assumed = old;
    RawT next =
        static_cast<RawT>(static_cast<float>(BitCastFromU16<RawT>(assumed)) +
                          static_cast<float>(casted));
    old = atomicCAS(address_as_u16, assumed, BitCastToU16(next));
  } while (assumed != old);
  return BitCastFromU16<RawT>(old);
#else
  unsigned int *address_as_ui;
  if (__musa_isspacep_shared(address)) {
    address_as_ui = reinterpret_cast<unsigned int *>(
        __musa_ptr_gen_to_shared(reinterpret_cast<void *>(address)));
  } else {
    address_as_ui = reinterpret_cast<unsigned int *>(
        __musa_ptr_gen_to_global(reinterpret_cast<void *>(address)));
  }
  address_as_ui = reinterpret_cast<unsigned int *>(
      reinterpret_cast<char *>(address_as_ui) -
      (reinterpret_cast<size_t>(address_as_ui) & 2));
  unsigned int old = *address_as_ui;
  unsigned int assumed;
  unsigned short old_u16;
  do {
    assumed = old;
    old_u16 =
        (reinterpret_cast<size_t>(address) & 2) ? (old >> 16) : (old & 0xffff);
    RawT next =
        static_cast<RawT>(static_cast<float>(BitCastFromU16<RawT>(old_u16)) +
                          static_cast<float>(casted));
    unsigned short next_u16 = BitCastToU16(next);
    unsigned int next_ui =
        (reinterpret_cast<size_t>(address) & 2)
            ? (old & 0xffff) | (static_cast<unsigned int>(next_u16) << 16)
            : (old & 0xffff0000) | static_cast<unsigned int>(next_u16);
    old = atomicCAS(address_as_ui, assumed, next_ui);
  } while (assumed != old);
  old_u16 =
      (reinterpret_cast<size_t>(address) & 2) ? (old >> 16) : (old & 0xffff);
  return BitCastFromU16<RawT>(old_u16);
#endif
}

template <typename T1, typename T2>
TL_DEVICE void AtomicAdd(T1 *address, T2 value, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  if constexpr (is_16bit_float_type<RawT>::value) {
    AtomicAdd16BitRet(address, value);
  } else if constexpr (std::is_same<RawT, int64_t>::value) {
    atomicAdd(reinterpret_cast<unsigned long long *>(address),
              BitCastToU64(static_cast<RawT>(value)));
  } else {
    using AtomicT = typename normalize_atomic_type<RawT>::type;
    atomicAdd(reinterpret_cast<AtomicT *>(address),
              static_cast<AtomicT>(value));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicAddRet(T1 *address, T2 value, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  if constexpr (is_16bit_float_type<RawT>::value) {
    return static_cast<T1>(AtomicAdd16BitRet(address, value));
  } else if constexpr (std::is_same<RawT, int64_t>::value) {
    auto old = atomicAdd(reinterpret_cast<unsigned long long *>(address),
                         BitCastToU64(static_cast<RawT>(value)));
    return static_cast<T1>(BitCastFromU64<RawT>(old));
  } else {
    using AtomicT = typename normalize_atomic_type<RawT>::type;
    return static_cast<T1>(atomicAdd(reinterpret_cast<AtomicT *>(address),
                                     static_cast<AtomicT>(value)));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMax(T1 *address, T2 value, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  RawT casted = static_cast<RawT>(value);
  if constexpr (std::is_same<RawT, float>::value) {
    int *address_as_i32 = reinterpret_cast<int *>(address);
    int old = *address_as_i32;
    while (casted > BitCastFromI32<float>(old)) {
      int assumed = old;
      old = atomicCAS(address_as_i32, assumed, BitCastToI32(casted));
      if (assumed == old) {
        break;
      }
    }
  } else if constexpr (is_16bit_float_type<RawT>::value) {
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310
    unsigned short *address_as_u16 =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old = *address_as_u16;
    while (static_cast<float>(casted) >
           static_cast<float>(BitCastFromU16<RawT>(old))) {
      unsigned short assumed = old;
      old = atomicCAS(address_as_u16, assumed, val_as_u16);
      if (assumed == old) {
        break;
      }
    }
#else
    unsigned int *address_as_ui;
    if (__musa_isspacep_shared(address)) {
      address_as_ui = reinterpret_cast<unsigned int *>(
          __musa_ptr_gen_to_shared(reinterpret_cast<void *>(address)));
    } else {
      address_as_ui = reinterpret_cast<unsigned int *>(
          __musa_ptr_gen_to_global(reinterpret_cast<void *>(address)));
    }
    address_as_ui = reinterpret_cast<unsigned int *>(
        reinterpret_cast<char *>(address_as_ui) -
        (reinterpret_cast<size_t>(address_as_ui) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    unsigned short val_as_u16 = BitCastToU16(casted);
    do {
      assumed = old;
      unsigned short old_u16 = (reinterpret_cast<size_t>(address) & 2)
                                   ? (old >> 16)
                                   : (old & 0xffff);
      if (static_cast<float>(casted) <=
          static_cast<float>(BitCastFromU16<RawT>(old_u16))) {
        break;
      }
      unsigned int next_ui =
          (reinterpret_cast<size_t>(address) & 2)
              ? (old & 0xffff) | (static_cast<unsigned int>(val_as_u16) << 16)
              : (old & 0xffff0000) | static_cast<unsigned int>(val_as_u16);
      old = atomicCAS(address_as_ui, assumed, next_ui);
    } while (assumed != old);
#endif
  } else {
    using AtomicT = typename normalize_atomic_type<RawT>::type;
    atomicMax(reinterpret_cast<AtomicT *>(address),
              static_cast<AtomicT>(casted));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMaxRet(T1 *address, T2 value, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  RawT casted = static_cast<RawT>(value);
  if constexpr (std::is_same<RawT, float>::value) {
    int *address_as_i32 = reinterpret_cast<int *>(address);
    int old = *address_as_i32;
    while (casted > BitCastFromI32<float>(old)) {
      int assumed = old;
      old = atomicCAS(address_as_i32, assumed, BitCastToI32(casted));
      if (assumed == old) {
        break;
      }
    }
    return static_cast<T1>(BitCastFromI32<float>(old));
  } else if constexpr (is_16bit_float_type<RawT>::value) {
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310
    unsigned short *address_as_u16 =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old = *address_as_u16;
    while (static_cast<float>(casted) >
           static_cast<float>(BitCastFromU16<RawT>(old))) {
      unsigned short assumed = old;
      old = atomicCAS(address_as_u16, assumed, val_as_u16);
      if (assumed == old) {
        break;
      }
    }
    return static_cast<T1>(BitCastFromU16<RawT>(old));
#else
    unsigned int *address_as_ui;
    if (__musa_isspacep_shared(address)) {
      address_as_ui = reinterpret_cast<unsigned int *>(
          __musa_ptr_gen_to_shared(reinterpret_cast<void *>(address)));
    } else {
      address_as_ui = reinterpret_cast<unsigned int *>(
          __musa_ptr_gen_to_global(reinterpret_cast<void *>(address)));
    }
    address_as_ui = reinterpret_cast<unsigned int *>(
        reinterpret_cast<char *>(address_as_ui) -
        (reinterpret_cast<size_t>(address_as_ui) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old_u16;
    do {
      assumed = old;
      old_u16 = (reinterpret_cast<size_t>(address) & 2) ? (old >> 16)
                                                        : (old & 0xffff);
      if (static_cast<float>(casted) <=
          static_cast<float>(BitCastFromU16<RawT>(old_u16))) {
        break;
      }
      unsigned int next_ui =
          (reinterpret_cast<size_t>(address) & 2)
              ? (old & 0xffff) | (static_cast<unsigned int>(val_as_u16) << 16)
              : (old & 0xffff0000) | static_cast<unsigned int>(val_as_u16);
      old = atomicCAS(address_as_ui, assumed, next_ui);
    } while (assumed != old);
    old_u16 =
        (reinterpret_cast<size_t>(address) & 2) ? (old >> 16) : (old & 0xffff);
    return static_cast<T1>(BitCastFromU16<RawT>(old_u16));
#endif
  } else {
    using AtomicT = typename normalize_atomic_type<RawT>::type;
    return static_cast<T1>(atomicMax(reinterpret_cast<AtomicT *>(address),
                                     static_cast<AtomicT>(casted)));
  }
}

template <typename T1, typename T2>
TL_DEVICE void AtomicMin(T1 *address, T2 value, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  RawT casted = static_cast<RawT>(value);
  if constexpr (std::is_same<RawT, float>::value) {
    int *address_as_i32 = reinterpret_cast<int *>(address);
    int old = *address_as_i32;
    while (casted < BitCastFromI32<float>(old)) {
      int assumed = old;
      old = atomicCAS(address_as_i32, assumed, BitCastToI32(casted));
      if (assumed == old) {
        break;
      }
    }
  } else if constexpr (is_16bit_float_type<RawT>::value) {
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310
    unsigned short *address_as_u16 =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old = *address_as_u16;
    while (static_cast<float>(casted) <
           static_cast<float>(BitCastFromU16<RawT>(old))) {
      unsigned short assumed = old;
      old = atomicCAS(address_as_u16, assumed, val_as_u16);
      if (assumed == old) {
        break;
      }
    }
#else
    unsigned int *address_as_ui;
    if (__musa_isspacep_shared(address)) {
      address_as_ui = reinterpret_cast<unsigned int *>(
          __musa_ptr_gen_to_shared(reinterpret_cast<void *>(address)));
    } else {
      address_as_ui = reinterpret_cast<unsigned int *>(
          __musa_ptr_gen_to_global(reinterpret_cast<void *>(address)));
    }
    address_as_ui = reinterpret_cast<unsigned int *>(
        reinterpret_cast<char *>(address_as_ui) -
        (reinterpret_cast<size_t>(address_as_ui) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    unsigned short val_as_u16 = BitCastToU16(casted);
    do {
      assumed = old;
      unsigned short old_u16 = (reinterpret_cast<size_t>(address) & 2)
                                   ? (old >> 16)
                                   : (old & 0xffff);
      if (static_cast<float>(casted) >=
          static_cast<float>(BitCastFromU16<RawT>(old_u16))) {
        break;
      }
      unsigned int next_ui =
          (reinterpret_cast<size_t>(address) & 2)
              ? (old & 0xffff) | (static_cast<unsigned int>(val_as_u16) << 16)
              : (old & 0xffff0000) | static_cast<unsigned int>(val_as_u16);
      old = atomicCAS(address_as_ui, assumed, next_ui);
    } while (assumed != old);
#endif
  } else {
    using AtomicT = typename normalize_atomic_type<RawT>::type;
    atomicMin(reinterpret_cast<AtomicT *>(address),
              static_cast<AtomicT>(casted));
  }
}

template <typename T1, typename T2>
TL_DEVICE T1 AtomicMinRet(T1 *address, T2 value, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  RawT casted = static_cast<RawT>(value);
  if constexpr (std::is_same<RawT, float>::value) {
    int *address_as_i32 = reinterpret_cast<int *>(address);
    int old = *address_as_i32;
    while (casted < BitCastFromI32<float>(old)) {
      int assumed = old;
      old = atomicCAS(address_as_i32, assumed, BitCastToI32(casted));
      if (assumed == old) {
        break;
      }
    }
    return static_cast<T1>(BitCastFromI32<float>(old));
  } else if constexpr (is_16bit_float_type<RawT>::value) {
#if defined(__MUSA_ARCH__) && __MUSA_ARCH__ >= 310
    unsigned short *address_as_u16 =
        reinterpret_cast<unsigned short *>(address);
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old = *address_as_u16;
    while (static_cast<float>(casted) <
           static_cast<float>(BitCastFromU16<RawT>(old))) {
      unsigned short assumed = old;
      old = atomicCAS(address_as_u16, assumed, val_as_u16);
      if (assumed == old) {
        break;
      }
    }
    return static_cast<T1>(BitCastFromU16<RawT>(old));
#else
    unsigned int *address_as_ui;
    if (__musa_isspacep_shared(address)) {
      address_as_ui = reinterpret_cast<unsigned int *>(
          __musa_ptr_gen_to_shared(reinterpret_cast<void *>(address)));
    } else {
      address_as_ui = reinterpret_cast<unsigned int *>(
          __musa_ptr_gen_to_global(reinterpret_cast<void *>(address)));
    }
    address_as_ui = reinterpret_cast<unsigned int *>(
        reinterpret_cast<char *>(address_as_ui) -
        (reinterpret_cast<size_t>(address_as_ui) & 2));
    unsigned int old = *address_as_ui;
    unsigned int assumed;
    unsigned short val_as_u16 = BitCastToU16(casted);
    unsigned short old_u16;
    do {
      assumed = old;
      old_u16 = (reinterpret_cast<size_t>(address) & 2) ? (old >> 16)
                                                        : (old & 0xffff);
      if (static_cast<float>(casted) >=
          static_cast<float>(BitCastFromU16<RawT>(old_u16))) {
        break;
      }
      unsigned int next_ui =
          (reinterpret_cast<size_t>(address) & 2)
              ? (old & 0xffff) | (static_cast<unsigned int>(val_as_u16) << 16)
              : (old & 0xffff0000) | static_cast<unsigned int>(val_as_u16);
      old = atomicCAS(address_as_ui, assumed, next_ui);
    } while (assumed != old);
    old_u16 =
        (reinterpret_cast<size_t>(address) & 2) ? (old >> 16) : (old & 0xffff);
    return static_cast<T1>(BitCastFromU16<RawT>(old_u16));
#endif
  } else {
    using AtomicT = typename normalize_atomic_type<RawT>::type;
    return static_cast<T1>(atomicMin(reinterpret_cast<AtomicT *>(address),
                                     static_cast<AtomicT>(casted)));
  }
}

template <typename T> TL_DEVICE T AtomicLoad(T *address, int memory_order) {
  volatile T *volatile_address = reinterpret_cast<volatile T *>(address);
  T value = *volatile_address;
  AtomicFenceForLoad(memory_order);
  return value;
}

template <typename T1, typename T2>
TL_DEVICE void AtomicStore(T1 *address, T2 value, int memory_order) {
  AtomicFenceForStore(memory_order);
  volatile T1 *volatile_address = reinterpret_cast<volatile T1 *>(address);
  *volatile_address = atomic_cast<T1>(value);
}

template <typename T1, typename T2>
TL_DEVICE void AtomicOr(T1 *address, T2 value, int memory_order = 0) {
  (void)memory_order;
  using RawT = std::remove_cv_t<T1>;
  using AtomicT = typename normalize_atomic_type<RawT>::type;
  atomicOr(reinterpret_cast<AtomicT *>(address),
           static_cast<AtomicT>(atomic_cast<RawT>(value)));
}

} // namespace tl
