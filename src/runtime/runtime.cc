/*!
 * \file tl/runtime/runtime.h
 * \brief Runtime functions.
 *
 */

#include "runtime.h"

#if defined(MUSA_MAJOR_VERSION)
#include <musa.h>
#else
#include "backend/cuda/codegen/stubs/cuda.h"
#endif

#include <cstdint>
#include <sstream>
#include <vector>

#include <tvm/ffi/function.h>
#include <tvm/node/node.h>

namespace tvm {
namespace tl {

#if 1
// Thread-local storage for restoring the L2 persisting cache limit
static thread_local size_t __tl_prev_persisting_l2_cache_size = 0;
static thread_local bool __tl_prev_persisting_l2_cache_saved = false;
#endif

template <typename T> static std::string ArrayToStr(const T *ptr, size_t n) {
  std::stringstream ss;
  ss << "[";
  for (size_t i = 0; i < n; i++) {
    if (i > 0)
      ss << ", ";
    ss << ptr[i]; // NOLINT(clang-analyzer-security.ArrayBound)
  }
  ss << "]";
  return ss.str();
}

#if defined(CUDA_MAJOR_VERSION) && (CUDA_MAJOR_VERSION >= 12)

struct TensorMapArgs {
  CUtensorMap *map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void *globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t boxDim[5], elementStrides[5];
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  static TensorMapArgs Extract(PackedArgs args) {
    TensorMapArgs T;
    int idx = 0;
    ICHECK(args.size() >= 8);
    T.map = reinterpret_cast<CUtensorMap *>(args[idx++].cast<void *>());
    T.type = static_cast<CUtensorMapDataType>(args[idx++].cast<int64_t>());
    T.tensorRank = static_cast<cuuint32_t>(args[idx++].cast<int64_t>());
    T.globalAddress = args[idx++].cast<void *>();
    ICHECK(T.tensorRank >= 1 && T.tensorRank <= 5);
    ICHECK(args.size() == static_cast<int>(8 + T.tensorRank * 4));
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalDim[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalStride[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.boxDim[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.elementStrides[i] = args[idx++].cast<cuuint64_t>();
    }
    T.interleave =
        static_cast<CUtensorMapInterleave>(args[idx++].cast<int64_t>());
    T.swizzle = static_cast<CUtensorMapSwizzle>(args[idx++].cast<int64_t>());
    T.l2Promotion =
        static_cast<CUtensorMapL2promotion>(args[idx++].cast<int64_t>());
    T.oobFill =
        static_cast<CUtensorMapFloatOOBfill>(args[idx++].cast<int64_t>());
    return T;
  }

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << '\n'
       << "format         " << type << '\n'
       << "dim            " << tensorRank << '\n'
       << "gmem_address   " << globalAddress << '\n'
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << '\n'
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << '\n'
       << "boxDim         " << ArrayToStr(boxDim, tensorRank) << '\n'
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << '\n'
       << "interleave     " << interleave << '\n'
       << "swizzle        " << swizzle << '\n'
       << "l2Promotion    " << l2Promotion << '\n'
       << "oobFill        " << oobFill << '\n';
    return ss.str();
  }
};

// set device api
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Register using the canonical names defined in runtime.h
  refl::GlobalDef().def_packed(
      tl::tvm_tensormap_create_tiled, [](PackedArgs args, Any *ret) {
        TensorMapArgs T = TensorMapArgs::Extract(args);
        CUresult result = cuTensorMapEncodeTiled(
            T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim,
            T.globalStride + 1, T.boxDim, T.elementStrides, T.interleave,
            T.swizzle, T.l2Promotion, T.oobFill);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to initialize the TMA descriptor " << result
                    << '\n'
                    << T.ToDebugString();
        }
        *ret = static_cast<int>(result);
      });
}

struct TensorMapIm2ColArgs {
  CUtensorMap *map;
  CUtensorMapDataType type;
  cuuint32_t tensorRank;
  void *globalAddress;
  cuuint64_t globalDim[5], globalStride[5];
  cuuint32_t elementStrides[5];
  int pixelBoxLowerCorner[3], pixelBoxUpperCorner[3];
  cuuint32_t smem_box_channel, smem_box_pixel;
  CUtensorMapInterleave interleave;
  CUtensorMapSwizzle swizzle;
  CUtensorMapL2promotion l2Promotion;
  CUtensorMapFloatOOBfill oobFill;

  static TensorMapIm2ColArgs Extract(PackedArgs args) {
    TensorMapIm2ColArgs T;
    int idx = 0;
    ICHECK(args.size() >= 8);
    T.map = reinterpret_cast<CUtensorMap *>(args[idx++].cast<void *>());
    T.type = static_cast<CUtensorMapDataType>(args[idx++].cast<int64_t>());
    T.tensorRank = static_cast<cuuint32_t>(args[idx++].cast<int64_t>());
    T.globalAddress = args[idx++].cast<void *>();
    ICHECK(T.tensorRank >= 3 && T.tensorRank <= 5);
    ICHECK(args.size() == static_cast<int>(6 + T.tensorRank * 5));
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalDim[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.globalStride[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank; i++) {
      T.elementStrides[i] = args[idx++].cast<cuuint64_t>();
    }
    for (size_t i = 0; i < T.tensorRank - 2; i++) {
      T.pixelBoxLowerCorner[i] = args[idx++].cast<int>();
    }
    for (size_t i = 0; i < T.tensorRank - 2; i++) {
      T.pixelBoxUpperCorner[i] = args[idx++].cast<int>();
    }
    T.smem_box_pixel = args[idx++].cast<cuuint64_t>();
    T.smem_box_channel = args[idx++].cast<cuuint64_t>();
    T.interleave =
        static_cast<CUtensorMapInterleave>(args[idx++].cast<int64_t>());
    T.swizzle = static_cast<CUtensorMapSwizzle>(args[idx++].cast<int64_t>());
    T.l2Promotion =
        static_cast<CUtensorMapL2promotion>(args[idx++].cast<int64_t>());
    T.oobFill =
        static_cast<CUtensorMapFloatOOBfill>(args[idx++].cast<int64_t>());
    return T;
  }

  std::string ToDebugString() {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << map << '\n'
       << "format         " << type << '\n'
       << "dim            " << tensorRank << '\n'
       << "gmem_address   " << globalAddress << '\n'
       << "globalDim      " << ArrayToStr(globalDim, tensorRank) << '\n'
       << "globalStrides  " << ArrayToStr(globalStride, tensorRank) << '\n'
       << "smem_box_pixel " << smem_box_pixel << '\n'
       << "smem_box_channel " << smem_box_channel << '\n'
       << "pixelBoxLowerCorner  "
       << ArrayToStr(pixelBoxLowerCorner, tensorRank - 2) << '\n'
       << "pixelBoxUpperCorner  "
       << ArrayToStr(pixelBoxUpperCorner, tensorRank - 2) << '\n'
       << "elementStrides " << ArrayToStr(elementStrides, tensorRank) << '\n'
       << "interleave     " << interleave << '\n'
       << "swizzle        " << swizzle << '\n'
       << "l2Promotion    " << l2Promotion << '\n'
       << "oobFill        " << oobFill << '\n';
    return ss.str();
  }
};

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      tl::tvm_tensormap_create_im2col, [](PackedArgs args, Any *ret) {
        TensorMapIm2ColArgs T = TensorMapIm2ColArgs::Extract(args);
        CUresult result = cuTensorMapEncodeIm2col(
            T.map, T.type, T.tensorRank, T.globalAddress, T.globalDim,
            T.globalStride + 1, T.pixelBoxLowerCorner, T.pixelBoxUpperCorner,
            T.smem_box_channel, T.smem_box_pixel, T.elementStrides,
            T.interleave, T.swizzle, T.l2Promotion, T.oobFill);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to initialize the TMA descriptor " << result
                    << '\n'
                    << T.ToDebugString();
        }
        *ret = static_cast<int>(result);
      });
}

#endif // defined(CUDA_MAJOR_VERSION) && (CUDA_MAJOR_VERSION >= 12)

#if defined(MUSA_MAJOR_VERSION)
static uint64_t PtrModulo(const void *ptr, uint64_t align) {
  return reinterpret_cast<uintptr_t>(ptr) % align;
}

static const char *
TensorDescriptorDataTypeToString(MUtensorDescriptorDataType type) {
  switch (type) {
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT16:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT16";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT16:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT16";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_TFLOAT32:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_TFLOAT32";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT64:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT64";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT64:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT64";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT64:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT64";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32_FTZ:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32_FTZ";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_TFLOAT32_FTZ:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_TFLOAT32_FTZ";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32_TFLOAT32_RNE:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32_TFLOAT32_RNE";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32_FTZ_TFLOAT32_RNE:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32_FTZ_TFLOAT32_RNE";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_16b4:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_16b4";
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_16b4_b8:
    return "MU_TENSOR_DESCRIPTOR_DATA_TYPE_16b4_b8";
  default:
    return "<unknown MUtensorDescriptorDataType>";
  }
}

static const char *
TensorDescriptorInterleaveToString(MUtensorDescriptorInterleave interleave) {
  switch (interleave) {
  case MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE:
    return "MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE";
  case MU_TENSOR_DESCRIPTOR_INTERLEAVE_16B:
    return "MU_TENSOR_DESCRIPTOR_INTERLEAVE_16B";
  case MU_TENSOR_DESCRIPTOR_INTERLEAVE_32B:
    return "MU_TENSOR_DESCRIPTOR_INTERLEAVE_32B";
  case MU_TENSOR_DESCRIPTOR_INTERLEAVE_64B:
    return "MU_TENSOR_DESCRIPTOR_INTERLEAVE_64B";
  case MU_TENSOR_DESCRIPTOR_INTERLEAVE_128B:
    return "MU_TENSOR_DESCRIPTOR_INTERLEAVE_128B";
  case MU_TENSOR_DESCRIPTOR_INTERLEAVE_256B:
    return "MU_TENSOR_DESCRIPTOR_INTERLEAVE_256B";
  default:
    return "<unknown MUtensorDescriptorInterleave>";
  }
}

static uint64_t TensorDescriptorDataTypeBits(MUtensorDescriptorDataType type) {
  switch (type) {
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT8:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT8:
    return 8;
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT16:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT16:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT16:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_BFLOAT16:
    return 16;
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT32:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT32:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_TFLOAT32:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32_FTZ:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_TFLOAT32_FTZ:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32_TFLOAT32_RNE:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT32_FTZ_TFLOAT32_RNE:
    return 32;
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_INT64:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_UINT64:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_FLOAT64:
    return 64;
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_16b4:
  case MU_TENSOR_DESCRIPTOR_DATA_TYPE_16b4_b8:
    return 4;
  default:
    return 0;
  }
}

static uint64_t
RequiredGlobalAddressAlignment(MUtensorDescriptorDataType type,
                               MUtensorDescriptorInterleave interleave) {
  if (interleave == MU_TENSOR_DESCRIPTOR_INTERLEAVE_32B ||
      type == MU_TENSOR_DESCRIPTOR_DATA_TYPE_16b4_b8) {
    return 32;
  }
  return 16;
}

static uint64_t
RequiredGlobalStrideAlignment(MUtensorDescriptorDataType type,
                              MUtensorDescriptorInterleave interleave) {
  if (interleave == MU_TENSOR_DESCRIPTOR_INTERLEAVE_32B ||
      type == MU_TENSOR_DESCRIPTOR_DATA_TYPE_16b4_b8) {
    return 32;
  }
  return 16;
}

static std::string MusaResultToString(MUresult result) {
  const char *error_name = nullptr;
  const char *error_string = nullptr;
  (void)muGetErrorName(result, &error_name);
  (void)muGetErrorString(result, &error_string);

  std::stringstream ss;
  ss << result;
  if (error_name != nullptr) {
    ss << " (" << error_name;
    if (error_string != nullptr) {
      ss << ": " << error_string;
    }
    ss << ")";
  } else if (error_string != nullptr) {
    ss << " (" << error_string << ")";
  }
  return ss.str();
}

static std::string
FormatValidationIssues(const std::vector<std::string> &issues) {
  std::stringstream ss;
  for (size_t i = 0; i < issues.size(); ++i) {
    ss << "  [" << (i + 1) << "] " << issues[i] << '\n';
  }
  return ss.str();
}

struct MusaTensorDescriptorArgs {
  MUtensorDescriptor *desc;
  MUtensorDescriptorDataType type;
  muuint32_t tensor_rank;
  void *global_address;
  muuint64_t global_dim[5];
  muuint64_t global_stride[5];
  MUtensorDescriptorInterleave interleave;
  muuint64_t oob_fill;

  // These fields are not consumed by muTensorDescriptorEncode, but are part of
  // create_tma_descriptor packed args and useful for diagnostics.
  muuint32_t smem_box[5];
  muuint32_t element_stride[5];
  int64_t swizzle;
  int64_t l2_promotion;

  static MusaTensorDescriptorArgs Extract(PackedArgs args) {
    MusaTensorDescriptorArgs t{};
    int idx = 0;

    ICHECK(args.size() >= 8)
        << "Invalid __tvm_tensormap_create_tiled args size: " << args.size();

    t.desc = reinterpret_cast<MUtensorDescriptor *>(args[idx++].cast<void *>());
    t.type =
        static_cast<MUtensorDescriptorDataType>(args[idx++].cast<int64_t>());
    t.tensor_rank = static_cast<muuint32_t>(args[idx++].cast<int64_t>());
    t.global_address = args[idx++].cast<void *>();

    ICHECK(t.tensor_rank >= 1 && t.tensor_rank <= 5)
        << "Invalid tensor rank for MUSA TMA descriptor: " << t.tensor_rank;

    // Expected packed args:
    // desc, dtype, rank, global_addr,
    // global_dim[rank], global_stride[rank], smem_box[rank], elem_stride[rank],
    // interleave, swizzle, l2_promotion, oob_fill
    int expected = static_cast<int>(8 + t.tensor_rank * 4);
    ICHECK_EQ(args.size(), expected)
        << "Unexpected __tvm_tensormap_create_tiled args size: got "
        << args.size() << ", expected " << expected;

    for (size_t i = 0; i < t.tensor_rank; ++i) {
      t.global_dim[i] = args[idx++].cast<muuint64_t>();
    }
    for (size_t i = 0; i < t.tensor_rank; ++i) {
      t.global_stride[i] = args[idx++].cast<muuint64_t>();
    }
    for (size_t i = 0; i < t.tensor_rank; ++i) {
      t.smem_box[i] = args[idx++].cast<muuint32_t>();
    }
    for (size_t i = 0; i < t.tensor_rank; ++i) {
      t.element_stride[i] = args[idx++].cast<muuint32_t>();
    }

    t.interleave =
        static_cast<MUtensorDescriptorInterleave>(args[idx++].cast<int64_t>());
    t.swizzle = args[idx++].cast<int64_t>();
    t.l2_promotion = args[idx++].cast<int64_t>();
    t.oob_fill = static_cast<muuint64_t>(args[idx++].cast<int64_t>());

    return t;
  }

  std::string ToDebugString() const {
    std::stringstream ss;
    ss << "TMA Desc Addr:   " << desc << " (mod64=" << PtrModulo(desc, 64)
       << ")\n"
       << "format         " << type << " ("
       << TensorDescriptorDataTypeToString(type) << ")\n"
       << "dim            " << tensor_rank << '\n'
       << "gmem_address   " << global_address
       << " (mod16=" << PtrModulo(global_address, 16)
       << ", mod32=" << PtrModulo(global_address, 32) << ")\n"
       << "globalDim      " << ArrayToStr(global_dim, tensor_rank) << '\n'
       << "globalStridesRaw " << ArrayToStr(global_stride, tensor_rank) << '\n'
       << "musaGlobalStrides "
       << ArrayToStr(global_stride + 1, tensor_rank == 0 ? 0 : tensor_rank - 1)
       << '\n'
       << "smemBox        " << ArrayToStr(smem_box, tensor_rank) << '\n'
       << "elementStrides " << ArrayToStr(element_stride, tensor_rank) << '\n'
       << "interleave     " << interleave << " ("
       << TensorDescriptorInterleaveToString(interleave) << ")\n"
       << "swizzleRaw     " << swizzle << '\n'
       << "l2PromotionRaw " << l2_promotion << '\n'
       << "oobFill        " << oob_fill << '\n';
    return ss.str();
  }
};

static std::vector<std::string>
ValidateMusaTensorDescriptorArgs(const MusaTensorDescriptorArgs &t) {
  std::vector<std::string> issues;
  uint64_t type_bits = TensorDescriptorDataTypeBits(t.type);
  uint64_t addr_align = RequiredGlobalAddressAlignment(t.type, t.interleave);
  uint64_t stride_align = RequiredGlobalStrideAlignment(t.type, t.interleave);

  if (t.desc == nullptr) {
    issues.push_back("tensorDesc must be non-null");
  } else if (PtrModulo(t.desc, 64) != 0) {
    issues.push_back("tensorDesc address must be 64-byte aligned, but mod64=" +
                     std::to_string(PtrModulo(t.desc, 64)));
  }

  if (type_bits == 0) {
    issues.push_back("tensorDataType is not a supported "
                     "MUtensorDescriptorDataType enum: " +
                     std::to_string(static_cast<int>(t.type)));
  }

  if (t.tensor_rank == 0 || t.tensor_rank > 5) {
    issues.push_back("tensorRank must be in [1, 5], but got " +
                     std::to_string(t.tensor_rank));
  }

  if (t.interleave != MU_TENSOR_DESCRIPTOR_INTERLEAVE_NONE &&
      t.tensor_rank < 3) {
    issues.push_back("tensorRank must be >= 3 when interleave is not NONE");
  }

  if (t.global_address == nullptr) {
    issues.push_back("globalAddress must be non-null");
  } else if (PtrModulo(t.global_address, addr_align) != 0) {
    issues.push_back("globalAddress must be " + std::to_string(addr_align) +
                     "-byte aligned, but mod" + std::to_string(addr_align) +
                     "=" +
                     std::to_string(PtrModulo(t.global_address, addr_align)));
  }

  for (size_t i = 0; i < t.tensor_rank; ++i) {
    if (t.global_dim[i] == 0) {
      issues.push_back("globalDim[" + std::to_string(i) + "] must be non-zero");
    }
    if (t.global_dim[i] > (uint64_t{1} << 32)) {
      issues.push_back("globalDim[" + std::to_string(i) +
                       "] must be <= 2^32, but got " +
                       std::to_string(t.global_dim[i]));
    }
  }

  for (size_t raw_i = 1; raw_i < t.tensor_rank; ++raw_i) {
    muuint64_t stride = t.global_stride[raw_i];
    size_t musa_i = raw_i - 1;
    if (stride == 0) {
      issues.push_back("effective musa globalStrides[" +
                       std::to_string(musa_i) + "] (raw globalStride[" +
                       std::to_string(raw_i) + "]) must be non-zero");
    }
    if (stride % stride_align != 0) {
      issues.push_back("effective musa globalStrides[" +
                       std::to_string(musa_i) + "] (raw globalStride[" +
                       std::to_string(raw_i) + "] = " + std::to_string(stride) +
                       ") must be a multiple of " +
                       std::to_string(stride_align) + " bytes");
    }
    if (stride >= (uint64_t{1} << 40)) {
      issues.push_back("effective musa globalStrides[" +
                       std::to_string(musa_i) + "] (raw globalStride[" +
                       std::to_string(raw_i) + "] = " + std::to_string(stride) +
                       ") must be < 2^40");
    }
  }

  return issues;
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      tl::tvm_tensormap_create_tiled, [](PackedArgs args, Any *ret) {
        MusaTensorDescriptorArgs t = MusaTensorDescriptorArgs::Extract(args);
        std::vector<std::string> issues = ValidateMusaTensorDescriptorArgs(t);
        if (!issues.empty()) {
          LOG_FATAL << "Invalid MUSA TMA descriptor arguments for "
                    << tl::tvm_tensormap_create_tiled << ":\n"
                    << FormatValidationIssues(issues) << t.ToDebugString();
        }
        MUresult result = muTensorDescriptorEncode(
            t.desc, t.type, t.tensor_rank, t.global_address, t.global_dim,
            t.global_stride + 1, t.interleave, t.oob_fill);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "muTensorDescriptorEncode failed with "
                    << MusaResultToString(result) << '\n'
                    << "No local MUSA TMA descriptor constraint violation was "
                       "detected before calling muTensorDescriptorEncode.\n"
                    << t.ToDebugString();
        }
        *ret = static_cast<int>(result);
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      tl::tvm_tensormap_create_im2col, [](PackedArgs args, Any *ret) {
        LOG(FATAL) << "__tvm_tensormap_create_im2col is not "
                      "supported in MUSA TVM FFI runtime yet.";
        *ret = -1;
      });
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def_packed(
      tl::tvm_musa_stream_set_access_policy_window,
      [](PackedArgs args, Any *ret) {
        ICHECK(args.size() >= 2) << "Expected at least base_ptr and num_bytes";

        void *base_ptr = args[0].cast<void *>();
        size_t num_bytes = static_cast<size_t>(args[1].cast<int64_t>());
        float hit_ratio = 0.8f;
        if (args.size() >= 3) {
          hit_ratio = static_cast<float>(args[2].cast<double>());
        }
        MUstream stream = nullptr;
        if (args.size() >= 4) {
          stream = reinterpret_cast<MUstream>(args[3].cast<void *>());
        }
        size_t l2_limit_bytes = num_bytes;
        if (args.size() >= 5) {
          l2_limit_bytes = static_cast<size_t>(args[4].cast<int64_t>());
        }

        MUdevice device;
        MUresult result = muCtxGetDevice(&device);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to get current MUSA device: " << result;
        }

        int max_persisting = 0;
        result = muDeviceGetAttribute(
            &max_persisting, MU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            device);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to query MAX_PERSISTING_L2_CACHE_SIZE: "
                    << result;
        }
        if (max_persisting > 0 &&
            l2_limit_bytes > static_cast<size_t>(max_persisting)) {
          l2_limit_bytes = static_cast<size_t>(max_persisting);
        }

        size_t init_persisting_l2_cache_size = 0;
        result = muCtxGetLimit(&init_persisting_l2_cache_size,
                               MU_LIMIT_PERSISTING_L2_CACHE_SIZE);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to get current persisting L2 cache size limit: "
                    << result;
        }
        __tl_prev_persisting_l2_cache_size = init_persisting_l2_cache_size;
        __tl_prev_persisting_l2_cache_saved = true;

        result =
            muCtxSetLimit(MU_LIMIT_PERSISTING_L2_CACHE_SIZE, l2_limit_bytes);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to set persisting L2 cache size limit: "
                    << result;
        }

        MUstreamAttrValue stream_attribute{};
        stream_attribute.accessPolicyWindow.base_ptr = base_ptr;
        stream_attribute.accessPolicyWindow.num_bytes = l2_limit_bytes;
        stream_attribute.accessPolicyWindow.hitRatio = hit_ratio;
        stream_attribute.accessPolicyWindow.hitProp =
            MU_ACCESS_PROPERTY_PERSISTING;
        stream_attribute.accessPolicyWindow.missProp =
            MU_ACCESS_PROPERTY_STREAMING;

        result = muStreamSetAttribute(stream,
                                      MU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                                      &stream_attribute);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to set stream access policy window: " << result;
        }

        *ret = static_cast<int>(result);
      });

  refl::GlobalDef().def_packed(
      tl::tvm_musa_stream_reset_access_policy_window,
      [](PackedArgs args, Any *ret) {
        MUstream stream = nullptr;
        if (args.size() >= 1) {
          stream = reinterpret_cast<MUstream>(args[0].cast<void *>());
        }

        MUstreamAttrValue stream_attribute{};
        stream_attribute.accessPolicyWindow.num_bytes = 0;

        MUresult result = muStreamSetAttribute(
            stream, MU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &stream_attribute);
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to reset stream access policy window: "
                    << result;
        }

        result = muCtxResetPersistingL2Cache();
        if (result != MUSA_SUCCESS) {
          LOG_FATAL << "Failed to reset persisting L2 cache lines: " << result;
        }

        if (__tl_prev_persisting_l2_cache_saved) {
          result = muCtxSetLimit(MU_LIMIT_PERSISTING_L2_CACHE_SIZE,
                                 __tl_prev_persisting_l2_cache_size);
          if (result != MUSA_SUCCESS) {
            LOG_FATAL << "Failed to restore persisting L2 cache size limit: "
                      << result;
          }
          __tl_prev_persisting_l2_cache_saved = false;
        }

        *ret = static_cast<int>(result);
      });
}
#endif // defined(MUSA_MAJOR_VERSION)

#if defined(CUDA_MAJOR_VERSION)
//
// CUDA L2 Persisting Cache Access Policy Window helpers.
// Exposed as TVM FFI packed functions similar to TMA initialization.
//
TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  // Set stream access policy window and adjust persisting L2 cache size
  // Args:
  //  [0]: void* base_ptr (required)
  //  [1]: int64 num_bytes (required)
  //  [2]: float hit_ratio (optional, default 0.8)
  //  [3]: void* stream (optional, default 0 => default stream)
  //  [4]: int64 l2_limit_bytes (optional, default = num_bytes)
  refl::GlobalDef().def_packed(
      tl::tvm_cuda_stream_set_access_policy_window,
      [](PackedArgs args, Any *ret) {
        ICHECK(args.size() >= 2) << "Expected at least base_ptr and num_bytes";

        void *base_ptr = args[0].cast<void *>();
        size_t num_bytes = static_cast<size_t>(args[1].cast<int64_t>());
        float hit_ratio = 0.8f;
        if (args.size() >= 3) {
          // Accept double/float
          hit_ratio = static_cast<float>(args[2].cast<double>());
        }
        CUstream stream = nullptr;
        if (args.size() >= 4) {
          stream = reinterpret_cast<CUstream>(args[3].cast<void *>());
        }
        size_t l2_limit_bytes = num_bytes;
        if (args.size() >= 5) {
          l2_limit_bytes = static_cast<size_t>(args[4].cast<int64_t>());
        }

        // Clamp requested limit to device capability
        CUdevice device;
        CUresult result = cuCtxGetDevice(&device);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to get current CUDA device: " << result;
        }
        int max_persisting = 0;
        result = cuDeviceGetAttribute(
            &max_persisting, CU_DEVICE_ATTRIBUTE_MAX_PERSISTING_L2_CACHE_SIZE,
            device);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to query MAX_PERSISTING_L2_CACHE_SIZE: "
                    << result;
        }
        if (max_persisting > 0 &&
            l2_limit_bytes > static_cast<size_t>(max_persisting)) {
          l2_limit_bytes = static_cast<size_t>(max_persisting);
        }

        // Save current limit to restore later
        size_t init_persisting_l2_cache_size = 0;
        result = cuCtxGetLimit(&init_persisting_l2_cache_size,
                               CU_LIMIT_PERSISTING_L2_CACHE_SIZE);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to get current persisting L2 cache size limit: "
                    << result;
        }
        __tl_prev_persisting_l2_cache_size = init_persisting_l2_cache_size;
        __tl_prev_persisting_l2_cache_saved = true;

        // Set new limit
        result =
            cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE, l2_limit_bytes);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to set persisting L2 cache size limit: "
                    << result;
        }

        // Apply access policy window to stream
        CUstreamAttrValue stream_attribute;
        memset(&stream_attribute, 0, sizeof(stream_attribute));
        stream_attribute.accessPolicyWindow.base_ptr = base_ptr;
        stream_attribute.accessPolicyWindow.num_bytes = l2_limit_bytes;
        stream_attribute.accessPolicyWindow.hitRatio = hit_ratio;
        stream_attribute.accessPolicyWindow.hitProp =
            CU_ACCESS_PROPERTY_PERSISTING;
        stream_attribute.accessPolicyWindow.missProp =
            CU_ACCESS_PROPERTY_STREAMING;

        result = cuStreamSetAttribute(stream,
                                      CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
                                      &stream_attribute);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to set stream access policy window: " << result;
        }

        *ret = static_cast<int>(result);
      });

  // Reset stream access policy window and restore the previous L2 cache size
  // Args:
  //  [0]: void* stream (optional, default 0)
  refl::GlobalDef().def_packed(
      tl::tvm_cuda_stream_reset_access_policy_window,
      [](PackedArgs args, Any *ret) {
        CUstream stream = nullptr;
        if (args.size() >= 1) {
          stream = reinterpret_cast<CUstream>(args[0].cast<void *>());
        }

        CUstreamAttrValue stream_attribute;
        memset(&stream_attribute, 0, sizeof(stream_attribute));
        // num_bytes = 0 disables the access policy window on the stream
        stream_attribute.accessPolicyWindow.num_bytes = 0;

        CUresult result = cuStreamSetAttribute(
            stream, CU_STREAM_ATTRIBUTE_ACCESS_POLICY_WINDOW,
            &stream_attribute);
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to reset stream access policy window: "
                    << result;
        }

        result = cuCtxResetPersistingL2Cache();
        if (result != CUDA_SUCCESS) {
          LOG_FATAL << "Failed to reset persisting L2 cache lines: " << result;
        }

        if (__tl_prev_persisting_l2_cache_saved) {
          result = cuCtxSetLimit(CU_LIMIT_PERSISTING_L2_CACHE_SIZE,
                                 __tl_prev_persisting_l2_cache_size);
          if (result != CUDA_SUCCESS) {
            LOG_FATAL << "Failed to restore persisting L2 cache size limit: "
                      << result;
          }
          __tl_prev_persisting_l2_cache_saved = false;
        }

        *ret = static_cast<int>(result);
      });
}
#endif // defined(CUDA_MAJOR_VERSION)

} // namespace tl
} // namespace tvm
