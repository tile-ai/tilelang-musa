/*!
 * \file musa/shared_memory.cc
 * \brief MUSA IPC and fabric-VMM ops registered through TVM FFI.
 *
 * This is intentionally limited to the transport primitives needed by the
 * MUSA peer table.  Multicast is deliberately not exposed by this module.
 */

#include <musa.h>
#include <musa_runtime.h>
#include <tvm/ffi/function.h>
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ffi/string.h>

#include <cstdint>
#include <cstring>
#include <limits>
#include <mutex>
#include <stdexcept>
#include <string>
#include <unordered_map>
#include <vector>

using namespace tvm;
using namespace tvm::ffi;

namespace {

[[noreturn]] void ThrowMusaRuntimeError(musaError_t error, const char *op) {
  const char *message = musaGetErrorString(error);
  throw std::runtime_error(
      std::string(op) +
      " failed: " + (message != nullptr ? message : "unknown MUSA error"));
}

[[noreturn]] void ThrowMusaDriverError(MUresult error, const char *op) {
  const char *message = nullptr;
  muGetErrorString(error, &message);
  throw std::runtime_error(
      std::string(op) + " failed: " +
      (message != nullptr ? message : "unknown MUSA driver error"));
}

void CheckRuntime(musaError_t error, const char *op) {
  if (error != musaSuccess) {
    ThrowMusaRuntimeError(error, op);
  }
}

void CheckDriver(MUresult error, const char *op) {
  if (error != MUSA_SUCCESS) {
    ThrowMusaDriverError(error, op);
  }
}

uintptr_t CheckedAddress(int64_t value, const char *name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) +
                                " must be a non-zero address");
  }
  return static_cast<uintptr_t>(value);
}

size_t CheckedSize(int64_t value, const char *name) {
  if (value <= 0) {
    throw std::invalid_argument(std::string(name) + " must be positive");
  }
  const uint64_t unsigned_value = static_cast<uint64_t>(value);
  if (unsigned_value > std::numeric_limits<size_t>::max()) {
    throw std::overflow_error(std::string(name) + " does not fit size_t");
  }
  return static_cast<size_t>(unsigned_value);
}

int64_t CheckedOutputAddress(MUdeviceptr ptr, const char *op) {
  if (ptr == 0 ||
      ptr > static_cast<MUdeviceptr>(std::numeric_limits<int64_t>::max())) {
    throw std::overflow_error(std::string(op) + " returned an invalid address");
  }
  return static_cast<int64_t>(ptr);
}

size_t AlignUp(size_t value, size_t alignment) {
  if (alignment == 0 ||
      value > std::numeric_limits<size_t>::max() - (alignment - 1)) {
    throw std::overflow_error("VMM allocation size overflow");
  }
  return ((value + alignment - 1) / alignment) * alignment;
}

void CheckExactBytes(const ffi::Bytes &value, size_t expected,
                     const char *name) {
  if (value.size() != expected) {
    throw std::invalid_argument(std::string(name) + " must contain exactly " +
                                std::to_string(expected) + " bytes, got " +
                                std::to_string(value.size()));
  }
}

MUmemAllocationProp FabricAllocationProperty(MUdevice device) {
  MUmemAllocationProp property{};
  property.type = MU_MEM_ALLOCATION_TYPE_PINNED;
  property.requestedHandleTypes = MU_MEM_HANDLE_TYPE_FABRIC;
  property.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
  property.location.id = static_cast<int>(device);
  return property;
}

void SetCurrentDeviceAccess(MUdeviceptr ptr, size_t size) {
  MUdevice device = 0;
  CheckDriver(muCtxGetDevice(&device), "muCtxGetDevice");
  MUmemAccessDesc access{};
  access.location.type = MU_MEM_LOCATION_TYPE_DEVICE;
  access.location.id = static_cast<int>(device);
  access.flags = MU_MEM_ACCESS_FLAGS_PROT_READWRITE;
  CheckDriver(muMemSetAccess(ptr, size, &access, 1), "muMemSetAccess");
}

constexpr size_t kVmmHandleSize = sizeof(uint64_t) + sizeof(MUmemFabricHandle);
constexpr size_t kIpcHandleSize = sizeof(uint64_t) * 2 + MUSA_IPC_HANDLE_SIZE;

bool TryGetAllocationRange(MUdeviceptr ptr, MUdeviceptr *base, size_t *size) {
  try {
    if (muMemGetAddressRange(base, size, ptr) == MUSA_SUCCESS && *base != 0 &&
        *size != 0) {
      return true;
    }
  } catch (const std::exception &) {
    // Older drivers may not export muMemGetAddressRange. Fall through to the
    // pointer-attribute API before declaring IPC unsupported.
  }

  *base = 0;
  *size = 0;
  try {
    const MUresult base_result =
        muPointerGetAttribute(base, MU_POINTER_ATTRIBUTE_RANGE_START_ADDR, ptr);
    const MUresult size_result =
        muPointerGetAttribute(size, MU_POINTER_ATTRIBUTE_RANGE_SIZE, ptr);
    return base_result == MUSA_SUCCESS && size_result == MUSA_SUCCESS &&
           *base != 0 && *size != 0;
  } catch (const std::exception &) {
    return false;
  }
}

struct VmmMapping {
  size_t size;
  MUmemGenericAllocationHandle handle;
};

struct VmmContiguousMapping {
  std::vector<size_t> chunk_sizes;
  size_t total_size;
  std::vector<MUmemGenericAllocationHandle> handles;
};

std::mutex &VmmMappingMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uintptr_t, VmmMapping> &VmmMappings() {
  static std::unordered_map<uintptr_t, VmmMapping> mappings;
  return mappings;
}

std::unordered_map<uintptr_t, VmmContiguousMapping> &VmmContiguousMappings() {
  static std::unordered_map<uintptr_t, VmmContiguousMapping> mappings;
  return mappings;
}

std::mutex &IpcMappingMutex() {
  static std::mutex mutex;
  return mutex;
}

std::unordered_map<uintptr_t, void *> &IpcMappingBases() {
  static std::unordered_map<uintptr_t, void *> mappings;
  return mappings;
}

int64_t VmmMalloc(int64_t requested_size) {
  const size_t raw_size = CheckedSize(requested_size, "size");
  MUdevice device = 0;
  CheckDriver(muCtxGetDevice(&device), "muCtxGetDevice");
  const MUmemAllocationProp property = FabricAllocationProperty(device);

  size_t granularity = 0;
  CheckDriver(muMemGetAllocationGranularity(&granularity, &property,
                                            MU_MEM_ALLOC_GRANULARITY_MINIMUM),
              "muMemGetAllocationGranularity");
  const size_t size = AlignUp(raw_size, granularity);

  MUmemGenericAllocationHandle handle = 0;
  MUdeviceptr ptr = 0;
  bool mapped = false;
  try {
    CheckDriver(muMemCreate(&handle, size, &property, 0), "muMemCreate");
    CheckDriver(muMemAddressReserve(&ptr, size, granularity, 0, 0),
                "muMemAddressReserve");
    CheckDriver(muMemMap(ptr, size, 0, handle, 0), "muMemMap");
    mapped = true;
    SetCurrentDeviceAccess(ptr, size);
    {
      std::lock_guard<std::mutex> lock(VmmMappingMutex());
      const auto inserted = VmmMappings().emplace(static_cast<uintptr_t>(ptr),
                                                  VmmMapping{size, handle});
      if (!inserted.second) {
        throw std::runtime_error("duplicate MUSA VMM pointer registration");
      }
    }
    handle = 0;
    return CheckedOutputAddress(ptr, "vmm_malloc");
  } catch (...) {
    if (mapped) {
      muMemUnmap(ptr, size);
    }
    if (ptr != 0) {
      muMemAddressFree(ptr, size);
    }
    if (handle != 0) {
      muMemRelease(handle);
    }
    throw;
  }
}

void VmmFree(int64_t ptr_value) {
  const uintptr_t address = CheckedAddress(ptr_value, "ptr");
  VmmMapping mapping{};
  {
    std::lock_guard<std::mutex> lock(VmmMappingMutex());
    auto it = VmmMappings().find(address);
    if (it == VmmMappings().end()) {
      throw std::invalid_argument("pointer is not an open MUSA VMM mapping");
    }
    mapping = it->second;
    VmmMappings().erase(it);
  }

  std::string first_error;
  auto record = [&first_error](MUresult result, const char *op) {
    if (result == MUSA_SUCCESS || !first_error.empty()) {
      return;
    }
    const char *message = nullptr;
    muGetErrorString(result, &message);
    first_error = std::string(op) + " failed: " +
                  (message != nullptr ? message : "unknown MUSA driver error");
  };
  record(muMemUnmap(static_cast<MUdeviceptr>(address), mapping.size),
         "muMemUnmap");
  record(muMemAddressFree(static_cast<MUdeviceptr>(address), mapping.size),
         "muMemAddressFree");
  record(muMemRelease(mapping.handle), "muMemRelease");
  if (!first_error.empty()) {
    throw std::runtime_error(first_error);
  }
}

ffi::Bytes CreateVmmHandle(int64_t ptr_value) {
  const uintptr_t address = CheckedAddress(ptr_value, "ptr");
  VmmMapping mapping{};
  {
    std::lock_guard<std::mutex> lock(VmmMappingMutex());
    auto it = VmmMappings().find(address);
    if (it == VmmMappings().end()) {
      throw std::invalid_argument("pointer is not an open MUSA VMM mapping");
    }
    mapping = it->second;
  }
  MUmemFabricHandle fabric{};
  CheckDriver(muMemExportToShareableHandle(&fabric, mapping.handle,
                                           MU_MEM_HANDLE_TYPE_FABRIC, 0),
              "muMemExportToShareableHandle");

  std::string packed(kVmmHandleSize, '\0');
  const uint64_t size64 = static_cast<uint64_t>(mapping.size);
  std::memcpy(packed.data(), &size64, sizeof(size64));
  std::memcpy(packed.data() + sizeof(size64), &fabric, sizeof(fabric));
  return ffi::Bytes(packed.data(), packed.size());
}

int64_t OpenVmmHandle(ffi::Bytes packed) {
  CheckExactBytes(packed, kVmmHandleSize, "handle_bytes");
  uint64_t size64 = 0;
  MUmemFabricHandle fabric{};
  std::memcpy(&size64, packed.data(), sizeof(size64));
  std::memcpy(&fabric, packed.data() + sizeof(size64), sizeof(fabric));
  if (size64 == 0 || size64 > std::numeric_limits<size_t>::max()) {
    throw std::invalid_argument(
        "VMM handle contains an invalid allocation size");
  }
  const size_t size = static_cast<size_t>(size64);

  MUmemGenericAllocationHandle handle = 0;
  MUdeviceptr ptr = 0;
  bool mapped = false;
  try {
    CheckDriver(muMemImportFromShareableHandle(&handle, &fabric,
                                               MU_MEM_HANDLE_TYPE_FABRIC),
                "muMemImportFromShareableHandle");
    CheckDriver(muMemAddressReserve(&ptr, size, 0, 0, 0),
                "muMemAddressReserve");
    CheckDriver(muMemMap(ptr, size, 0, handle, 0), "muMemMap");
    mapped = true;
    SetCurrentDeviceAccess(ptr, size);
    {
      std::lock_guard<std::mutex> lock(VmmMappingMutex());
      const auto inserted = VmmMappings().emplace(static_cast<uintptr_t>(ptr),
                                                  VmmMapping{size, handle});
      if (!inserted.second) {
        throw std::runtime_error("duplicate MUSA VMM pointer registration");
      }
    }
    handle = 0;
    return CheckedOutputAddress(ptr, "open_vmm_handle");
  } catch (...) {
    if (mapped) {
      muMemUnmap(ptr, size);
    }
    if (ptr != 0) {
      muMemAddressFree(ptr, size);
    }
    if (handle != 0) {
      muMemRelease(handle);
    }
    throw;
  }
}

int64_t OpenVmmHandlesContiguous(int64_t count_value, ffi::Bytes packed) {
  if (count_value <= 0) {
    throw std::invalid_argument("count must be positive");
  }
  const size_t count = static_cast<size_t>(count_value);
  if (count > std::numeric_limits<size_t>::max() / kVmmHandleSize) {
    throw std::overflow_error("packed VMM handle size overflow");
  }
  CheckExactBytes(packed, count * kVmmHandleSize, "handle_bytes");

  size_t total_size = 0;
  std::vector<size_t> chunk_sizes(count);
  std::vector<MUmemFabricHandle> fabrics(count);
  for (size_t index = 0; index < count; ++index) {
    uint64_t size64 = 0;
    const char *entry = packed.data() + index * kVmmHandleSize;
    std::memcpy(&size64, entry, sizeof(size64));
    std::memcpy(&fabrics[index], entry + sizeof(size64),
                sizeof(MUmemFabricHandle));
    if (size64 == 0 || size64 > std::numeric_limits<size_t>::max()) {
      throw std::invalid_argument(
          "VMM handle contains an invalid allocation size");
    }
    chunk_sizes[index] = static_cast<size_t>(size64);
    if (chunk_sizes[index] > std::numeric_limits<size_t>::max() - total_size) {
      throw std::overflow_error("contiguous VMM mapping size overflow");
    }
    total_size += chunk_sizes[index];
  }

  MUdeviceptr base = 0;
  size_t mapped_count = 0;
  size_t mapped_size = 0;
  std::vector<MUmemGenericAllocationHandle> handles(count, 0);
  try {
    CheckDriver(muMemAddressReserve(&base, total_size, 0, 0, 0),
                "muMemAddressReserve(contiguous)");
    for (size_t index = 0; index < count; ++index) {
      CheckDriver(muMemImportFromShareableHandle(&handles[index],
                                                 &fabrics[index],
                                                 MU_MEM_HANDLE_TYPE_FABRIC),
                  "muMemImportFromShareableHandle(contiguous)");
      CheckDriver(muMemMap(base + mapped_size, chunk_sizes[index], 0,
                           handles[index], 0),
                  "muMemMap(contiguous)");
      mapped_size += chunk_sizes[index];
      mapped_count = index + 1;
    }
    SetCurrentDeviceAccess(base, total_size);
    {
      std::lock_guard<std::mutex> lock(VmmMappingMutex());
      const auto inserted = VmmContiguousMappings().emplace(
          static_cast<uintptr_t>(base),
          VmmContiguousMapping{chunk_sizes, total_size, handles});
      if (!inserted.second) {
        throw std::runtime_error(
            "duplicate MUSA contiguous VMM pointer registration");
      }
    }
    return CheckedOutputAddress(base, "open_vmm_handles_contiguous");
  } catch (...) {
    size_t mapped_offset = 0;
    for (size_t index = 0; index < mapped_count; ++index) {
      muMemUnmap(base + mapped_offset, chunk_sizes[index]);
      mapped_offset += chunk_sizes[index];
    }
    if (base != 0) {
      muMemAddressFree(base, total_size);
    }
    for (MUmemGenericAllocationHandle handle : handles) {
      if (handle != 0) {
        muMemRelease(handle);
      }
    }
    throw;
  }
}

void CloseVmmHandlesContiguous(int64_t ptr_value) {
  const uintptr_t address = CheckedAddress(ptr_value, "ptr");
  VmmContiguousMapping mapping{};
  {
    std::lock_guard<std::mutex> lock(VmmMappingMutex());
    auto it = VmmContiguousMappings().find(address);
    if (it == VmmContiguousMappings().end()) {
      throw std::invalid_argument(
          "pointer is not an open MUSA contiguous VMM mapping");
    }
    mapping = std::move(it->second);
    VmmContiguousMappings().erase(it);
  }

  std::string first_error;
  auto record = [&first_error](MUresult result, const char *op) {
    if (result == MUSA_SUCCESS || !first_error.empty()) {
      return;
    }
    const char *message = nullptr;
    muGetErrorString(result, &message);
    first_error = std::string(op) + " failed: " +
                  (message != nullptr ? message : "unknown MUSA driver error");
  };
  size_t mapped_offset = 0;
  for (size_t index = 0; index < mapping.handles.size(); ++index) {
    record(muMemUnmap(static_cast<MUdeviceptr>(address) + mapped_offset,
                      mapping.chunk_sizes[index]),
           "muMemUnmap(contiguous)");
    mapped_offset += mapping.chunk_sizes[index];
  }
  record(
      muMemAddressFree(static_cast<MUdeviceptr>(address), mapping.total_size),
      "muMemAddressFree(contiguous)");
  for (MUmemGenericAllocationHandle handle : mapping.handles) {
    record(muMemRelease(handle), "muMemRelease(contiguous)");
  }
  if (!first_error.empty()) {
    throw std::runtime_error(first_error);
  }
}

ffi::Bytes CreateIpcHandle(int64_t ptr_value) {
  const MUdeviceptr ptr =
      static_cast<MUdeviceptr>(CheckedAddress(ptr_value, "ptr"));
  MUdeviceptr base = 0;
  size_t allocation_size = 0;
  if (!TryGetAllocationRange(ptr, &base, &allocation_size)) {
    throw std::runtime_error(
        "cannot determine the MUSA allocation range for IPC export");
  }
  if (ptr < base || ptr - base >= allocation_size) {
    throw std::runtime_error(
        "musaIpcGetMemHandle returned an invalid allocation range");
  }
  const uint64_t offset = static_cast<uint64_t>(ptr - base);
  const uint64_t size64 = static_cast<uint64_t>(allocation_size);
  musaIpcMemHandle_t handle{};
  CheckRuntime(musaIpcGetMemHandle(&handle, reinterpret_cast<void *>(base)),
               "musaIpcGetMemHandle");

  std::string packed(kIpcHandleSize, '\0');
  std::memcpy(packed.data(), &offset, sizeof(offset));
  std::memcpy(packed.data() + sizeof(offset), &size64, sizeof(size64));
  std::memcpy(packed.data() + sizeof(offset) + sizeof(size64), handle.reserved,
              MUSA_IPC_HANDLE_SIZE);
  return ffi::Bytes(packed.data(), packed.size());
}

int64_t OpenIpcHandle(ffi::Bytes packed) {
  CheckExactBytes(packed, kIpcHandleSize, "handle_bytes");
  uint64_t offset = 0;
  uint64_t allocation_size = 0;
  std::memcpy(&offset, packed.data(), sizeof(offset));
  std::memcpy(&allocation_size, packed.data() + sizeof(offset),
              sizeof(allocation_size));
  if (allocation_size == 0 || offset >= allocation_size) {
    throw std::invalid_argument(
        "IPC handle contains an invalid allocation offset");
  }
  musaIpcMemHandle_t handle{};
  std::memcpy(handle.reserved,
              packed.data() + sizeof(offset) + sizeof(allocation_size),
              MUSA_IPC_HANDLE_SIZE);
  void *base = nullptr;
  CheckRuntime(
      musaIpcOpenMemHandle(&base, handle, musaIpcMemLazyEnablePeerAccess),
      "musaIpcOpenMemHandle");
  const MUdeviceptr view = reinterpret_cast<MUdeviceptr>(base) + offset;
  int64_t result = 0;
  try {
    result = CheckedOutputAddress(view, "open_ipc_handle");
    std::lock_guard<std::mutex> lock(IpcMappingMutex());
    const auto inserted =
        IpcMappingBases().emplace(static_cast<uintptr_t>(view), base);
    if (!inserted.second) {
      throw std::runtime_error("duplicate MUSA IPC view pointer registration");
    }
  } catch (...) {
    musaIpcCloseMemHandle(base);
    throw;
  }
  return result;
}

void CloseIpcHandle(int64_t ptr_value) {
  const uintptr_t view = CheckedAddress(ptr_value, "ptr");
  void *base = nullptr;
  {
    std::lock_guard<std::mutex> lock(IpcMappingMutex());
    auto it = IpcMappingBases().find(view);
    if (it == IpcMappingBases().end()) {
      throw std::invalid_argument("pointer is not an open MUSA IPC mapping");
    }
    base = it->second;
    IpcMappingBases().erase(it);
  }
  CheckRuntime(musaIpcCloseMemHandle(base), "musaIpcCloseMemHandle");
}

size_t CheckedRankCount(int64_t rank, int64_t count) {
  if (count <= 0 || rank < 0 || rank >= count) {
    throw std::invalid_argument("rank must be in [0, num_ranks)");
  }
  return static_cast<size_t>(count);
}

template <typename Open, typename Close>
void SyncHandles(int64_t rank, int64_t count, int64_t table_address,
                 ffi::Bytes packed, size_t handle_size, Open open,
                 Close close) {
  const size_t rank_count = CheckedRankCount(rank, count);
  if (handle_size > std::numeric_limits<size_t>::max() / rank_count) {
    throw std::overflow_error("packed handle size overflow");
  }
  CheckExactBytes(packed, handle_size * rank_count, "packed_handles");
  void *table = reinterpret_cast<void *>(
      CheckedAddress(table_address, "buffer_ptrs_gpu_addr"));

  std::vector<uint64_t> pointers(rank_count, 0);
  try {
    for (size_t peer = 0; peer < rank_count; ++peer) {
      if (peer == static_cast<size_t>(rank)) {
        continue;
      }
      ffi::Bytes handle(packed.data() + peer * handle_size, handle_size);
      pointers[peer] = static_cast<uint64_t>(open(handle));
    }
    CheckRuntime(musaMemcpy(table, pointers.data(),
                            sizeof(uint64_t) * rank_count,
                            musaMemcpyHostToDevice),
                 "musaMemcpy(peer table)");
    CheckRuntime(musaDeviceSynchronize(), "musaDeviceSynchronize");
  } catch (...) {
    for (size_t peer = 0; peer < rank_count; ++peer) {
      if (peer != static_cast<size_t>(rank) && pointers[peer] != 0) {
        try {
          close(static_cast<int64_t>(pointers[peer]));
        } catch (...) {
        }
      }
    }
    throw;
  }
}

bool SupportsVmmFabric() {
  try {
    MUdevice device = 0;
    if (muCtxGetDevice(&device) != MUSA_SUCCESS) {
      return false;
    }
    int fabric_supported = 0;
    if (muDeviceGetAttribute(&fabric_supported,
                             MU_DEVICE_ATTRIBUTE_HANDLE_TYPE_FABRIC_SUPPORTED,
                             device) != MUSA_SUCCESS ||
        fabric_supported == 0) {
      return false;
    }

    // Some MUSA driver releases report the fabric attribute as false while
    // export/import is functional.  Do not probe those releases here: calling
    // the fabric export API when the active driver reports no support may
    // terminate the process instead of returning an error.
    const int64_t owner = VmmMalloc(1);
    try {
      const ffi::Bytes handle = CreateVmmHandle(owner);
      const int64_t imported = OpenVmmHandle(handle);
      VmmFree(imported);
      VmmFree(owner);
    } catch (...) {
      VmmFree(owner);
      throw;
    }
    return true;
  } catch (...) {
    return false;
  }
}

void CloseVmmHandle(int64_t ptr_value) { VmmFree(ptr_value); }

} // namespace

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;

  refl::GlobalDef().def("tl.musa.shared_memory.vmm_malloc", VmmMalloc);
  refl::GlobalDef().def("tl.musa.shared_memory.vmm_free", VmmFree);
  refl::GlobalDef().def("tl.musa.shared_memory.create_vmm_handle",
                        CreateVmmHandle);
  refl::GlobalDef().def("tl.musa.shared_memory.open_vmm_handle", OpenVmmHandle);
  refl::GlobalDef().def("tl.musa.shared_memory.close_vmm_handle",
                        CloseVmmHandle);
  refl::GlobalDef().def("tl.musa.shared_memory.open_vmm_handles_contiguous",
                        OpenVmmHandlesContiguous);
  refl::GlobalDef().def("tl.musa.shared_memory.close_vmm_handles_contiguous",
                        CloseVmmHandlesContiguous);
  refl::GlobalDef().def(
      "tl.musa.shared_memory.sync_vmm_handles",
      [](int64_t rank, int64_t count, int64_t table, ffi::Bytes handles) {
        SyncHandles(rank, count, table, handles, kVmmHandleSize, OpenVmmHandle,
                    VmmFree);
      });

  refl::GlobalDef().def("tl.musa.shared_memory.create_ipc_handle",
                        CreateIpcHandle);
  refl::GlobalDef().def("tl.musa.shared_memory.open_ipc_handle", OpenIpcHandle);
  refl::GlobalDef().def("tl.musa.shared_memory.close_ipc_handle",
                        CloseIpcHandle);
  refl::GlobalDef().def(
      "tl.musa.shared_memory.sync_ipc_handles",
      [](int64_t rank, int64_t count, int64_t table, ffi::Bytes handles) {
        SyncHandles(rank, count, table, handles, kIpcHandleSize, OpenIpcHandle,
                    CloseIpcHandle);
      });

  refl::GlobalDef().def("tl.musa.shared_memory.supports_vmm_fabric",
                        SupportsVmmFabric);
  refl::GlobalDef().def("tl.musa.shared_memory.supports_multicast",
                        []() { return false; });
}
