/*!
 * \file backend/musa/transform/lower_async_copy.cc
 * \brief Lower MUSA global-to-shared copies into async-copy intrinsics.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/target/target.h>
#include <tvm/tirx/transform.h>

#include "backend/musa/target_utils.h"
#include "cuda/transform/ptx_async_copy_injector.h"
#include "op/builtin.h"

namespace tvm {
namespace tl {

using namespace tirx;
using namespace tirx::transform;

tvm::transform::Pass LowerMUSAAsyncCopy() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    auto target = f->GetAttr<Target>(tvm::attr::kTarget);
    if (!target.defined() || !TargetIsMusa(target.value()) ||
        !TargetMusaHasAsyncCopy(target.value())) {
      return f;
    }

    bool enable_auto_async_copy =
        ctx->GetConfig<Bool>(kEnableAsyncCopy, Bool(true)).value();
    bool disable_thread_storage_sync =
        ctx->GetConfig<Bool>(kDisableThreadStorageSync, Bool(false)).value();

    auto *n = f.CopyOnWrite();
    auto result = InjectPTXAsyncCopy(
        n->body, enable_auto_async_copy,
        /*async_without_async_commit_wait=*/false,
        disable_thread_storage_sync,
        /*sync_inside_conditionals=*/true);
    n->body = result.stmt;
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.musa.LowerAsyncCopy", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerMUSAAsyncCopy",
                        LowerMUSAAsyncCopy);
}

} // namespace tl
} // namespace tvm
