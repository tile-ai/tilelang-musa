#pragma once

#include <tvm/tirx/stmt.h>

namespace tvm {
namespace tl {

struct MUSAAsyncCopyInjectResult {
  tvm::tirx::Stmt stmt;
  bool injected_ptx_async_copy{false};
};

/*! \\brief Inject MUSA async-copy lowering patterns into a statement. */
MUSAAsyncCopyInjectResult
InjectMUSAAsyncCopy(const tvm::tirx::Stmt &body,
                    bool async_without_async_commit_wait = false);

} // namespace tl
} // namespace tvm

