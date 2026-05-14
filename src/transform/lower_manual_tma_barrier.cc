/*
 * Licensed to the Apache Software Foundation (ASF) under one
 * or more contributor license agreements.  See the NOTICE file
 * distributed with this work for additional information
 * regarding copyright ownership. The ASF licenses this file
 * to you under the Apache License, Version 2.0 (the
 * "License"); you may not use this file except in compliance
 * with the License.  You may obtain a copy of the License at
 *
 *   http://www.apache.org/licenses/LICENSE-2.0
 *
 * Unless required by applicable law or agreed to in writing,
 * software distributed under the License is distributed on an
 * "AS IS" BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY
 * KIND, either express or implied.  See the License for the
 * specific language governing permissions and limitations
 * under the License.
 */

/*!
 * \file lower_manual_tma_barrier.cc
 * \brief Lower tl.manual_tma_barrier markers on TMA load barrier operands.
 */

#include <tvm/ffi/reflection/registry.h>
#include <tvm/tir/expr.h>
#include <tvm/tir/stmt_functor.h>
#include <tvm/tir/transform.h>

#include "../op/builtin.h"

namespace tvm {
namespace tl {

using namespace tir;
using namespace tir::transform;

namespace {

bool IsManualTmaBarrier(const PrimExpr &barrier) {
  if (const auto *call = barrier.as<CallNode>()) {
    return call->op.same_as(manual_tma_barrier()) && call->args.size() == 1;
  }
  return false;
}

PrimExpr UnwrapManualTmaBarrier(const PrimExpr &barrier) {
  if (const auto *call = barrier.as<CallNode>()) {
    if (call->op.same_as(manual_tma_barrier()) && call->args.size() == 1) {
      return call->args[0];
    }
  }
  return barrier;
}

bool Is1DTmaLoad(const CallNode *op) {
  if (!op->op.same_as(tma_load())) {
    return false;
  }
  auto arg0 = op->args[0].as<Call>();
  return arg0 && !arg0.value()->op.same_as(create_tma_descriptor()) &&
         !arg0.value()->op.same_as(create_tma_im2col_descriptor());
}

int GetTmaBarrierArgIndex(const CallNode *op) {
  return Is1DTmaLoad(op) ? 2 : 1;
}

class ManualTmaBarrierRewriter : public StmtExprMutator {
private:
  PrimExpr VisitExpr_(const CallNode *call) final {
    if (call->op.same_as(tma_load()) || call->op.same_as(tma_load_im2col())) {
      int barrier_arg_index = GetTmaBarrierArgIndex(call);
      PrimExpr barrier = call->args[barrier_arg_index];
      if (IsManualTmaBarrier(barrier)) {
        Array<PrimExpr> new_args = call->args;
        new_args.Set(barrier_arg_index, UnwrapManualTmaBarrier(barrier));
        return Call(call->dtype, call->op, new_args, call->annotations);
      }
    }
    return StmtExprMutator::VisitExpr_(call);
  }
};

} // namespace

tvm::transform::Pass LowerManualTmaBarrier() {
  auto pass_func = [=](PrimFunc f, const IRModule &m, const PassContext &ctx) {
    f.CopyOnWrite()->body = ManualTmaBarrierRewriter()(f->body);
    return f;
  };
  return CreatePrimFuncPass(pass_func, 0, "tl.LowerManualTmaBarrier", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef().def("tl.transform.LowerManualTmaBarrier",
                        LowerManualTmaBarrier);
}

} // namespace tl
} // namespace tvm
