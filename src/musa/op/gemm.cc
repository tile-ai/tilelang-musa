/*!
 * \file tl/musa/op/gemm.cc
 * \brief MUSA implementation for tl.gemm instruction dispatch.
 */

#include "op/gemm.h"

#include "musa/op/gemm/mp31_sqmma.h"
#include "musa/op/gemm/mp31_wmma.h"
#include "musa/target_utils.h"
#include "op/utils.h"

#include <tvm/runtime/logging.h>
#include <tvm/tirx/op_attr_types.h>

namespace tvm {
namespace tl {

using namespace tirx;
using namespace ffi;

namespace musa {

struct Gemm {
private:
  static String SelectSqmmaGemm(const GemmNode &op, int block_size,
                                Target target) {
    ICHECK(TargetIsMP31(target))
        << "T.sqmma_gemm is only supported on MP31, target=" << target;
    return mp31::SQMMA::SelectInst(op, block_size, target);
  }

  static String SelectWmmaGemm(const GemmNode &op, int block_size,
                               Target target) {
    ICHECK(TargetIsMP31(target))
        << "T.wmma_gemm is only supported on MP31, target=" << target;
    return mp31::WMMA::SelectInst(op, block_size, target);
  }

  static String SelectGemm(const GemmNode &op, int block_size, Target target) {
    if (IsSharedBuffer(op.a_) && IsSharedBuffer(op.b_)) {
      return SelectSqmmaGemm(op, block_size, target);
    }
    if (IsFragmentBuffer(op.a_) && IsFragmentBuffer(op.b_)) {
      return SelectWmmaGemm(op, block_size, target);
    }
    LOG(FATAL) << "MUSA T.gemm supports shared/shared (SQMMA) or "
                  "fragment/fragment (WMMA) operands, but got A(scope="
               << op.a_.scope() << "), B(scope=" << op.b_.scope()
               << "), C(scope=" << op.c_.scope() << "), target=" << target;
    return {};
  }

public:
  static String SelectInst(const GemmNode &op, int block_size, Target target) {
    // Explicit APIs are handled first so their operation-specific contract is
    // preserved even when the operand scopes do not match.  Plain T.gemm then
    // uses the MP31 operand-scope convention to select an implementation.
    if (op.annotations_.Get("is_sqmma")) {
      return SelectSqmmaGemm(op, block_size, target);
    }
    if (op.annotations_.Get("is_wmma")) {
      return SelectWmmaGemm(op, block_size, target);
    }
    return SelectGemm(op, block_size, target);
  }

  static std::pair<int, int>
  ComputeWarpPartition(const GemmWarpPolicyNode &policy, int M, int N,
                       int block_size, Target target, String gemm_inst) {
    if (mp31::WMMA::IsInstruction(gemm_inst)) {
      return mp31::WMMA::ComputeWarpPartition(policy, M, N, block_size, target,
                                              gemm_inst);
    }
    if (mp31::SQMMA::IsInstruction(gemm_inst)) {
      return mp31::SQMMA::ComputeWarpPartition(policy, M, N, block_size, target,
                                               gemm_inst);
    }
    LOG(FATAL) << "MUSA GEMM has no warp partition implementation for target="
               << target << ", instruction=" << gemm_inst;
    return {0, 0};
  }

  static bool ReuseExistingSharedLayout(String gemm_inst) {
    if (mp31::WMMA::IsInstruction(gemm_inst)) {
      return mp31::WMMA::ReuseExistingSharedLayout(gemm_inst);
    }
    if (mp31::SQMMA::IsInstruction(gemm_inst)) {
      return mp31::SQMMA::ReuseExistingSharedLayout(gemm_inst);
    }
    LOG(FATAL) << "MUSA GEMM has no shared-layout policy for instruction "
               << gemm_inst;
    return false;
  }
};

} // namespace musa

namespace {

TVM_REGISTER_OP("tl.tileop.sqmma_gemm")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "sqmma_gemm")
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque))
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_sqmma",
                                       IntImm(DataType::Int(32), 1));
                               return Gemm(args, ann);
                             });

TVM_REGISTER_OP("tl.tileop.wmma_gemm")
    .set_attr<TScriptPrinterName>("TScriptPrinterName", "wmma_gemm")
    .set_attr<OpBuilderFunc>("TLOpBuilder",
                             [](Array<PrimExpr> args,
                                Map<String, ObjectRef> annotations) {
                               Map<String, ObjectRef> ann = annotations;
                               ann.Set("is_wmma", IntImm(DataType::Int(32), 1));
                               return Gemm(args, ann);
                             })
    .set_attr<TCallEffectKind>("TCallEffectKind",
                               Integer(CallEffectKind::kOpaque));

bool MatchMUSAGemmTarget(Target target) { return TargetIsMUSA(target); }

bool RegisterMUSAGemm() {
  RegisterGemmImpl(GemmImpl{
      "musa.Gemm",
      MatchMUSAGemmTarget,
      musa::Gemm::SelectInst,
      musa::Gemm::ComputeWarpPartition,
      musa::Gemm::ReuseExistingSharedLayout,
  });
  return true;
}

const bool musa_gemm_registered = RegisterMUSAGemm();

} // namespace

} // namespace tl
} // namespace tvm
