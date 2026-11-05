#include "codegen_musa.h"
#include "runtime/musa/musa_module.h"
#include "runtime/pack_args.h"
#include "support/check.h"
#include "transform/common/attr.h"
#include <tvm/ffi/reflection/registry.h>
#include <tvm/ir/cast.h>
#include <tvm/ir/transform.h>

namespace tvm {
namespace codegen {

using namespace ffi;

static Map<String, runtime::FunctionInfo> ExtractFuncInfo(const IRModule &mod) {
  Map<String, runtime::FunctionInfo> fmap;

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<tirx::PrimFuncNode>())
        << "Can only lower IR Module with PrimFuncs";
    auto f = Downcast<tirx::PrimFunc>(kv.second);

    Array<DLDataType> arg_types;
    Array<String> launch_param_tags;
    for (size_t i = 0; i < f->params.size(); ++i) {
      if (f->params[i]->dtype.is_handle()) {
        auto ptr = f->params[i]->type_annotation.as<PointerTypeNode>();
        if (ptr && ptr->storage_scope == "grid_constant") {
          arg_types.push_back(DataType(runtime::kDLGridConstant, 64, 1));
          continue;
        }
      }
      DataType dtype = f->params[i].dtype();
      // Device runtime cannot directly take bool arguments, map to int32.
      if (dtype.is_bool())
        dtype = DataType::Int(32);
      arg_types.push_back(dtype);
    }
    if (f->HasNonzeroAttr(tl::attr::kHasGridSync)) {
      launch_param_tags.push_back(
          runtime::launch_param::kUseProgramaticDependentLaunch);
    }
    if (f->HasNonzeroAttr("use_cooperative_groups")) {
      launch_param_tags.push_back(runtime::launch_param::kUseCooperativeLaunch);
    }
    if (f->GetAttr<Array<Integer>>("cluster_dims").defined()) {
      launch_param_tags.push_back(runtime::launch_param::kClusterDimX);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimY);
      launch_param_tags.push_back(runtime::launch_param::kClusterDimZ);
    }
    if (auto opt = f->GetAttr<Array<String>>(tirx::attr::kKernelLaunchParams)) {
      for (const auto &tag : opt.value()) {
        if (tag != runtime::launch_param::kClusterDimX &&
            tag != runtime::launch_param::kClusterDimY &&
            tag != runtime::launch_param::kClusterDimZ) {
          launch_param_tags.push_back(tag);
        }
      }
    }
    auto global_symbol = f->GetAttr<String>(tvm::attr::kGlobalSymbol);
    std::string name = static_cast<std::string>(global_symbol.value());
    fmap.Set(String(name), runtime::FunctionInfo(String(name), arg_types,
                                                 launch_param_tags, {}));
  }
  return fmap;
}

ffi::Module BuildTileLangMUSA(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangMUSA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangMUSA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_musa_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  std::string fmt = "ptx";
  std::string mubin_or_path;
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_musa_compile")) {
    // Fetch current pass context config and pass into the compile callback
    tvm::transform::PassContext pass_ctx =
        tvm::transform::PassContext::Current();
    mubin_or_path = (*f)(code, target, pass_ctx->config).cast<std::string>();
    if (!mubin_or_path.empty() && mubin_or_path[0] != '/') {
      fmt = "mubin";
    }
  } else {
    ICHECK(0);
  }
  return runtime::MUSAModuleCreate(mubin_or_path, fmt, ExtractFuncInfo(mod),
                                   code);
}

ffi::Module BuildTileLangMUSAWithoutCompile(IRModule mod, Target target) {
  bool output_ssa = false;
  CodeGenTileLangMUSA cg;
  cg.Init(output_ssa);

  for (auto kv : mod->functions) {
    ICHECK(kv.second->IsInstance<PrimFuncNode>())
        << "CodeGenTileLangMUSA: Can only take PrimFunc";
    auto gvar = Downcast<GlobalVar>(kv.first);
    auto f = Downcast<PrimFunc>(kv.second);
    auto calling_conv = f->GetAttr<Integer>(tvm::attr::kCallingConv);
    ICHECK(calling_conv == CallingConv::kDeviceKernelLaunch);
    cg.AddFunction(gvar, f);
  }

  std::string code = cg.Finish();
  if (const auto f =
          ffi::Function::GetGlobal("tilelang_callback_musa_postproc")) {
    code = (*f)(code, target).cast<std::string>();
  }
  return CSourceModuleCreate(code, "mu", {});
}

TVM_FFI_STATIC_INIT_BLOCK() {
  namespace refl = tvm::ffi::reflection;
  refl::GlobalDef()
      .def("target.build.tilelang_musa", BuildTileLangMUSA)
      .def("target.build.tilelang_musa_without_compile",
           BuildTileLangMUSAWithoutCompile);
}

} // namespace codegen
} // namespace tvm
