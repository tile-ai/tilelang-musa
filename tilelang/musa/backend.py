"""MUSA backend manifest for the unified backend registry."""

from tilelang.backend.host_codegen import STANDARD_HOST_CODEGENS
from tilelang.backend.module import BackendModule, register_backend
from tilelang.backend.pass_pipeline import PassPipeline

from . import codegen, execution_backend, pipeline
from .target import target_get_arch


def _supports_mp31(target):
    return target_get_arch(target) == "mp_31"


BACKEND = register_backend(
    BackendModule(
        name="musa",
        target_kinds=("musa",),
        supports_target=_supports_mp31,
        pipelines={"musa": PassPipeline("musa", pipeline.MUSAPassPipelineBody)},
        device_codegens={"musa": codegen.DEVICE_CODEGEN},
        execution_backends=execution_backend.EXECUTION_BACKENDS,
        host_codegens=STANDARD_HOST_CODEGENS,
    )
)
