from .pass_pipeline import PassPipeline, register_lazy_pipeline  # noqa: F401
from .device_codegen import DeviceCodegen  # noqa: F401
from .host_codegen import HostCodegen, HostCodegenHook  # noqa: F401
from .execution_backend import ExecutionBackendSpec  # noqa: F401
from .module import (  # noqa: F401
    BackendContext,
    BackendModule,
    create_backend_context,
    get_backend,
    list_backends,
    register_backend,
)
from .runtime_device import (  # noqa: F401
    RuntimeDevice,
    register_runtime_device,
    resolve_runtime_device,
)
from .target import (  # noqa: F401
    auto_detect_target,
    list_target_detectors,
    register_target_detector,
    register_target_normalizer,
)

register_lazy_pipeline("musa", "tilelang.musa.pipeline")
