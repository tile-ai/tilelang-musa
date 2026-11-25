from __future__ import annotations
from tvm import IRModule
from tvm.target import Target
from tilelang import tvm as tvm
from typing import Any
import re
from .utils import match_declare_kernel, get_annotated_mod, pythonic_expr
from tvm.tirx.stmt_functor import post_order_visit

PREDEF_INIT_FUNC = """
#define ERROR_BUF_SIZE 1024
static char error_buf[ERROR_BUF_SIZE];

extern "C" const char* get_last_error() {{
    return error_buf;
}}

extern "C" int init() {{
    error_buf[0] = '\\0';
    {0}
    return 0;
}}
"""

PREDEF_HOST_FUNC = """
extern "C" int call({}) {{
{}
\treturn 0;
}}
"""

PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY = """
    musaError_t result_{0} = musaFuncSetAttribute({0}, musaFuncAttributeMaxDynamicSharedMemorySize, {1});
    if (result_{0} != musaSuccess) {{
        snprintf(error_buf, ERROR_BUF_SIZE, "Failed to set the allowed dynamic shared memory size to %d with error: %s", {1}, musaGetErrorString(result_{0}));
        return -1;
    }}
"""

L2_PERSISTENT_MAP_INIT_FUNC = """
\tstream_attribute.accessPolicyWindow.hitRatio = {1};
\tstream_attribute.accessPolicyWindow.hitProp = musaAccessPropertyPersisting;
\tstream_attribute.accessPolicyWindow.missProp = musaAccessPropertyStreaming;
\tmusaDeviceSetLimit(musaLimitPersistingL2CacheSize, {2});
\tstream_attribute.accessPolicyWindow.base_ptr = (void*)({0});
\tstream_attribute.accessPolicyWindow.num_bytes = {2};
\tmusaStreamSetAttribute(stream, musaStreamAttributeAccessPolicyWindow, &stream_attribute);
"""

L2_PERSISTENT_MAP_CREATE_HANDLE = """
\tmusaStreamAttrValue stream_attribute;
\tsize_t init_persisting_l2_cache_size;
\tmusaDeviceGetLimit(&init_persisting_l2_cache_size, musaLimitPersistingL2CacheSize);
"""

L2_PERSISTENT_MAP_RESET_HANDLE = """
\tstream_attribute.accessPolicyWindow.num_bytes = 0;
\tmusaStreamSetAttribute(stream, musaStreamAttributeAccessPolicyWindow, &stream_attribute);
\tmusaCtxResetPersistingL2Cache();
\tmusaDeviceSetLimit(musaLimitPersistingL2CacheSize, init_persisting_l2_cache_size);
"""

# 0: name, eg. A_desc, B_desc ...
# 1: date type
# 2: tensor rank
# 3: global address
# 4: dim
# 5: stride
# 6: box dim
# 7: element stride
TMA_DESC_INIT_FUNC = """
\tMUtensorDescriptor {0};
\tMUtensorDescriptorDataType {0}_type= (MUtensorDescriptorDataType){1};
\tmuuint32_t {0}_tensorRank= {2};
\tvoid *{0}_globalAddress= {3};
\tmuuint64_t {0}_globalDim[{2}]= {{{4}}};
\tmuuint64_t {0}_globalStride[{2}]= {{{5}}};
\tMUtensorDescriptorInterleave {0}_interleave= (MUtensorDescriptorInterleave){8};
\tmuuint32_t {0}_oobFill= {11};

\tMUresult {0}_result = muTensorDescriptorEncode(
    &{0}, {0}_type, {0}_tensorRank, {0}_globalAddress, {0}_globalDim, {0}_globalStride + 1, {0}_interleave, {0}_oobFill);

\tif ({0}_result != MUSA_SUCCESS) {{
\t\tstd::stringstream ss;
\t\tss << "Error: Failed to initialize the TMA descriptor {0}";
\t\tsnprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
\t\treturn -1;
\t}}
"""

TMA_IM2COL_DESC_INIT_FUNC = """
\tMUtensorDescriptor {0};
\tMUtensorDescriptorDataType {0}_type= (MUtensorDescriptorDataType){1};
\tmuuint32_t {0}_tensorRank= {2};
\tvoid *{0}_globalAddress= {3};
\tmuuint64_t {0}_globalDim[{2}]= {{{4}}};
\tmuuint64_t {0}_globalStride[{2}]= {{{5}}};
\tMUtensorDescriptorInterleave {0}_interleave= (MUtensorDescriptorInterleave){6};
\tmuuint64_t {0}_oobFill= {7};

\tMUresult {0}_desc_result = muTensorDescriptorEncode(
\t\t&{0}, {0}_type, {0}_tensorRank, {0}_globalAddress,
\t\t{0}_globalDim, {0}_globalStride + 1, {0}_interleave, {0}_oobFill);
\tif ({0}_desc_result != MUSA_SUCCESS) {{
\t\tstd::stringstream ss;
\t\tss << "Error: Failed to initialize the TME descriptor {0}";
\t\tsnprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
\t\treturn -1;
\t}}

\tMUconvParamer {0}_mu_conv_param{{}};
\tmuuint32_t {0}_numSpatial= {8};
\tmuuint32_t {0}_lowerPad[3]= {{static_cast<muuint32_t>({9}), static_cast<muuint32_t>({10}), static_cast<muuint32_t>({11})}};
\tmuuint32_t {0}_convStride[3]= {{static_cast<muuint32_t>({12}), static_cast<muuint32_t>({13}), static_cast<muuint32_t>({14})}};
\tmuuint32_t {0}_dilation[3]= {{static_cast<muuint32_t>({15}), static_cast<muuint32_t>({16}), static_cast<muuint32_t>({17})}};
\tMUresult {0}_conv_result = muTensorIm2colConvParamEncode(
\t\t&{0}_mu_conv_param, {0}_numSpatial, {0}_lowerPad,
\t\t{0}_convStride, {0}_dilation);
\tif ({0}_conv_result != MUSA_SUCCESS) {{
\t\tstd::stringstream ss;
\t\tss << "Error: Failed to initialize the TME im2col conv param {0}";
\t\tsnprintf(error_buf, ERROR_BUF_SIZE, "%s", ss.str().c_str());
\t\treturn -1;
\t}}
\tint* {0}_conv_param_raw = reinterpret_cast<int*>(&{0}_mu_conv_param);
\t__musa::tme_conv_param_t {0}_conv_param = {{{0}_conv_param_raw[0], {0}_conv_param_raw[1], {0}_conv_param_raw[2]}};
"""

TMA_IM2COL_OUTPUT_DIM_INIT_FUNC = """
\t{0} {1} = {{{2}}};
"""

ROBUST_DESC_INIT_FUNC = """
	tl::robust_param_desc_t {0} = tl::make_robust_desc(
		{1}, static_cast<uint64_t>({2}));
"""


class TLMUSASourceWrapper:
    _TYPE_MAP = {
        "float32": "float",
        "custom[tfloat32]": "tfloat32_t",
        "float16": "half_t",
        "bfloat16": "bfloat16_t",
        "float8_e4m3": "fp8_e4_t",
        "float8_e5m2": "fp8_e5_t",
        "float64": "double",
        "int64": "int64_t",
        "int32": "int",
        "uint64": "muuint64_t",
        "uint32": "unsigned int",
        "bool": "int8_t",
        "int8": "int8_t",
        "uint8": "uint8_t",
        "int16": "int16_t",
        "uint16": "uint16_t",
        "uchar": "uint8_t",
    }

    backend = "tl"
    device_mod: IRModule | None = None
    host_mod: IRModule | None = None
    pass_configs: dict[str, Any] | None = None

    def __init__(
        self,
        scheduled_ir_module: IRModule,
        source: str,
        target: Target,
        device_mod: IRModule | None = None,
        host_mod: IRModule | None = None,
        pass_configs: dict[str, Any] | None = None,
    ):
        self.mod = scheduled_ir_module
        self.target = target
        self.source = source
        self.pass_configs = pass_configs
        self.device_mod = device_mod
        self.host_mod = host_mod
        self.function_names: str | None = None
        self.dynamic_smem_buf: int | None = None
        self.block_info: list[int] | dict = [1, 1, 1]
        self.grid_info: list[int] | dict = [1, 1, 1]
        self.tma_descriptor_args: dict | None = None
        self.tma_im2col_output_dim_args: dict | None = None
        self.fast_divmod_host_args: dict | None = None
        self.robust_descriptor_args: dict | None = None
        self.l2_persistent_map: dict[str, dict] | None = {}
        self.parse_source_information()
        self.srcpath: str | None = None
        self.libpath: str | None = None
        self.lib_code: str | None = self.update_lib_code(source)

    def _pythonic_expr(self, expr: tvm.tirx.PrimExpr) -> str:
        return pythonic_expr(expr, self._TYPE_MAP, conditional_style="cpp")

    def _lookup_type(self, dtype: str | Any) -> str:
        key = dtype if isinstance(dtype, str) else str(dtype)
        result = self._TYPE_MAP.get(key)
        assert result is not None, f"Unsupported dtype {dtype}"
        return result

    def is_tma_descriptor_arg(self, arg_name: str) -> bool:
        return arg_name in self.prim_func.buffer_map

    def create_dispatch_func(self, code, function_informations):
        # Extract the set of dynamic symbolic names used in the primary function
        dynamic_symbolic_set = self.get_dynamic_symbolic_set(self.prim_func)

        function_args = []

        # Collect function arguments based on primary function's parameters and buffer mappings
        # QA(@lei): Why not use device_mod.params?
        # device func lack buffer map (to convert buffer handle to buffer)
        for param in self.prim_func.params:
            if param in self.prim_func.buffer_map:
                buffer = self.prim_func.buffer_map[param]
                function_args.append(
                    {
                        "name": buffer.data.name,
                        "type": self._lookup_type(buffer.dtype) + "* __restrict__",
                    }
                )
            elif isinstance(param, tvm.tirx.Var):
                function_args.append({"name": param.name, "type": self._lookup_type(param.dtype)})
            else:
                raise ValueError(f"Parameter {param} is not in the buffer map of the primary function.")
        # Add dynamic symbols as integer arguments
        for dyn_sym in dynamic_symbolic_set:
            if dyn_sym not in [arg["name"] for arg in function_args]:
                function_args.append({"name": dyn_sym, "type": "int"})

        function_args.append(self.get_stream_type())

        # Format the function arguments for declaration
        def_args = ", ".join([f"{arg['type']} {arg['name']}" for arg in function_args])

        def func_call_args(
            s,
            function_args,
            function_params,
            desc_name_map: dict[str, str] | None = None,
            desc_name_var_map: dict[str, tvm.tirx.Var] | None = None,
            fast_divmod_name_var_map: dict[str, tvm.tirx.PrimExpr] | None = None,
            robust_desc_name_var_map: dict[str, tvm.tirx.Var] | None = None,
        ):
            # Extract the function call arguments matching the function definition
            def maybe_desc(name: str, matches: list[str], i: int):
                match = matches[i]
                if not (match == name + "_desc" or match.startswith(name + "_desc_")):
                    return False
                # Types belong to a single comma-delimited kernel parameter.
                # Looking at adjacent tokens can borrow the next parameter's
                # descriptor type for an im2col output dimension.
                is_descriptor = match in descriptor_names
                if is_descriptor and desc_name_map is not None:
                    desc_name_map[match] = name
                return is_descriptor

            pattern = r"[,\s]*(?:\w+\s*\*+\s*__restrict__\s+)?(\w+)"
            # The caller supplies a complete kernel declaration, including its
            # name and trailing body. Match types only inside the parameter list.
            parameter_list = s[s.index("(") + 1 : s.rindex(")")] if "(" in s and ")" in s else s
            matches = re.findall(pattern, parameter_list)
            descriptor_names = {
                decl.group(1)
                for parameter in parameter_list.split(",")
                # Codegen can prefix descriptors with grid_constant and const.
                # Keep the end anchor so adjacent scalar parameters stay scalar.
                if (decl := re.search(r"\bMUtensorDescriptor\s+(\w+)\s*$", parameter))
            }
            robust_desc_names = set(re.findall(r"tl::robust_param_desc_t\s+(\w+)", parameter_list))
            call_args = []
            for i, match in enumerate(matches):
                if match.startswith("fast_div_multiplier_") or match.startswith("fast_div_shift_") or match in robust_desc_names:
                    call_args.append(match)
                    assert len(call_args) <= len(function_params), (
                        f"Function {function_name} has {len(function_params)} parameters, but {len(call_args)} arguments"
                    )
                    if fast_divmod_name_var_map is not None and (
                        match.startswith("fast_div_multiplier_") or match.startswith("fast_div_shift_")
                    ):
                        fast_divmod_name_var_map[match] = function_params[len(call_args) - 1]
                    if robust_desc_name_var_map is not None and match in robust_desc_names:
                        robust_desc_name_var_map[match] = function_params[len(call_args) - 1]
                    continue
                for arg in function_args:
                    if arg["name"] == match:
                        call_args.append(match)
                    elif maybe_desc(arg["name"], matches, i):
                        call_args.append(match)
                        assert len(call_args) <= len(function_params), (
                            f"Function {function_name} has {len(function_params)} parameters, but {len(call_args)} arguments"
                        )
                        desc_name_var_map[match] = function_params[len(call_args) - 1]
                if match.endswith("_conv_param") and match not in call_args:
                    desc_handle = match[: -len("_conv_param")]
                    if desc_handle in desc_name_map or any(match.startswith(name) for name in desc_name_map):
                        call_args.append(match)
                if match.endswith("_output_dim") and match not in call_args:
                    desc_handle = match[: -len("_output_dim")]
                    if desc_handle in desc_name_map or any(match.startswith(name) for name in desc_name_map):
                        call_args.append(match)

            return call_args

        has_l2_persistent_map = False
        for function_name, _ in function_informations.items():
            if function_name in self.l2_persistent_map:
                has_l2_persistent_map = True
                break

        kernel_launch_code = """"""
        if has_l2_persistent_map:
            kernel_launch_code += L2_PERSISTENT_MAP_CREATE_HANDLE
        desc_name_map: dict[str, str] = {}
        desc_name_var_map: dict[str, tvm.tirx.Var] = {}
        fast_divmod_name_var_map: dict[str, tvm.tirx.PrimExpr] = {}
        robust_desc_name_var_map: dict[str, tvm.tirx.Var] = {}
        for function_name, function_info in function_informations.items():
            block_info = function_info["block_info"]
            grid_info = function_info["grid_info"]
            dynamic_smem_buf = function_info["dynamic_smem_buf"]
            function_params = function_info["function_params"]

            # Find the location of the global kernel function in the code
            index = match_declare_kernel(code, function_name + "(")

            # Analyze the function declaration to prepare for argument extraction
            # Restrict parsing to the signature: body calls can contain
            # parentheses and names that resemble kernel parameters.
            declaration_end = min(pos for token in (";", "{") if (pos := code.find(token, index)) != -1)
            declaration = code[index:declaration_end]

            # Identify the start of the function body to insert arguments
            index = code.index("{", index)

            block_str = (
                f"dim3({self._pythonic_expr(block_info[0])}, {self._pythonic_expr(block_info[1])}, {self._pythonic_expr(block_info[2])})"
            )
            grid_str = (
                f"dim3({self._pythonic_expr(grid_info[0])}, {self._pythonic_expr(grid_info[1])}, {self._pythonic_expr(grid_info[2])})"
            )
            smem_str = 0 if dynamic_smem_buf is None else dynamic_smem_buf
            init_l2_persistent_map = self.generate_l2_persistent_map(function_name)
            kernel_launch_code += init_l2_persistent_map

            if self.use_cooperative_groups[function_name]:
                args_list = func_call_args(
                    declaration,
                    function_args,
                    function_params,
                    desc_name_map,
                    desc_name_var_map,
                    fast_divmod_name_var_map,
                    robust_desc_name_var_map,
                )
                assert len(function_params) == len(args_list), (
                    f"Function {function_name} has {len(function_params)} parameters, but {len(args_list)} arguments"
                )
                args_array = [f"(void*)&{arg}" for arg in args_list]
                call_args = f"\tvoid* {function_name}_args[] = {{{', '.join(args_array)}}};\n"
                kernel_launch_code += call_args
                # Using musaLaunchCooperativeKernel to launch the kernel
                kernel_launch_code += "\tTILELANG_CHECK(musaLaunchCooperativeKernel((void*){}, {}, {}, {}, {}, stream));\n".format(
                    function_name, grid_str, block_str, function_name + "_args", smem_str
                )
            else:
                args_list = func_call_args(
                    declaration,
                    function_args,
                    function_params,
                    desc_name_map,
                    desc_name_var_map,
                    fast_divmod_name_var_map,
                    robust_desc_name_var_map,
                )
                assert len(function_params) == len(args_list), (
                    f"Function {function_name} has {len(function_params)} parameters, but {len(args_list)} arguments"
                )
                call_args = ", ".join(args_list)
                kernel_launch_code += f"\t{function_name}<<<{grid_str}, {block_str}, {smem_str}, stream>>>({call_args});\n"
                kernel_launch_code += f'\tTILELANG_CHECK_LAST_ERROR("{function_name}");\n'
            if has_l2_persistent_map:
                kernel_launch_code += L2_PERSISTENT_MAP_RESET_HANDLE

        init_tma_descriptor_args = self.generate_tma_descriptor_args(desc_name_map, desc_name_var_map)
        init_tma_output_dim_args = self.generate_tma_output_dim_args()
        init_fast_divmod_host_args = self.generate_fast_divmod_host_args(fast_divmod_name_var_map)
        init_robust_descriptor_args = self.generate_robust_descriptor_args(robust_desc_name_var_map)
        kernel_launch_code = init_tma_descriptor_args + init_tma_output_dim_args + init_fast_divmod_host_args + init_robust_descriptor_args + kernel_launch_code

        # Wrap the kernel dispatch logic in an external C function
        host_func = PREDEF_HOST_FUNC.format(def_args, kernel_launch_code)
        return host_func

    def generate_fast_divmod_host_args(self, name_var_map: dict[str, tvm.tirx.PrimExpr]) -> str:
        if self.fast_divmod_host_args is None:
            return ""

        def find_name(var: tvm.tirx.PrimExpr) -> str:
            for name, mapped_var in name_var_map.items():
                if mapped_var.same_as(var):
                    return name
            raise ValueError(f"Fast-divmod kernel parameter {var} was not found in the generated declaration")

        init = ""
        for index, (multiplier_var, args) in enumerate(self.fast_divmod_host_args.items()):
            if len(args) != 2:
                raise ValueError(f"Fast-divmod host args must be (divisor, shift_var), got {len(args)} entries")
            divisor, shift_var = args
            multiplier_name = find_name(multiplier_var)
            shift_name = find_name(shift_var)
            divisor_expr = self._pythonic_expr(divisor)
            divisor_name = f"fast_div_divisor_{index}"
            params_name = f"fast_divmod_host_{index}"
            init += f"\tint32_t {divisor_name} = {divisor_expr};\n"
            init += (
                f"\tif ({divisor_name} <= 0) {{ snprintf(error_buf, ERROR_BUF_SIZE, "
                f'"T.fast_div divisor must be strictly positive"); return -1; }}\n'
            )
            init += f"\tmutlass::FastDivmod {params_name}({divisor_name});\n"
            init += f"\tuint32_t {multiplier_name} = {params_name}.multiplier;\n"
            init += f"\tuint32_t {shift_name} = {params_name}.shift_right;\n"
        return init

    def generate_l2_persistent_map(self, function_name: str) -> str:
        if function_name not in self.l2_persistent_map:
            return ""
        init_l2_persistent_map = ""
        for buffer_name, (hit_ratio, size_in_bytes) in self.l2_persistent_map[function_name].items():
            # get persisting_l2_cache_max_size
            from tilelang.carver.arch.driver import get_persisting_l2_cache_max_size

            persisting_l2_cache_max_size = get_persisting_l2_cache_max_size()
            try:
                num_bytes = min(size_in_bytes, persisting_l2_cache_max_size)
            except Exception:
                # as size_in_bytes maybe a symbolic expression
                num_bytes = persisting_l2_cache_max_size
            init_l2_persistent_map += L2_PERSISTENT_MAP_INIT_FUNC.format(buffer_name, float(hit_ratio), self._pythonic_expr(num_bytes))

        return init_l2_persistent_map

    def generate_tma_descriptor_args(self, desc_name_map: dict[str, str], desc_name_var_map: dict[str, tvm.tirx.Var]) -> str:
        tma_descripter_init = ""
        if self.tma_descriptor_args is None:
            return tma_descripter_init
        for handle_name, _ in desc_name_map.items():
            assert handle_name in desc_name_var_map, f"Handle name {handle_name} not found in desc_name_var_map"
            desc_var = desc_name_var_map[handle_name]

            assert desc_var in self.tma_descriptor_args, f"TMA descriptor {desc_var} not found in {self.tma_descriptor_args}"
            args = self.tma_descriptor_args[desc_var]
            # Skip __tvm_tensormap_create_tiled
            if len(args) < 3:
                raise ValueError(f"TMA descriptor args too short: {len(args)} elements, expected at least 3")

            tma_create_str, _, dtype, tensor_rank, globalAddress, *remaining_args = args

            is_img2col = tma_create_str.value == "__tvm_musa_tensormap_create_im2col"
            dtype = self._pythonic_expr(dtype)
            tensor_rank = int(self._pythonic_expr(tensor_rank))

            # Validate tensor_rank
            if not isinstance(tensor_rank, int) or tensor_rank <= 0:
                raise ValueError(f"Invalid tensor_rank: {tensor_rank}. Must be a positive integer")

            if not is_img2col:
                # Calculate required length for remaining_args
                expected_args_len = 4 * tensor_rank + 4  # 4 groups of tensor_rank size + 4 parameters
                if len(remaining_args) < expected_args_len:
                    raise ValueError(
                        f"Insufficient remaining args: got {len(remaining_args)}, "
                        f"expected {expected_args_len} for tensor_rank {tensor_rank}"
                    )

                # Extract dimensions and strides using list slicing
                global_dim = remaining_args[:tensor_rank]
                global_stride = remaining_args[tensor_rank : 2 * tensor_rank]
                box_dim = remaining_args[2 * tensor_rank : 3 * tensor_rank]
                element_strides = remaining_args[3 * tensor_rank : 4 * tensor_rank]

                global_dim = [self._pythonic_expr(i) for i in global_dim]
                global_stride = [self._pythonic_expr(i) for i in global_stride]
                box_dim = [self._pythonic_expr(i) for i in box_dim]
                element_strides = [self._pythonic_expr(i) for i in element_strides]

                # Extract remaining parameters
                try:
                    interleave, swizzle, l2Promotion, oobFill = remaining_args[4 * tensor_rank : 4 * tensor_rank + 4]
                    interleave = self._pythonic_expr(interleave)
                    swizzle = self._pythonic_expr(swizzle)
                    l2Promotion = self._pythonic_expr(l2Promotion)
                    oobFill = self._pythonic_expr(oobFill)
                except ValueError as e:
                    raise ValueError("Failed to unpack the final 4 TMA parameters (interleave, swizzle, l2Promotion, oobFill)") from e

                tma_descripter_init += TMA_DESC_INIT_FUNC.format(
                    handle_name,
                    dtype,
                    tensor_rank,
                    globalAddress,
                    ",".join(global_dim),
                    ",".join(global_stride),
                    ",".join(box_dim),
                    ",".join(element_strides),
                    interleave,
                    swizzle,
                    l2Promotion,
                    oobFill,
                )
            else:
                # MUSA im2col descriptors use the ordinary tensor descriptor
                # plus a separately encoded convolution parameter.
                expected_args_len = 2 * tensor_rank + 15
                if len(remaining_args) < expected_args_len:
                    raise ValueError(
                        f"Insufficient remaining args: got {len(remaining_args)}, "
                        f"expected {expected_args_len} for tensor_rank {tensor_rank}"
                    )

                global_dim = [self._pythonic_expr(i) for i in remaining_args[:tensor_rank]]
                global_stride = [self._pythonic_expr(i) for i in remaining_args[tensor_rank : 2 * tensor_rank]]
                tail = remaining_args[2 * tensor_rank :]
                try:
                    (
                        pad_w,
                        pad_h,
                        pad_d,
                        dilation_w,
                        dilation_h,
                        dilation_d,
                        stride_w,
                        stride_h,
                        stride_d,
                        interleave,
                        oob_fill,
                        num_spatial,
                        _smem_box_pixel,
                        _smem_box_channel,
                        _swizzle,
                    ) = tail[:15]
                    interleave = self._pythonic_expr(interleave)
                    oob_fill = self._pythonic_expr(oob_fill)
                    num_spatial = self._pythonic_expr(num_spatial)
                    pad_w, pad_h, pad_d = (self._pythonic_expr(v) for v in (pad_w, pad_h, pad_d))
                    stride_w, stride_h, stride_d = (self._pythonic_expr(v) for v in (stride_w, stride_h, stride_d))
                    dilation_w, dilation_h, dilation_d = (self._pythonic_expr(v) for v in (dilation_w, dilation_h, dilation_d))
                except ValueError as e:
                    raise ValueError("Failed to unpack MUSA im2col descriptor parameters") from e

                tma_descripter_init += TMA_IM2COL_DESC_INIT_FUNC.format(
                    handle_name,
                    dtype,
                    tensor_rank,
                    globalAddress,
                    ",".join(global_dim),
                    ",".join(global_stride),
                    interleave,
                    oob_fill,
                    num_spatial,
                    pad_w,
                    pad_h,
                    pad_d,
                    stride_w,
                    stride_h,
                    stride_d,
                    dilation_w,
                    dilation_h,
                    dilation_d,
                )

        return tma_descripter_init

    def generate_tma_output_dim_args(self) -> str:
        output_dim_init = ""
        if self.tma_im2col_output_dim_args is None:
            return output_dim_init
        for output_dim_var, args in self.tma_im2col_output_dim_args.items():
            lanes = len(args)
            if lanes == 2:
                musa_type = "v2i32_t"
            elif lanes == 3:
                musa_type = "v3i32_t"
            else:
                raise ValueError(f"Unsupported im2col output-dimension lanes: {lanes}")
            output_dim_init += TMA_IM2COL_OUTPUT_DIM_INIT_FUNC.format(
                musa_type,
                output_dim_var.name,
                ", ".join(self._pythonic_expr(arg) for arg in args),
            )
        return output_dim_init

    def _robust_pointer_expr(self, expr: tvm.tirx.PrimExpr) -> str:
        if isinstance(expr, tvm.tirx.Var):
            return expr.name
        if isinstance(expr, tvm.tirx.Call) and getattr(expr.op, "name", "") in {
            "tir.handle_add_byte_offset",
            "tirx.handle_add_byte_offset",
        }:
            if len(expr.args) != 2:
                raise ValueError(f"Unsupported robust descriptor byte offset: {expr}")
            base = self._robust_pointer_expr(expr.args[0])
            offset = self._pythonic_expr(expr.args[1])
            return f"(reinterpret_cast<const char*>({base}) + ({offset}))"
        if isinstance(expr, tvm.tirx.Call) and getattr(expr.op, "name", "") in {
            "tir.address_of",
            "tirx.address_of",
        }:
            if len(expr.args) != 1 or not isinstance(expr.args[0], tvm.tirx.BufferLoad):
                raise ValueError(f"Unsupported robust descriptor address: {expr}")
            load = expr.args[0]
            if len(load.indices) != 1:
                raise ValueError(f"Expected flattened robust descriptor address, got {expr}")
            index = self._pythonic_expr(load.indices[0])
            return f"(&({load.buffer.data.name}[{index}]))"
        raise ValueError(f"Unsupported robust descriptor base pointer: {expr}")

    def generate_robust_descriptor_args(self, desc_name_var_map: dict[str, tvm.tirx.Var]) -> str:
        if self.robust_descriptor_args is None:
            return ""

        init = ""
        for handle_name, desc_var in desc_name_var_map.items():
            if desc_var not in self.robust_descriptor_args:
                raise ValueError(f"Robust descriptor {desc_var} not found in {self.robust_descriptor_args}")
            args = self.robust_descriptor_args[desc_var]
            if len(args) != 4:
                raise ValueError(f"Robust descriptor args must be (creator, desc, base, size), got {len(args)}")
            _, _, base, size = args
            init += ROBUST_DESC_INIT_FUNC.format(
                handle_name,
                self._robust_pointer_expr(base),
                self._pythonic_expr(size),
            )
        return init

    def parse_source_information(self):
        if self.device_mod is None or self.host_mod is None:
            with tvm.transform.PassContext(opt_level=3, config=self.pass_configs):
                device_mod, host_mod = get_annotated_mod(self.mod, self.target)
            self.device_mod = device_mod
            self.host_mod = host_mod
        assert len(self.device_mod.functions) >= 1, "Device module should have at least one function."
        assert len(self.host_mod.functions) == 1, "Only support one function in host module."

        block_info_map = {}
        grid_info_map = {}
        dynamic_smem_buf_map = {}
        function_names = []
        use_cooperative_groups_map = {}
        for g_var, func in self.device_mod.functions.items():
            # Default block and grid configurations
            block_info = [1, 1, 1]
            grid_info = [1, 1, 1]
            function_name = g_var.name_hint
            attrs = func.attrs
            dynamic_smem_buf = None
            use_cooperative_groups = False
            if "use_cooperative_groups" in attrs:
                use_cooperative_groups = attrs["use_cooperative_groups"]
            if "dyn_shared_memory_buf" in attrs:
                dynamic_smem_buf = int(attrs["dyn_shared_memory_buf"])
            if "thread_extent" in attrs:
                # Extract block and grid sizes from thread extents
                thread_extent = attrs["thread_extent"]
                for tag, extent in thread_extent.items():
                    if "threadIdx" in tag:
                        block_info["xyz".index(tag[-1])] = extent
                    elif "blockIdx" in tag:
                        grid_info["xyz".index(tag[-1])] = extent
            # Map the extracted configurations to each function
            block_info_map[function_name] = block_info
            grid_info_map[function_name] = grid_info
            dynamic_smem_buf_map[function_name] = dynamic_smem_buf
            use_cooperative_groups_map[function_name] = use_cooperative_groups
            function_names.append(function_name)

        # Store the mappings for use in code generation
        self.block_info = block_info_map
        self.grid_info = grid_info_map
        self.dynamic_smem_buf = dynamic_smem_buf_map
        self.use_cooperative_groups = use_cooperative_groups_map

        function_names_index = {}
        for _, func in self.host_mod.functions.items():
            if "tma_descriptor_args" in func.attrs:
                self.tma_descriptor_args = func.attrs["tma_descriptor_args"]
            if "tma_im2col_output_dim_args" in func.attrs:
                self.tma_im2col_output_dim_args = func.attrs["tma_im2col_output_dim_args"]
            if "fast_divmod_host_args" in func.attrs:
                self.fast_divmod_host_args = func.attrs["fast_divmod_host_args"]
            if "robust_descriptor_args" in func.attrs:
                self.robust_descriptor_args = func.attrs["robust_descriptor_args"]
            if "l2_persistent_map" in func.attrs:
                self.l2_persistent_map[function_name] = func.attrs["l2_persistent_map"]

            host_code = str(func)
            for function_name in function_names:
                index = host_code.index(f'T.call_packed("{function_name}"')
                function_names_index[function_name] = index
        # sort function_names
        function_names = sorted(function_names, key=lambda x: function_names_index[x])
        self.function_names = function_names

    def get_dynamic_symbolic_set(self, prim_func):
        # Determine the set of dynamic symbols used in the function
        dynamic_symbolic_set: list[str] = []

        def unique_push_back(name: str):
            if name not in dynamic_symbolic_set:
                dynamic_symbolic_set.append(name)

        for param in prim_func.params:
            if param in prim_func.buffer_map:
                buffer = prim_func.buffer_map[param]
                for dim in buffer.shape:
                    if isinstance(dim, tvm.tirx.Var):
                        unique_push_back(dim.name)

        # Note: In buffer definitions, any dynamic symbols appearing in strides are listed after those in the shape.
        for param in prim_func.params:
            if param in prim_func.buffer_map:
                buffer = prim_func.buffer_map[param]
                for stride in buffer.strides:
                    if isinstance(stride, tvm.tirx.Var):
                        unique_push_back(stride.name)

        return dynamic_symbolic_set

    def get_init_func(self):
        # Initialize an empty string for the MUSA function call
        call_str = """"""
        # If dynamic shared memory buffer is specified, prepare the musaFuncSetAttribute call
        for function_name, dynamic_smem_buf in self.dynamic_smem_buf.items():
            if dynamic_smem_buf is not None:
                # Format the musaFuncSetAttribute call for dynamic shared memory
                call_str += PREDEF_ATTRIBUTE_SET_DYNAMIC_MEMORY.format(function_name, dynamic_smem_buf)
        # Format the initialization function using the call_str
        init_funcs = PREDEF_INIT_FUNC.format(call_str)
        return init_funcs

    def update_lib_code(self, code: str):
        # Update the library code with the given code string
        self.lib_code = code
        # Get the function names
        function_names = self.function_names
        # Get the MUSA initialization function
        init_func = self.get_init_func()

        # Organize function information for code generation
        function_informations = {}
        for function_name in function_names:
            # Do not update function with dispatch host function
            if (function_name not in self.block_info) or (function_name not in self.grid_info):
                continue
            assert function_name in self.device_mod, f"Function {function_name} not found in device module"
            device_func = self.device_mod[function_name]
            kernel_params_cnt = len(device_func.params)
            function_params: list[str] = None

            def visitor(node, fn=function_name, param_cnt=kernel_params_cnt):
                nonlocal function_params
                if isinstance(node, tvm.tirx.Call):
                    if not (hasattr(node, "op") and node.op == tvm.ir.Op.get("tirx.tvm_call_packed")):
                        return
                    args = node.args
                    if not args or args[0] != fn:
                        return
                    if len(args) < 1 + param_cnt:
                        raise AssertionError("tvm_call_packed should have at least 1 argument and match device function parameters")
                    function_params = args[1 : 1 + param_cnt]

            post_order_visit(self.host_func.body, visitor)
            assert function_params is not None, "function_params should not be None"

            function_informations[function_name] = {
                "function_name": function_name,
                "block_info": self.block_info[function_name],
                "grid_info": self.grid_info[function_name],
                "dynamic_smem_buf": self.dynamic_smem_buf[function_name],
                "function_params": function_params,
            }

        # Create the host function wrapper for the MUSA kernel
        host_func = self.create_dispatch_func(code, function_informations)
        # Combine the source, initialization function, and host function to form the complete library code
        lib_code = self.source + init_func + host_func
        return lib_code

    def get_stream_type(self) -> dict[str, str]:
        return {"name": "stream=musaStreamDefault", "type": "musaStream_t"}

    @property
    def prim_func(self):
        if len(self.mod.get_global_vars()) == 1:
            return self.mod[self.mod.get_global_vars()[0]]
        elif "main" in self.mod:
            return self.mod["main"]
        else:
            for _, function in self.mod.functions_items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")

    @property
    def device_func(self):
        if len(self.device_mod.get_global_vars()) == 1:
            return self.device_mod[self.device_mod.get_global_vars()[0]]
        elif "main" in self.device_mod:
            return self.device_mod["main"]
        else:
            for _, function in self.device_mod.functions.items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")

    @property
    def host_func(self):
        if len(self.host_mod.get_global_vars()) == 1:
            return self.host_mod[self.host_mod.get_global_vars()[0]]
        elif "main" in self.host_mod:
            return self.host_mod["main"]
        else:
            for _, function in self.host_mod.functions.items():
                attr = function.attrs
                if "tir.is_global_func" in attr and attr["tir.is_global_func"]:
                    return function
            raise ValueError("Cannot find primary function in the module.")
