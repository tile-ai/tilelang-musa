# pylint: disable=invalid-name
# modified from apache tvm python/tvm/contrib/mcc.py
"""Utility to invoke mcc compiler in the system"""

from __future__ import annotations

import os
import shlex
import subprocess
import tempfile
from tilelang.env import (
    ENV_FALSE_VALUES,
    ENV_TRUE_VALUES,
    MUSA_HOME,
    MUTLASS_INCLUDE_DIR,
    TILELANG_TEMPLATE_PATH,
    TILELANG_DUMP_ASM_ENV,
    TILELANG_DUMP_KERNEL_SOURCE_ENV,
    TILELANG_PRINT_ASM_ENV,
    TILELANG_REPLACE_ASM_ENV,
    env,
    env_enabled_or_path,
    env_path_or_default,
)
import tvm_ffi
from tilelang import tvm as tvm
from tvm.target import Target

from tvm.base import py_str
from tvm.contrib import utils


def _resolve_artifact_paths(temp, file_name, target_format, kernels_output_dir=None):
    if kernels_output_dir is None:
        return temp.relpath(f"{file_name}.mu"), temp.relpath(f"{file_name}.{target_format}")

    os.makedirs(kernels_output_dir, exist_ok=True)
    fd, temp_code = tempfile.mkstemp(prefix=f"{file_name}_", suffix=".mu", dir=kernels_output_dir)
    os.close(fd)
    file_stem, _ = os.path.splitext(os.path.basename(temp_code))
    temp_target = os.path.join(kernels_output_dir, f"{file_stem}.{target_format}")
    return temp_code, temp_target


def _resolve_dump_path(value: str, default_name: str) -> str:
    if value.lower() in ENV_TRUE_VALUES:
        return os.path.join(os.getcwd(), default_name)

    dump_path = os.path.abspath(os.path.expanduser(value))
    if dump_path.endswith(os.sep) or os.path.isdir(dump_path):
        return os.path.join(dump_path, default_name)
    return dump_path


def _shell_join(cmd: list[str]) -> str:
    return " ".join(shlex.quote(str(arg)) for arg in cmd)


def _mcc_target_ext(cmd: list[str]) -> str:
    output_idx = cmd.index("-o")
    _, target_ext = os.path.splitext(str(cmd[output_idx + 1]))
    return target_ext or ".mubin"


def _next_numbered_dump_path(dump_path: str, target_ext: str) -> str:
    dump_prefix, dump_ext = os.path.splitext(dump_path)
    for idx in range(10000):
        candidate = f"{dump_prefix}_{idx:04d}{dump_ext}"
        candidate_prefix, _ = os.path.splitext(candidate)
        related_paths = (
            candidate,
            candidate + ".cmd",
            candidate_prefix + target_ext,
            candidate_prefix + ".s",
        )
        if not any(os.path.exists(path) for path in related_paths):
            return candidate
    raise RuntimeError(f"Cannot find an available numbered dump path for {dump_path}")


def _mcc_asm_cmd(cmd: list[str], file_target: str) -> list[str]:
    prefix, _ = os.path.splitext(file_target)
    asm_path = prefix + ".s"
    output_idx = cmd.index("-o")
    return cmd[:output_idx] + ["-S", "-o", asm_path] + cmd[output_idx + 2 :]


def _retarget_mcc_dump_cmd(cmd: list[str], source_path: str, dump_path: str) -> list[str]:
    output_idx = cmd.index("-o")
    target_ext = _mcc_target_ext(cmd)

    dump_prefix, _ = os.path.splitext(dump_path)
    retargeted_cmd = list(cmd)
    retargeted_cmd[output_idx + 1] = dump_prefix + target_ext
    for idx, arg in enumerate(retargeted_cmd):
        if str(arg) == source_path:
            retargeted_cmd[idx] = dump_path
    return retargeted_cmd


def _dump_kernel_source(source_path: str, code: str, cmd: list[str]) -> None:
    value = os.environ.get(TILELANG_DUMP_KERNEL_SOURCE_ENV)
    value = value.strip() if value is not None else ""
    if value == "" or value.lower() in ENV_FALSE_VALUES:
        return

    dump_path = _resolve_dump_path(value, "tilelang_musa_kernel.mu")
    dump_path = _next_numbered_dump_path(dump_path, _mcc_target_ext(cmd))
    os.makedirs(os.path.dirname(dump_path), exist_ok=True)
    with open(dump_path, "w", encoding="utf-8") as f:
        f.write(code)
    compile_cmd = _retarget_mcc_dump_cmd(cmd, source_path, dump_path)
    command_path = dump_path + ".cmd"
    with open(command_path, "w", encoding="utf-8") as f:
        f.write("# compile\n")
        f.write(_shell_join(compile_cmd))
        f.write("\n\n# asm\n")
        f.write(_shell_join(_mcc_asm_cmd(compile_cmd, compile_cmd[compile_cmd.index("-o") + 1])))
        f.write("\n")
    print(f"compile_musa dumped kernel source to: {dump_path}")
    print(f"compile_musa dumped kernel compile command to: {command_path}")


def _run_logged_command(cmd: list[str], stdout_path: str | None = None) -> str:
    suffix = f" > {stdout_path}" if stdout_path else ""
    print(f"compile_musa command: {_shell_join(cmd)}{suffix}")
    proc = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, check=False)
    output = py_str(proc.stdout) if proc.stdout is not None else ""
    if stdout_path is not None:
        with open(stdout_path, "w", encoding="utf-8") as f:
            f.write(output)
    if proc.returncode != 0:
        raise RuntimeError(f"Command failed with exit code {proc.returncode}: {_shell_join(cmd)}\n{output}")
    return output


def _dump_mcc_asm(cmd: list[str], file_target: str) -> None:
    if not env_enabled_or_path(TILELANG_PRINT_ASM_ENV):
        return

    asm_cmd = _mcc_asm_cmd(cmd, file_target)
    asm_path = asm_cmd[asm_cmd.index("-o") + 1]
    _run_logged_command(asm_cmd)
    print(f"===== TileLang MUSA ASM: {asm_path} =====")
    with open(asm_path, "r", encoding="utf-8") as f:
        print(f.read())
    print(f"===== End TileLang MUSA ASM: {asm_path} =====")


def _dump_musaasm_fatbin(file_target: str, arch: str) -> None:
    if not env_enabled_or_path(TILELANG_DUMP_ASM_ENV):
        return

    asm_path = env_path_or_default(TILELANG_DUMP_ASM_ENV, "tilelang_musaasm.asm")
    os.makedirs(os.path.dirname(asm_path), exist_ok=True)
    cmd = [os.path.join(find_musa_path(), "bin", "musaasm"), f"-mcpu={arch}", "--type=fatbin", file_target, "-M"]
    _run_logged_command(cmd, stdout_path=asm_path)
    print(f"compile_musa dumped editable musaasm to: {asm_path}")


def _replace_musaasm_fatbin(file_target: str, arch: str) -> None:
    if not env_enabled_or_path(TILELANG_REPLACE_ASM_ENV):
        return

    asm_path = env_path_or_default(TILELANG_REPLACE_ASM_ENV, "tilelang_musaasm.asm")
    if not os.path.isfile(asm_path):
        raise RuntimeError(f"{TILELANG_REPLACE_ASM_ENV} requested but asm does not exist: {asm_path}")

    musa_bin = os.path.join(find_musa_path(), "bin")
    bundler = os.path.join(musa_bin, "clang-offload-bundler")
    musaasm = os.path.join(musa_bin, "musaasm")
    targets = _run_logged_command([bundler, "--list", "--type=o", f"--inputs={file_target}"])
    targets = [line.strip() for line in targets.splitlines() if line.strip()]
    device_targets = [target for target in targets if target.startswith("musa-")]
    host_targets = [target for target in targets if target not in device_targets]
    if len(host_targets) != 1 or len(device_targets) != 1:
        raise RuntimeError(f"Unexpected offload targets in {file_target}: {targets}")

    with tempfile.TemporaryDirectory(prefix="tilelang_musaasm_replace_") as workdir:
        host_o = os.path.join(workdir, "host.o")
        old_device_o = os.path.join(workdir, "old_device.o")
        new_device_o = os.path.join(workdir, "new_device.o")
        _run_logged_command(
            [
                bundler,
                "--unbundle",
                "--type=o",
                f"--inputs={file_target}",
                f"--targets={host_targets[0]},{device_targets[0]}",
                f"--outputs={host_o},{old_device_o}",
            ]
        )
        _run_logged_command(
            [musaasm, asm_path, f"-mcpu={arch}", "--binShrink", "--filetype=obj", "-o", new_device_o]
        )
        _run_logged_command(
            [
                bundler,
                "--type=o",
                f"--inputs={host_o},{new_device_o}",
                f"--targets={host_targets[0]},{device_targets[0]}",
                f"--outputs={file_target}",
            ]
        )


def compile_musa(code, target_format="ptx", arch=None, options=None, path_target=None, verbose=False):
    """Compile musa code with MCC from env.

    Parameters
    ----------
    code : str
        The musa code.

    target_format : str
        The target format of mcc compiler.

    arch : str
        The musa architecture.

    options : str or list of str
        The additional options.

    path_target : str, optional
        Output file.

    Return
    ------
    mubin : bytearray
        The bytearray of the mubin
    """
    if arch is None:
        # If None, then it will use `tvm.target.Target.current().arch`.
        # Target arch could be a str like "mp_xx"
        compute_version = get_musa_compute_version(Target.current(allow_none=True))
        target_arch = get_musa_arch(compute_version)
        arch = f"mp_{target_arch}"

    temp = utils.tempdir(keep_for_debug=not env.should_cleanup_temp_files())
    file_name = "tvm_kernels"
    if target_format not in ["mubin", "asm"]:
        raise ValueError("target_format must be in mubin, asm")
    pass_context = tvm.get_global_func("transform.GetCurrentPassContext")()
    kernels_output_dir = pass_context.config.get("musa.kernels_output_dir", None)
    temp_code, temp_target = _resolve_artifact_paths(temp, file_name, target_format, kernels_output_dir=kernels_output_dir)

    with open(temp_code, "w") as out_file:
        out_file.write(code)

    file_target = path_target if path_target else temp_target
    cmd = [get_mcc_compiler(), "--cuda-device-only"]
    if target_format == "asm":
        cmd += ["-S"]
    # TODO(xtyi): check lineinfo
    cmd += [f"--cuda-gpu-arch={arch}"]

    if options:
        if isinstance(options, str):
            cmd += [options]
        elif isinstance(options, list):
            cmd += options
        else:
            raise ValueError("options must be str or list of str")

    cmd += ["-o", file_target]
    cmd += [temp_code]
    _dump_kernel_source(temp_code, code, cmd)

    # NOTE: ccbin option can be used to tell mcc where to find the c++ compiler
    # just in case it is not in the path. On Windows it is not in the path by default.
    # However, we cannot use TVM_CXX_COMPILER_PATH because the runtime env.
    # Because it is hard to do runtime compiler detection, we require mcc is configured
    # correctly by default.
    # if cxx_compiler_path != "":
    #    cmd += ["-ccbin", cxx_compiler_path]

    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)

    (out, _) = proc.communicate()

    if env.is_print_device_compile_command_enabled():
        print(f"compile_musa command: {' '.join(cmd)}")

    if verbose:
        print(py_str(out))

    if proc.returncode != 0:
        msg = f"{code}\nCompilation error:\n{py_str(out)}\nCommand: {' '.join(cmd)}\n"
        raise RuntimeError(msg)

    if target_format != "asm":
        _dump_mcc_asm(cmd, file_target)
        _dump_musaasm_fatbin(file_target, arch)
        _replace_musaasm_fatbin(file_target, arch)

    with open(file_target, "rb") as f:
        data = bytearray(f.read())
        if not data:
            raise RuntimeError("Compilation error: empty result is generated")
        return data


def default_compile_options(compile_flags: list[str] | None = None) -> list[str]:
    """
    Build a set of default MCC compile options for TileLang generated sources.

    Includes C++ standard and common include paths (TileLang templates, MUTLASS,
    MUSA include). Merges user-provided compile flags if given.

    Parameters
    ----------
    compile_flags : Optional[List[str]]
        Additional flags to include. Items are split on whitespace.

    Returns
    -------
    List[str]
        A list of flags suitable for MCC's command line.
    """
    options: list[str] = ["-std=c++17"]
    try:
        if TILELANG_TEMPLATE_PATH:
            options.append(f"-I{TILELANG_TEMPLATE_PATH}")
    except Exception:
        pass

    try:
        if MUTLASS_INCLUDE_DIR:
            options.append(f"-I{MUTLASS_INCLUDE_DIR}")
    except Exception:
        pass

    try:
        if MUSA_HOME:
            options.append(f"-I{os.path.join(MUSA_HOME, 'include')}")
    except Exception:
        pass

    # Preserve user flags exactly, including repeated tokens required by MCC
    # (e.g., multiple "-gencode" pairs or repeated "-Xcompiler" entries).
    if compile_flags:
        import shlex

        for flag in compile_flags:
            # Split each string like a shell would, preserving quoted args
            tokens = shlex.split(flag) if isinstance(flag, str) else [str(flag)]
            options.extend(tokens)
    return options


def find_musa_path():
    """Utility function to find musa path

    Returns
    -------
    path : str
        Path to musa root.
    """
    if MUSA_HOME:
        return MUSA_HOME
    raise RuntimeError(
        "Failed to automatically detect MUSA installation. Please set the MUSA_HOME environment variable manually (e.g., export MUSA_HOME=/usr/local/musa)."
    )


def get_musa_version(musa_path=None):
    """Utility function to get musa version

    Parameters
    ----------
    musa_path : Optional[str]

        Path to musa root.  If None is passed, will use
        `find_musa_path()` as default.

    Returns
    -------
    version : float
        The musa version

    """
    if musa_path is None:
        musa_path = find_musa_path()

    # todo: read from version.json

    cmd = [os.path.join(musa_path, "bin", "mcc"), "--version"]
    proc = subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.STDOUT)
    (out, _) = proc.communicate()
    out = py_str(out)
    if proc.returncode == 0:
        release_line = [l for l in out.split("\n") if "mcc version" in l][0]
        release_fields = [s.strip() for s in release_line.split(" ")]
        version_str = release_fields[-1]
        return tuple(int(field) for field in version_str.split("."))
    raise RuntimeError("Cannot read musa version file")


@tvm_ffi.register_global_func("tvm.contrib.mcc.get_compute_version", override=True)
def get_musa_compute_version(target=None):
    """Utility function to get compute capability of compilation target.

    Looks for the target arch in three different places, first in the target input, then the
    Target.current() scope, and finally the GPU device (if it exists).

    Parameters
    ----------
    target : tvm.target.Target, optional
        The compilation target

    Returns
    -------
    compute_version : str
        compute capability of a GPU (e.g. "8.6" or "9.0")
    """
    # 1. input target object
    # 2. Target.current()
    target = target or Target.current()
    if target and target.arch:
        arch = str(target.arch).split("_")[-1]
        if "." in arch:
            return arch
        if len(arch) < 2:
            raise ValueError(f"Invalid MUSA arch format: {target.arch}")
        if len(arch) == 2:
            return arch[0] + "." + arch[1]
        return arch[:-1] + "." + arch[-1]

    # 3. GPU compute version
    if hasattr(tvm, "musa"):
        dev = tvm.musa(0)
        if dev.exist:
            return dev.compute_version

    # 4. PyTorch MUSA runtime fallback (useful when TVM MUSA device API
    # is not available in the current build).
    try:
        import torch

        if hasattr(torch, "musa") and torch.musa.is_available():
            props = torch.musa.get_device_properties(0)
            major = getattr(props, "major", None)
            minor = getattr(props, "minor", None)
            if major is not None and minor is not None:
                return f"{int(major)}.{int(minor)}"
    except Exception:
        pass

    raise ValueError("No MUSA architecture was specified or GPU detected.Try specifying it by adding '-arch=mp_xx' to your target.")


def parse_musa_compute_version(compute_version) -> tuple[int, int]:
    """Parse compute capability string to divide major and minor version

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.0")

    Returns
    -------
    major : int
        major version number
    minor : int
        minor version number
    """
    split_ver = compute_version.split(".")
    try:
        major = int(split_ver[0])
        minor = int(split_ver[1])
        return major, minor
    except (IndexError, ValueError) as err:
        # pylint: disable=raise-missing-from
        raise RuntimeError("Compute version parsing error") from err


def get_musa_arch(compute_version) -> str:
    major, minor = parse_musa_compute_version(compute_version)
    target_arch = str(major * 10 + minor)
    return target_arch


def have_fp16(compute_version):
    """Either fp16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version: str
        compute capability of a GPU (e.g. "6.0")
    """
    major, _ = parse_musa_compute_version(compute_version)
    return major >= 2


def have_int8(compute_version):
    """Either int8 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "6.1")
    """
    major, _ = parse_musa_compute_version(compute_version)
    return major >= 2


@tvm_ffi.register_global_func("tvm.contrib.mcc.supports_bf16", override=True)
def have_bf16(compute_version):
    """Either bf16 support is provided in the compute capability or not

    Parameters
    ----------
    compute_version : str
        compute capability of a GPU (e.g. "8.0")
    """
    major, _ = parse_musa_compute_version(compute_version)
    return major >= 2


@tvm_ffi.register_global_func("tvm.contrib.mcc.supports_fp8", override=True)
def have_fp8(compute_version):
    """Whether fp8 support is provided in the specified compute capability or not

    Parameters
    ----------
    compute_version : str
        GPU capability
    """
    major, _ = parse_musa_compute_version(compute_version)
    # fp8 is supported in S5000 or later architectures.
    return major >= 3


@tvm_ffi.register_global_func("tvm.contrib.mcc.supports_tma", override=True)
def have_tma(target):
    """Whether TMA support is provided in the specified compute capability or not

    Parameters
    ----------
    target : tvm.target.Target
        The compilation target
    """
    if target.kind.name != "musa":
        return False
    compute_version = get_musa_compute_version(target)
    major, _ = parse_musa_compute_version(compute_version)
    # TMA is supported in S5000 or later architectures.
    return major >= 3


def is_qy2(target):
    if target.kind.name != "musa":
        return False
    compute_version = get_musa_compute_version(target)
    major, minor = parse_musa_compute_version(compute_version)
    return major == 2 and minor == 2


def is_ph1(target):
    if target.kind.name != "musa":
        return False
    compute_version = get_musa_compute_version(target)
    major, minor = parse_musa_compute_version(compute_version)
    return major == 3 and minor == 1


def get_mcc_compiler() -> str:
    """Get the path to the mcc compiler"""
    return os.path.join(find_musa_path(), "bin", "mcc")
