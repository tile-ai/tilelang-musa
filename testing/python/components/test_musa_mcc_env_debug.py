from __future__ import annotations

import ast
import importlib.util
import os
import shlex
import subprocess
import sys
from pathlib import Path
from types import SimpleNamespace


def _load_env_module():
    spec = importlib.util.spec_from_file_location("tilelang_env_for_test", "tilelang/env.py")
    module = importlib.util.module_from_spec(spec)
    sys.modules[spec.name] = module
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def _load_mcc_helpers():
    env = _load_env_module()
    source = Path("tilelang/contrib/mcc.py").read_text(encoding="utf-8")
    tree = ast.parse(source)
    wanted = {
        "_resolve_dump_path",
        "_shell_join",
        "_mcc_target_ext",
        "_next_numbered_dump_path",
        "_mcc_asm_cmd",
        "_retarget_mcc_dump_cmd",
        "_dump_kernel_source",
        "_run_logged_command",
        "_dump_mcc_asm",
    }
    module = ast.Module(
        body=[node for node in tree.body if isinstance(node, ast.FunctionDef) and node.name in wanted],
        type_ignores=[],
    )
    namespace = {
        "ENV_FALSE_VALUES": env.ENV_FALSE_VALUES,
        "ENV_TRUE_VALUES": env.ENV_TRUE_VALUES,
        "TILELANG_DUMP_KERNEL_SOURCE_ENV": env.TILELANG_DUMP_KERNEL_SOURCE_ENV,
        "TILELANG_PRINT_ASM_ENV": env.TILELANG_PRINT_ASM_ENV,
        "env_enabled_or_path": env.env_enabled_or_path,
        "os": os,
        "py_str": lambda data: data.decode("utf-8", errors="replace"),
        "shlex": shlex,
        "subprocess": subprocess,
    }
    exec(compile(module, "tilelang/contrib/mcc.py", "exec"), namespace)
    return SimpleNamespace(env=env, **namespace)


def test_dump_kernel_source_writes_numbered_source_and_commands(tmp_path, monkeypatch):
    mcc = _load_mcc_helpers()
    monkeypatch.setenv(mcc.env.TILELANG_DUMP_KERNEL_SOURCE_ENV, str(tmp_path / "kernel.mu"))

    cmd = [
        "mcc",
        "--cuda-device-only",
        "--cuda-gpu-arch=mp_31",
        "-o",
        str(tmp_path / "original.mubin"),
        str(tmp_path / "original.mu"),
    ]

    mcc._dump_kernel_source(str(tmp_path / "original.mu"), "__global__ void k0(){}\n", cmd)
    mcc._dump_kernel_source(str(tmp_path / "original.mu"), "__global__ void k1(){}\n", cmd)

    assert (tmp_path / "kernel_0000.mu").read_text(encoding="utf-8") == "__global__ void k0(){}\n"
    assert (tmp_path / "kernel_0001.mu").read_text(encoding="utf-8") == "__global__ void k1(){}\n"

    command_text = (tmp_path / "kernel_0000.mu.cmd").read_text(encoding="utf-8")
    assert "# compile\n" in command_text
    assert f"-o {tmp_path / 'kernel_0000.mubin'}" in command_text
    assert str(tmp_path / "kernel_0000.mu") in command_text
    assert "\n\n# asm\n" in command_text
    assert f"-S -o {tmp_path / 'kernel_0000.s'}" in command_text


def test_dump_mcc_asm_runs_asm_command_and_prints_file(tmp_path, monkeypatch, capsys):
    mcc = _load_mcc_helpers()
    monkeypatch.setenv(mcc.env.TILELANG_PRINT_ASM_ENV, "1")
    target_path = tmp_path / "kernel.mubin"

    def fake_run(cmd, stdout, stderr, check):
        assert stdout == subprocess.PIPE
        assert stderr == subprocess.STDOUT
        assert check is False
        assert "-S" in cmd
        asm_path = cmd[cmd.index("-o") + 1]
        assert asm_path == str(tmp_path / "kernel.s")
        assert cmd[-1] == str(tmp_path / "kernel.mu")
        (tmp_path / "kernel.s").write_text("// generated asm\n", encoding="utf-8")
        return subprocess.CompletedProcess(cmd, 0, stdout=b"")

    monkeypatch.setattr(mcc.subprocess, "run", fake_run)

    mcc._dump_mcc_asm(
        [
            "mcc",
            "--cuda-device-only",
            "--cuda-gpu-arch=mp_31",
            "-o",
            str(target_path),
            str(tmp_path / "kernel.mu"),
        ],
        str(target_path),
    )

    output = capsys.readouterr().out
    assert "compile_musa command:" in output
    assert "===== TileLang MUSA ASM:" in output
    assert "// generated asm" in output
