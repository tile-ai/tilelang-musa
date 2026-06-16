from __future__ import annotations

import shlex
import subprocess

import tilelang.contrib.mcc as mcc


def test_dump_kernel_source_writes_numbered_source_and_commands(tmp_path, monkeypatch):
    monkeypatch.setenv(mcc.TILELANG_DUMP_KERNEL_SOURCE_ENV, str(tmp_path / "kernel.mu"))

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

    expected_compile_cmd = [
        "mcc",
        "--cuda-device-only",
        "--cuda-gpu-arch=mp_31",
        "-o",
        str(tmp_path / "kernel_0000.mubin"),
        str(tmp_path / "kernel_0000.mu"),
    ]
    expected_asm_cmd = [
        "mcc",
        "--cuda-device-only",
        "--cuda-gpu-arch=mp_31",
        "-S",
        "-o",
        str(tmp_path / "kernel_0000.s"),
        str(tmp_path / "kernel_0000.mu"),
    ]
    assert (tmp_path / "kernel_0000.mu.cmd").read_text(encoding="utf-8") == (
        f"# compile\n{shlex.join(expected_compile_cmd)}\n\n"
        f"# asm\n{shlex.join(expected_asm_cmd)}\n"
    )


def test_dump_mcc_asm_runs_asm_command_and_prints_file(tmp_path, monkeypatch, capsys):
    monkeypatch.setenv(mcc.TILELANG_PRINT_ASM_ENV, "1")
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
