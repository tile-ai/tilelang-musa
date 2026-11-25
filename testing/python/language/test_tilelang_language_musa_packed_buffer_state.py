"""Per-function packed buffer state must not affect later functions."""

import re

import pytest
import tilelang
import tvm_ffi
from tvm import tirx


@pytest.mark.parametrize("plain_dtype", ["float16", "float32"])
@pytest.mark.parametrize("operation", ["load", "store"])
def test_packed_buffer_state_is_local_to_function(plain_dtype, operation):
    tvm = tilelang.tvm
    data = tirx.Var("scratch", tvm.ir.PointerType(tvm.ir.PrimType("uint8"), "local"))

    def make_func(name, dtype):
        buffer = tirx.decl_buffer((8,), dtype=dtype, data=data, scope="local", name="scratch")
        access = (
            tirx.BufferStore(buffer, tirx.const(0, dtype), [0])
            if operation == "store"
            else tirx.Evaluate(tirx.call_extern("int32", "consume", buffer[0]))
        )
        return (
            tirx.PrimFunc([], tirx.SeqStmt([tirx.AllocBuffer(buffer), access]))
            .with_attr("global_symbol", name)
            .with_attr("calling_conv", 2)
        )

    module = tvm.IRModule({"packed": make_func("packed", "float4_e2m1fn"), "plain": make_func("plain", plain_dtype)})
    build = tvm_ffi.get_global_func("target.build.tilelang_musa_without_compile")
    code = build(module, tvm.target.Target({"kind": "musa", "arch": "mp_31"})).inspect_source()
    packed = re.search(r'void packed\(\) \{(.*?)\n\}', code, re.S).group(1)
    plain = re.search(r'void plain\(\) \{(.*?)\n\}', code, re.S).group(1)
    assert code.index("void packed() {") < code.index("void plain() {")
    assert "scratch_packed" in packed
    assert "tl_fp4_packed_" in packed
    assert "scratch[8]" in plain
    assert "scratch_packed" not in plain
    assert "tl_fp4_packed_" not in plain
