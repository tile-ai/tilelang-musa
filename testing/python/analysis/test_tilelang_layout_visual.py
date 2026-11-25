"""Tests for fragment layout visualization."""

import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm


@tilelang.jit
def _layout_visual_kernel():
    @T.prim_func
    def main():
        with T.Kernel(1, threads=32):
            fragment = T.alloc_fragment((8, 4), T.float32)
            layout = T.Fragment((8, 4), lambda i, j: (i * 4 + j, 0))
            T.annotate_layout({fragment: layout})
            for i, j in T.Parallel(8, 4):
                fragment[i, j] = T.float32(0)

    return main


def test_layout_visual_visits_sblock(capsys):
    """LayoutVisual must dispatch to and recurse through TIRX SBlocks."""
    func = _layout_visual_kernel.get_tir()
    mod = tvm.IRModule.from_expr(func)

    tilelang.analysis.LayoutVisual(formats="txt")(mod)

    out = capsys.readouterr().out
    assert out.count("fragment inferred layout:") == 1
    assert "Shape: (8, 4) -> (1,)" in out
    assert "Thread:" in out
    assert "Index:" in out


if __name__ == "__main__":
    tilelang.testing.main()
