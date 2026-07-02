from tilelang import tvm as tvm
import tilelang.language as T
import tilelang.testing
from tilelang.musa import transform as musa_transform

musa_target = tvm.target.Target("musa -arch=mp_31", host="llvm")


@tilelang.testing.requires_musa
def test_tma_descriptor_init_after_alloc_global():
    @T.prim_func
    def before():
        T.func_attr({"tir.is_entry_func": True, "tl.has_tma": T.bool(True)})
        Output_partial = T.allocate([32], "float16", "global")
        with T.launch_thread("threadIdx.x", 1):
            T.evaluate(
                T.create_tma_descriptor(
                    6,
                    4,
                    Output_partial,
                    8,
                    2,
                    2,
                    1,
                    2,
                    16,
                    32,
                    64,
                    8,
                    1,
                    2,
                    1,
                    1,
                    1,
                    1,
                    1,
                    0,
                    0,
                    2,
                    0,
                )
            )

    mod = tvm.IRModule.from_expr(before.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(musa_target)(mod)
    mod = musa_transform.LowerPHIntrin()(mod)
    func = mod["main"]

    assert not tvm.tir.analysis.undefined_vars(func.body, func.params)
    body_text = func.script()
    assert body_text.index('T.allocate([32], "float16", "global")') < body_text.index('T.call_packed("__tvm_tensormap_create_tiled"')


if __name__ == "__main__":
    tilelang.testing.main()
