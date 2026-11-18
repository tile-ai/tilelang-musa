import tilelang as tl
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tirx as tir


def _vectorize(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    return tl.transform.VectorizeLoop()(mod)["main"]


def _get_single_store(func):
    stores = []

    def visitor(obj):
        if isinstance(obj, tir.BufferStore):
            stores.append(obj)

    tir.stmt_functor.post_order_visit(func.body, visitor)
    assert len(stores) == 1
    return stores[0]


def test_contiguous_store_uses_thread_bound_to_form_ramp():
    @T.prim_func
    def main(B: T.Tensor((16384,), "float32")):
        for worker in T.thread_binding(640, thread="threadIdx.x"):
            for lane in T.vectorized(8):
                B[
                    T.if_then_else(
                        worker < 640,
                        worker * 8 + lane,
                        worker * 16 + lane * 2,
                    )
                ] = T.float32(1)

    store = _get_single_store(_vectorize(main))
    ramp = store.indices[0]
    assert isinstance(ramp, tir.Ramp)
    assert ramp.lanes == 8
    assert tir.analysis.expr_deep_equal(ramp.stride, 1)


def test_unbounded_store_index_is_not_forced_to_ramp():
    @T.prim_func
    def main(B: T.Tensor((16384,), "float32"), worker: T.int32):
        for lane in T.vectorized(8):
            B[
                T.if_then_else(
                    worker < 640,
                    worker * 8 + lane,
                    worker * 16 + lane * 2,
                )
            ] = T.float32(1)

    store = _get_single_store(_vectorize(main))
    assert not isinstance(store.indices[0], tir.Ramp)


def test_non_contiguous_store_is_not_forced_to_unit_stride_ramp():
    @T.prim_func
    def main(B: T.Tensor((16384,), "float32")):
        for worker in T.thread_binding(640, thread="threadIdx.x"):
            for lane in T.vectorized(8):
                B[worker * 16 + lane * 2] = T.float32(1)

    store = _get_single_store(_vectorize(main))
    ramp = store.indices[0]
    assert isinstance(ramp, tir.Ramp)
    assert not tir.analysis.expr_deep_equal(ramp.stride, 1)


if __name__ == "__main__":
    tl.testing.main()
