import tilelang
import tilelang.language as T
import tilelang.testing
from tilelang import tvm
from tvm import tir

tilelang.disable_cache()


def run_lower_shared_barrier(func):
    mod = tvm.IRModule.from_expr(func.with_attr("global_symbol", "main"))
    mod = tvm.tir.transform.BindTarget(tvm.target.Target("musa", host="llvm"))(mod)
    mod = tilelang.transform.LowerSharedBarrier()(mod)
    return mod["main"]


def make_before():
    @T.prim_func
    def before():
        with T.Kernel(1, threads=128):
            mbars = T.alloc_buffer((2,), "uint64", scope="shared.barrier")
            T.block_attr({"barrier_init": {mbars.data: [T.int32(128), T.int32(256)]}})
            T.evaluate(tir.Call("handle", "tir.ptx_arrive_barrier", [mbars[0]]))
            T.evaluate(
                tir.Call(
                    "handle",
                    "tir.ptx_arrive_barrier_expect_tx",
                    [mbars[1], T.int32(64)],
                )
            )

    return before


def make_before_with_dynamic_barrier_index():
    @T.prim_func
    def before_dynamic():
        with T.Kernel(1, threads=128):
            mbars = T.alloc_buffer((4,), "uint64", scope="shared.barrier")
            T.block_attr({"barrier_init": {mbars.data: [T.int32(128), T.int32(128), T.int32(128), T.int32(128)]}})
            for ko in range(2):
                idx = ko % 2
                T.evaluate(
                    tir.Call(
                        "handle",
                        "tir.ptx_arrive_barrier",
                        [mbars[idx]],
                    )
                )
                T.evaluate(
                    tir.Call(
                        "handle",
                        "tir.ptx_arrive_barrier_expect_tx",
                        [mbars[idx + 2], T.int32(64)],
                    )
                )

    return before_dynamic


def has_shared_barrier_alloc(stmt):
    found = False

    def visit(node):
        nonlocal found
        if isinstance(node, tir.Block):
            for buf in node.alloc_buffers:
                if getattr(buf.data.type_annotation, "storage_scope", None) == "shared.barrier":
                    found = True

    tir.stmt_functor.post_order_visit(stmt, visit)
    return found


def collect_lowered_stats(stmt):
    barrier_calls = {
        "tir.ptx_init_barrier_thread_count",
        "tir.ptx_arrive_barrier",
        "tir.ptx_arrive_barrier_expect_tx",
    }

    placeholder_calls = 0
    has_buffer_load_arg = False
    has_add_arg = False

    def visit(node):
        nonlocal placeholder_calls, has_buffer_load_arg, has_add_arg
        if not isinstance(node, tir.Call):
            return
        op = node.op
        if isinstance(op, tvm.ir.Op) and op.name == "tl.barrier_id_placeholder":
            placeholder_calls += 1
            return
        if isinstance(op, tvm.ir.Op) and op.name in barrier_calls and node.args:
            arg0 = node.args[0]
            has_buffer_load_arg = has_buffer_load_arg or isinstance(arg0, tir.BufferLoad)
            has_add_arg = has_add_arg or isinstance(arg0, tir.Add)

    tir.stmt_functor.post_order_visit(stmt, visit)
    return placeholder_calls, has_buffer_load_arg, has_add_arg


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_lower_shared_barrier_to_named_barrier():
    lowered = run_lower_shared_barrier(make_before())
    assert not has_shared_barrier_alloc(lowered.body)

    placeholder_calls, has_buffer_load_arg, _ = collect_lowered_stats(lowered.body)
    assert placeholder_calls == 2
    assert not has_buffer_load_arg


@tilelang.testing.requires_musa_compute_version_ge(3, 1)
def test_lower_shared_barrier_dynamic_index_uses_base_plus_idx():
    lowered = run_lower_shared_barrier(make_before_with_dynamic_barrier_index())
    assert not has_shared_barrier_alloc(lowered.body)

    placeholder_calls, has_buffer_load_arg, has_add_arg = collect_lowered_stats(lowered.body)
    assert placeholder_calls == 4
    assert not has_buffer_load_arg
    assert has_add_arg


def main():
    before = make_before_with_dynamic_barrier_index()
    print("=== Before LowerSharedBarrier ===")
    print(before.script())

    lowered = run_lower_shared_barrier(before)
    print("=== After LowerSharedBarrier ===")
    print(lowered.script())

    assert not has_shared_barrier_alloc(lowered.body)
    placeholder_calls, has_buffer_load_arg, has_add_arg = collect_lowered_stats(lowered.body)
    assert placeholder_calls == 4
    assert not has_buffer_load_arg
    assert has_add_arg
    print("pass!")


if __name__ == "__main__":
    main()
