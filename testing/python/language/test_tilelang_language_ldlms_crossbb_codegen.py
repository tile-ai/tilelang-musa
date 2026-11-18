import tilelang
import tilelang.language as T


def test_async_copy_group_markers_codegen():
    @T.prim_func
    def main(out: T.Tensor((1,), T.int32)):
        with T.Kernel(1, threads=32):
            T.async_copy_commit_group(7)
            T.async_copy_wait_group(7)
            T.async_copy_commit_group(0)
            T.async_copy_wait_group(0)
            T.async_copy_commit_group(1)
            T.async_copy_wait_group(1)
            T.sched_boundary()
            out[0] = 1

    artifact = tilelang.lower(main, target="musa", enable_device_compile=False)
    source = artifact.kernel_source
    print(source)
    expected_markers = (
        "tl::async_copy_commit_group<7>();",
        "tl::async_copy_wait_group<7>();",
        "tl::async_copy_commit_group<0>();",
        "tl::async_copy_wait_group<0>();",
        "tl::async_copy_commit_group<1>();",
        "tl::async_copy_wait_group<1>();",
        "tl::sched_boundary();",
    )
    for marker in expected_markers:
        assert marker in source
