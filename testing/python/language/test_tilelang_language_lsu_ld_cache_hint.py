import pytest

import tilelang
import tilelang.language as T
import tilelang.testing


@pytest.mark.parametrize("arch", ["mp_31"])
@tilelang.testing.requires_musa
def test_lsu_ld_cache_hint_codegen(arch):
    n = 128

    @T.prim_func
    def main(
        x: T.Tensor((n,), T.float32),
        y: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n, threads=128) as pid:
            y[pid] = T.lsu_ld_cache_hint(
                x[pid],
                inner_persistence="cache_persist",
                outer_persistence="cache_once",
                chrnt="l1",
                l2="new_alloc",
            )

    kernel = tilelang.compile(
        main,
        out_idx=[1],
        target={"kind": "musa", "arch": arch},
    )
    source = kernel.get_kernel_source()
    assert "tl::lsu_ld_cache_hint<3, 1, 0, 0, false>" in source


@tilelang.testing.requires_musa
def test_lsu_ld_cache_hint_default_and_volatile_codegen():
    n = 128

    @T.prim_func
    def main(
        x: T.Tensor((n,), T.float32),
        y: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n, threads=128) as pid:
            y[pid] = T.lsu_ld_cache_hint(x[pid]) + T.lsu_ld_cache_hint(
                x[pid],
                inner_persistence="last_use",
                outer_persistence="cache_persist",
                chrnt="l2_l3",
                l2="bypass",
                is_volatile=True,
            )

    kernel = tilelang.compile(
        main,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
    )
    source = kernel.get_kernel_source()
    assert "tl::lsu_ld_cache_hint<4, 2, 0, 0, false>" in source
    assert "tl::lsu_ld_cache_hint<5, 3, 1, 1, true>" in source


@pytest.mark.parametrize("arch", ["mp_31"])
@tilelang.testing.requires_musa
def test_copy_lsu_ld_cache_hint_codegen(arch):
    n = 128

    @T.prim_func
    def main(
        x: T.Tensor((n,), T.float32),
        y: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n, threads=128) as pid:
            T.copy(
                x[pid],
                y[pid],
                inner_cache_policy="cache_persist",
                outer_cache_policy="cache_once",
                chrnt="l1",
                l2="new_alloc",
            )

    kernel = tilelang.compile(
        main,
        out_idx=[1],
        target={"kind": "musa", "arch": arch},
    )
    source = kernel.get_kernel_source()
    assert "tl::lsu_ld_cache_hint<3, 1, 0, 0, false>" in source


@tilelang.testing.requires_musa
def test_copy_lsu_ld_cache_hint_default_and_volatile_codegen():
    n = 128

    @T.prim_func
    def main(
        x: T.Tensor((n,), T.float32),
        y: T.Tensor((n,), T.float32),
        z: T.Tensor((n,), T.float32),
    ):
        with T.Kernel(n, threads=128) as pid:
            T.copy(x[pid], y[pid], is_volatile=False)
            T.copy(
                x[pid],
                z[pid],
                inner_cache_policy="last_use",
                outer_cache_policy="cache_persist",
                chrnt="l2_l3",
                l2="bypass",
                is_volatile=True,
            )

    kernel = tilelang.compile(
        main,
        out_idx=[1, 2],
        target={"kind": "musa", "arch": "mp_31"},
    )
    source = kernel.get_kernel_source()
    assert "tl::lsu_ld_cache_hint<4, 2, 0, 0, false>" in source
    assert "tl::lsu_ld_cache_hint<5, 3, 1, 1, true>" in source


if __name__ == "__main__":
    tilelang.testing.main()
