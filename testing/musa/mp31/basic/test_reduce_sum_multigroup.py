"""Named barriers must synchronize every concurrent AllReduce group."""

import re

import pytest
import torch

import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


@tilelang.jit(target="musa")
def reduce_sum_multigroup(src, dst, threads=256, offset=0, participants=256, batch=1):
    src: T.Tensor((32, 256), T.float32)
    dst: T.Tensor((32,), T.float32)
    with T.Kernel(1, threads=threads):
        tx = T.get_thread_binding()
        if tx >= offset and tx < offset + participants:
            src_local = T.alloc_fragment((32, 256), T.float32)
            dst_local = T.alloc_fragment((32,), T.float32)
            T.copy(src, src_local)
            T.reduce_sum(src_local, dst_local, dim=1, batch=batch)
            T.copy(dst_local, dst)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
@pytest.mark.parametrize(
    "threads,offset,participants",
    [(256, 0, 256), (96, 32, 64), (256, 64, 128)],
)
@pytest.mark.parametrize("batch", [1, 4])
def test_reduce_sum_multigroup(threads, offset, participants, batch):
    kernel = reduce_sum_multigroup.compile(threads=threads, offset=offset, participants=participants, batch=batch)
    source = kernel.get_kernel_source()
    # A group still reduces one row across 64 threads, even when several
    # independent groups share the same pair of CTA-wide named barriers.
    match = re.search(
        rf"tl::AllReduce<tl::SumOp,\s*64,\s*1,\s*{offset},\s*tl::NamedBarrier<(\d+)>",
        source,
    )
    assert match is not None, source
    base = int(match.group(1))
    for barrier in (base, base + 1):
        init = re.search(
            rf"__musa_async_init_arrival\(\s*{barrier}\s*,\s*([^,]+),\s*0\s*\)",
            source,
        )
        assert init is not None, source
        count = re.sub(r"\s+", "", init.group(1))
        assert count in {
            str(participants // 32),
            f"(({participants}+31)/32)",
        }, source
    if batch > 1:
        assert re.search(rf",\s*{batch},\s*{participants}>::run_batch", source), source
    device_func = next(iter(kernel.artifact.device_mod.functions.values()))
    assert int(device_func.attrs["dyn_shared_memory_buf"]) >= participants * batch * 4

    generator = torch.Generator().manual_seed(1)
    src_cpu = torch.randn((32, 256), generator=generator)
    expected = src_cpu.double().sum(dim=1).float()
    src = src_cpu.to("musa")
    dst = torch.empty((32,), dtype=torch.float32, device="musa")
    for _ in range(100):
        dst.fill_(float("nan"))
        kernel(src, dst)
        torch.testing.assert_close(dst.cpu(), expected, rtol=1e-4, atol=1e-4)
