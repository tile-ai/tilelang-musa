"""Tests for MP31 fragment-A SQMMA GEMM legalization and 188 KiB SMEM guard."""

import pytest
import torch
import torch.nn.functional as F
import tilelang
import tilelang.language as T
import tilelang.testing

tilelang.disable_cache()


def _flash_attention_prefill(
    batch,
    heads,
    seq_len,
    dim,
    block_M=64,
    block_N=64,
    threads=128,
    policy=T.GemmWarpPolicy.Square,
    num_stages=1,
):
    scale = (1.0 / dim) ** 0.5 * 1.44269504  # log2(e) / sqrt(d)

    @T.prim_func
    def main(
        Q: T.Tensor((batch, seq_len, heads, dim), "float16"),
        K: T.Tensor((batch, seq_len, heads, dim), "float16"),
        V: T.Tensor((batch, seq_len, heads, dim), "float16"),
        Output: T.Tensor((batch, seq_len, heads, dim), "float16"),
    ):
        with T.Kernel(T.ceildiv(seq_len, block_M), heads, batch, threads=threads) as (bx, by, bz):
            Q_shared = T.alloc_shared((block_M, dim), "float16")
            K_shared = T.alloc_shared((block_N, dim), "float16")
            V_shared = T.alloc_shared((block_N, dim), "float16")
            acc_s = T.alloc_fragment((block_M, block_N), "float32")
            acc_s_cast = T.alloc_fragment((block_M, block_N), "float16")
            acc_o = T.alloc_fragment((block_M, dim), "float32")
            scores_max = T.alloc_fragment((block_M,), "float32")
            scores_max_prev = T.alloc_fragment((block_M,), "float32")
            scores_scale = T.alloc_fragment((block_M,), "float32")
            scores_sum = T.alloc_fragment((block_M,), "float32")
            logsum = T.alloc_fragment((block_M,), "float32")

            T.copy(Q[bz, bx * block_M : (bx + 1) * block_M, by, :], Q_shared)
            T.fill(acc_o, 0.0)
            T.fill(logsum, 0.0)
            T.fill(scores_max, -T.infinity("float32"))

            for k in T.Pipelined(T.ceildiv(seq_len, block_N), num_stages=num_stages):
                T.copy(K[bz, k * block_N : (k + 1) * block_N, by, :], K_shared)
                T.clear(acc_s)
                T.gemm(Q_shared, K_shared, acc_s, transpose_B=True, policy=policy)

                T.copy(scores_max, scores_max_prev)
                T.fill(scores_max, -T.infinity("float32"))
                T.reduce_max(acc_s, scores_max, dim=1, clear=False)
                for i in T.Parallel(block_M):
                    scores_max[i] = T.max(scores_max[i], scores_max_prev[i])
                    scores_scale[i] = T.exp2(scores_max_prev[i] * scale - scores_max[i] * scale)
                for i, j in T.Parallel(block_M, dim):
                    acc_o[i, j] *= scores_scale[i]
                for i, j in T.Parallel(block_M, block_N):
                    acc_s[i, j] = T.exp2(acc_s[i, j] * scale - scores_max[i] * scale)
                T.reduce_sum(acc_s, scores_sum, dim=1)
                for i in T.Parallel(block_M):
                    logsum[i] = logsum[i] * scores_scale[i] + scores_sum[i]
                T.copy(acc_s, acc_s_cast)

                T.copy(V[bz, k * block_N : (k + 1) * block_N, by, :], V_shared)
                # Fragment-A (acc_s_cast) x Shared-B (V_shared) GEMM legalized on MP31
                T.gemm(acc_s_cast, V_shared, acc_o, policy=policy)

            for i, j in T.Parallel(block_M, dim):
                acc_o[i, j] /= logsum[i]
            T.copy(acc_o, Output[bz, bx * block_M : (bx + 1) * block_M, by, :])

    return main


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_fragment_a_gemm_flash_attention_e2e():
    B, H, S, D = 1, 2, 256, 64
    prog = _flash_attention_prefill(B, H, S, D, block_M=64, block_N=64, threads=128)
    kernel = tilelang.compile(
        prog,
        out_idx=[3],
        target={"kind": "musa", "arch": "mp_31"},
        pass_configs={"tl.disable_warp_specialized": True},
    )

    q = torch.randn((B, S, H, D), dtype=torch.float16, device="musa") * 0.5
    k = torch.randn((B, S, H, D), dtype=torch.float16, device="musa") * 0.5
    v = torch.randn((B, S, H, D), dtype=torch.float16, device="musa") * 0.5
    out = kernel(q, k, v)
    torch.musa.synchronize()

    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
    ).transpose(1, 2).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_fragment_a_gemm_flash_attention_ws_pipeline_and_fast_math():
    B, H, S, D = 1, 2, 256, 64
    prog = _flash_attention_prefill(B, H, S, D, block_M=64, block_N=64, threads=128)
    kernel = tilelang.compile(
        prog,
        out_idx=[3],
        target={"kind": "musa", "arch": "mp_31"},
        pass_configs={"tl.enable_fast_math": True},
    )
    src = kernel.get_kernel_source()
    assert "acc_s_cast_gemm_a_staging" in src
    # Verify unmerged producer releases K's forward barrier before waiting on V's bp barrier.
    else_idx = src.rfind("} else {")
    assert else_idx != -1
    prod_src = src[else_idx:]
    k_load_pos = prod_src.find("K_desc")
    v_load_pos = prod_src.find("V_desc")
    assert 0 <= k_load_pos < v_load_pos
    between_kv = prod_src[k_load_pos:v_load_pos]
    assert "tme_barrier_arrive" in between_kv, (
        "Expected K forward barrier arrive before V load/bp-wait in unmerged WS producer"
    )

    q = torch.randn((B, S, H, D), dtype=torch.float16, device="musa") * 0.5
    k = torch.randn((B, S, H, D), dtype=torch.float16, device="musa") * 0.5
    v = torch.randn((B, S, H, D), dtype=torch.float16, device="musa") * 0.5
    out = kernel(q, k, v)
    torch.musa.synchronize()

    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
    ).transpose(1, 2).to(torch.float16)
    torch.testing.assert_close(out, ref, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_per_squad_partial_barrier_in_multi_squad_fa():
    B, H, S, D = 1, 2, 256, 64
    q = torch.randn((B, S, H, D), dtype=torch.float16, device="musa") * 0.5
    k = torch.randn((B, S, H, D), dtype=torch.float16, device="musa") * 0.5
    v = torch.randn((B, S, H, D), dtype=torch.float16, device="musa") * 0.5
    ref = F.scaled_dot_product_attention(
        q.transpose(1, 2).float(),
        k.transpose(1, 2).float(),
        v.transpose(1, 2).float(),
    ).transpose(1, 2).to(torch.float16)

    # 1) FullRow multi-squad FA: each 128T squad owns a disjoint half of
    #    acc_s_cast_gemm_a_staging -> lowers into per-squad 128T barriers.
    prog_fullrow = _flash_attention_prefill(
        B, H, S, D, block_M=128, block_N=64, threads=256,
        policy=T.GemmWarpPolicy.FullRow, num_stages=2,
    )
    kernel_fr = tilelang.compile(
        prog_fullrow,
        out_idx=[3],
        target={"kind": "musa", "arch": "mp_31"},
        pass_configs={"tl.enable_fast_math": True},
    )
    src_fr = kernel_fr.get_kernel_source()
    assert "tl::tme_barrier_init_arrival(10, ((128 + 31) / 32), 0);" in src_fr
    assert "tl::tme_barrier_init_arrival(11, ((128 + 31) / 32), 0);" in src_fr
    assert "tl::tme_barrier_arrive(((((int)threadIdx.x) >> 7) + 10));" in src_fr
    out_fr = kernel_fr(q, k, v)
    torch.musa.synchronize()
    torch.testing.assert_close(out_fr, ref, rtol=1e-2, atol=1e-2)

    # 2) Non-disjoint consumer SMEM (e.g. WS GEMM with shared C + TME store
    #    read by thread 0): must conservatively keep a single 256T barrier.
    @T.prim_func
    def ws_gemm_shared_c(
        A: T.Tensor((128, 128), "float16"),
        B: T.Tensor((128, 128), "float16"),
        C: T.Tensor((128, 128), "float32"),
    ):
        with T.Kernel(1, threads=256):
            As = T.alloc_shared((128, 64), "float16")
            Bs = T.alloc_shared((128, 64), "float16")
            Cl = T.alloc_fragment((128, 128), "float32")
            Cs = T.alloc_shared((128, 128), "float32")
            T.clear(Cl)
            for k in T.Pipelined(2, num_stages=2):
                T.copy(A[:, k * 64 : (k + 1) * 64], As)
                T.copy(B[:, k * 64 : (k + 1) * 64], Bs)
                T.gemm(As, Bs, Cl, transpose_B=True)
            T.copy(Cl, Cs)
            T.copy(Cs, C, prefer_instruction="tma")

    kernel_non_disjoint = tilelang.compile(
        ws_gemm_shared_c,
        out_idx=[2],
        target={"kind": "musa", "arch": "mp_31"},
    )
    src_nd = kernel_non_disjoint.get_kernel_source()
    assert "tl::tme_barrier_init_arrival(5, ((256 + 31) / 32), 0);" in src_nd
    assert "tl::tme_barrier_arrive(5);" in src_nd
    a = torch.randn((128, 128), dtype=torch.float16, device="musa") * 0.1
    b = torch.randn((128, 128), dtype=torch.float16, device="musa") * 0.1
    out_nd = kernel_non_disjoint(a, b)
    torch.musa.synchronize()
    torch.testing.assert_close(out_nd.cpu(), a.cpu().float() @ b.cpu().float().T, rtol=1e-2, atol=1e-2)


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_fragment_a_gemm_disable_flag_fails_loudly():
    prog = _flash_attention_prefill(1, 1, 128, 64, block_M=64, block_N=64, threads=128)
    with pytest.raises(Exception):
        tilelang.compile(
            prog,
            out_idx=[3],
            target={"kind": "musa", "arch": "mp_31"},
            pass_configs={
                "tl.disable_warp_specialized": True,
                "tl.enable_musa_legalize_fragment_a_gemm": False,
            },
        )


@tilelang.testing.requires_musa_compute_version_eq(3, 1)
def test_mp31_smem_guard_rejects_oversized_and_allows_bypass():
    @T.prim_func
    def huge_smem(A: T.Tensor((200, 512), "float16"), B: T.Tensor((200, 512), "float16")):
        with T.Kernel(1, threads=128):
            # 200 * 512 * 2 B = 204,800 B (200 KiB) > 188 KiB (192,512 B) MP31 launch ceiling.
            buf = T.alloc_shared((200, 512), "float16")
            T.copy(A[0, 0], buf)
            T.sync_threads()
            T.copy(buf, B[0, 0])

    with pytest.raises(ValueError, match="MP31 shared memory limit exceeded"):
        tilelang.compile(
            huge_smem,
            out_idx=[1],
            target={"kind": "musa", "arch": "mp_31"},
            pass_configs={"tl.disable_warp_specialized": True},
        )

    # Disabling the guard allows compilation/IR inspection to proceed.
    compiled = tilelang.compile(
        huge_smem,
        out_idx=[1],
        target={"kind": "musa", "arch": "mp_31"},
        pass_configs={
            "tl.disable_warp_specialized": True,
            "tl.enable_musa_smem_limit_check": False,
        },
    )
    assert "main_kernel" in compiled.get_kernel_source() or "huge_smem_kernel" in compiled.get_kernel_source()


if __name__ == "__main__":
    tilelang.testing.main()
