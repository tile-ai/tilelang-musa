import tilelang.testing

from . import example_gqa_fwd_bshd
from . import example_mha_fwd_bshd
from . import example_mha_fwd_varlen
from . import example_mha_fwd_bhsd
from . import example_gqa_fwd_varlen


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_example_gqa_fwd_bshd():
    example_gqa_fwd_bshd.main(batch=1, heads=16, seq_len=1024, dim=128, is_causal=False, groups=16, tune=False)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_example_mha_fwd_bhsd():
    example_mha_fwd_bhsd.main()


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_example_mha_fwd_bshd():
    example_mha_fwd_bshd.main(batch=1, seq_len=256)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_example_mha_fwd_varlen():
    example_mha_fwd_varlen.main(batch=4, heads=16, seq_len=512, dim=64, causal=False)
    example_mha_fwd_varlen.main(batch=4, heads=16, seq_len=512, dim=64, causal=True)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_example_gqa_fwd_varlen():
    example_gqa_fwd_varlen.main(batch=4, heads=16, q_seqlen=512, k_seqlen=512, dim=64, is_causal=False)
    example_gqa_fwd_varlen.main(batch=4, heads=16, q_seqlen=512, k_seqlen=512, dim=64, is_causal=True)


if __name__ == "__main__":
    tilelang.testing.main()
