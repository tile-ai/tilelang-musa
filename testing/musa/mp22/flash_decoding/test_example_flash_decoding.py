import tilelang.testing

from . import example_mha_inference
from . import example_gqa_decode_varlen_logits


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_example_example_mha_inference():
    example_mha_inference.main(BATCH=1, H=32, Q_CTX=128, KV_CTX=2048, D_HEAD=128, causal=False)


@tilelang.testing.requires_musa_compute_version_eq(2, 2)
def test_example_example_gqa_decode_varlen_logits():
    example_gqa_decode_varlen_logits.main()


if __name__ == "__main__":
    tilelang.testing.main()
