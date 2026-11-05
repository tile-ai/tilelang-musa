import os
import random
import pytest

SEED = int(os.environ.get("TL_TEST_SEED", "0"))


def test_seed_fixture_reproducible_python():
    seq1 = [random.random() for _ in range(4)]
    random.seed(SEED)
    seq2 = [random.random() for _ in range(4)]
    assert seq1 == seq2


def test_seed_fixture_reproducible_numpy():
    np = pytest.importorskip("numpy")
    seq1 = np.random.rand(4)
    np.random.seed(SEED)
    seq2 = np.random.rand(4)
    assert np.array_equal(seq1, seq2)


def test_seed_fixture_reproducible_torch():
    torch = pytest.importorskip("torch")
    seq1 = torch.rand(4)
    torch.manual_seed(SEED)
    seq2 = torch.rand(4)
    assert torch.equal(seq1, seq2)


def test_torch_backend_fixture():
    torch = pytest.importorskip("torch")
    assert torch.backends.cudnn.benchmark is False
    assert torch.backends.mudnn.deterministic is True
    assert torch.backends.mudnn.allow_tf32 is False
