"""Tests for deterministic seed management."""

import random

import numpy as np
import torch

from c5_snn.utils.seed import set_global_seed


def _sample_all_simple():
    """Draw one sample from each RNG, returning comparable values."""
    r = random.random()
    n = float(np.random.rand())
    t = torch.rand(5).clone()
    return r, n, t


class TestSetGlobalSeed:
    def test_same_seed_produces_identical_sequences(self):
        """Two calls with the same seed must produce identical random sequences."""
        set_global_seed(42)
        r1, n1, t1 = _sample_all_simple()

        set_global_seed(42)
        r2, n2, t2 = _sample_all_simple()

        assert r1 == r2, "random.random() not deterministic"
        assert n1 == n2, "numpy.random.rand() not deterministic"
        assert torch.equal(t1, t2), "torch.rand() not deterministic"

    def test_different_seeds_produce_different_sequences(self):
        """Two calls with different seeds must produce different random sequences."""
        set_global_seed(42)
        r1, n1, t1 = _sample_all_simple()

        set_global_seed(99)
        r2, n2, t2 = _sample_all_simple()

        # At least one must differ (extremely unlikely all match by chance)
        differs = (r1 != r2) or (n1 != n2) or (not torch.equal(t1, t2))
        assert differs, "Different seeds produced identical sequences"

    def test_cudnn_flags_set(self):
        """Verify cudnn deterministic flags are set after seeding."""
        set_global_seed(0)
        assert torch.backends.cudnn.deterministic is True
        assert torch.backends.cudnn.benchmark is False
