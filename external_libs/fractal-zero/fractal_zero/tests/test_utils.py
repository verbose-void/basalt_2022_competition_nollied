# import torch
import numpy as np
import torch

from fractal_zero.utils import cloning_primitive


def test_cloning_primitive():
    np.random.seed(1)

    for _ in range(100):
        n = 100

        x = np.arange(n)
        orig_x = x.copy()
        partners = np.random.choice(range(n), size=n)
        clone_mask = (np.random.uniform(size=n) < 0.1).astype(bool)

        # yes, these indices are commutative.
        np.testing.assert_equal(x[partners[clone_mask]], x[partners][clone_mask])

        np_cloned = cloning_primitive(x.copy(), partners, clone_mask)
        th_cloned = cloning_primitive(
            torch.tensor(x.copy()), torch.tensor(partners), torch.tensor(clone_mask)
        )
        list_cloned = cloning_primitive(x.copy().tolist(), partners, clone_mask)

        np.testing.assert_equal(x, orig_x)
        assert np_cloned.tolist() == list_cloned
        assert th_cloned.tolist() == list_cloned

    np.random.seed()
