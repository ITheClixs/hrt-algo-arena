import numpy as np
import pytest

from arena.blend import BlendModel


class Const:
    def __init__(self, v):
        self.v = v

    def fit(self, full, y, unlabeled=None):
        return self

    def predict(self, seen):
        return self.v * np.arange(1, seen.n + 1, dtype=float)


def test_blend_is_scale_free_per_member(keyed, y):
    a = BlendModel((Const(1.0), Const(100.0)), (1.0, 1.0)).fit(keyed, y).predict(keyed)
    b = BlendModel((Const(1.0),), (2.0,)).fit(keyed, y).predict(keyed)
    np.testing.assert_allclose(a, b)


def test_blend_checks_weights(keyed, y):
    with pytest.raises(ValueError):
        BlendModel((Const(1.0),), (1.0, 2.0))
