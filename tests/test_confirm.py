import numpy as np

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.confirm import later_window_check
from arena.generator import GeneratorModel
from arena.universe import load_universe


class _Noise:
    """A model with no information: random positions."""

    def fit(self, full, y, unlabeled=None):
        return self

    def predict(self, seen):
        return np.random.default_rng(seen.n).normal(size=seen.n)


def test_same_model_gives_zero_and_signal_beats_noise(arcade_dir):
    u = load_universe(arcade_dir)
    model = GeneratorModel(SPEC)
    same = later_window_check(model, model, u, SPEC, seeds=(42,), k=3, n_boot=50)
    assert same.delta == 0 and same.z == 0 and len(same.delta_by_offset) == 4
    better = later_window_check(model, _Noise(), u, SPEC, seeds=(42,), k=3, n_boot=50)
    assert better.delta > 0 and better.se > 0 and better.z > 0
    assert "later windows" in str(better)


def test_offsets_that_do_not_fit_are_skipped(arcade_dir):
    u = load_universe(arcade_dir)
    model = GeneratorModel(SPEC)
    check = later_window_check(model, model, u, SPEC, seeds=(42,), k=3, n_boot=10, offsets=(10, 45))
    assert check.offsets == (10,)
