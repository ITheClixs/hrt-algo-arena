import numpy as np

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.data import from_long, subset, truncate
from arena.modelapi import as_fit_predict_with_unlabeled
from arena.models import ensure_keys
from arena.references import BaselineDrift
from arena.synth import make_synthetic
from arena.universe import load_universe


class Recorder:
    def __init__(self):
        self.calls = []

    def fit(self, full, y, unlabeled=None):
        shape = None if unlabeled is None else (unlabeled.n, unlabeled.n_bars)
        self.calls.append((full.n, full.n_bars, shape))
        return self

    def predict(self, seen):
        return np.zeros(seen.n)


def test_adapter_passes_test_fold_first_halves_as_unlabeled(keyed, y):
    other = make_synthetic(50, seed=3, start_id=9000)
    oos = truncate(ensure_keys(from_long(other.bars, other.headlines)), 49)
    rec = Recorder()
    fp = as_fit_predict_with_unlabeled(rec, keyed, y, unlabeled=oos)
    assert fp(np.arange(100, 400), np.arange(100)).shape == (100,)
    assert rec.calls == [(300, 100, (150, 50))]


def test_baseline_drift_never_peeks_past_halfway(arcade_dir):
    u = load_universe(arcade_dir)
    fitted = BaselineDrift(SPEC).fit(subset(u.ins, np.arange(150)), u.y[:150])
    test = subset(u.ins, np.arange(150, u.ins.n))
    np.testing.assert_allclose(fitted.predict(test), fitted.predict(truncate(test, 49)))
