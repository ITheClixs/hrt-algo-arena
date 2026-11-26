import numpy as np
import pytest

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.pipeline import CANDIDATES, Candidate, write_version
from arena.universe import load_universe


def test_candidate_labels_are_unique_and_the_primary_comes_first():
    labels = [c.label for c in CANDIDATES]
    assert labels[0] == "primary" and len(set(labels)) == len(labels)
    assert {"volinv", "baseline"} <= set(labels)


class _Broken:
    def fit(self, full, y, unlabeled=None):
        return self

    def predict(self, seen):
        p = np.ones(seen.n)
        p[3] = np.nan
        return p


def test_write_version_refuses_non_finite_predictions(arcade_dir, tmp_path):
    u = load_universe(arcade_dir)
    broken = Candidate("broken", lambda spec: _Broken(), "test")
    with pytest.raises(ValueError, match="1 non-finite"):
        write_version(broken, u, SPEC, tmp_path, 9, with_extra=False)
    assert not list(tmp_path.glob("*.csv"))
