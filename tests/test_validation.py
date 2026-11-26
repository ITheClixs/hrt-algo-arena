import numpy as np
import pytest

from arena.cv import signal_fit_predict
from arena.validation import (
    CONFIRM_SEEDS,
    DEV_SEEDS,
    KEEP_Z,
    compare,
    evaluate,
    paired_bootstrap_se,
)


def test_constants_match_the_spec():
    assert DEV_SEEDS == (42, 43, 44) and CONFIRM_SEEDS == (1042, 1043, 1044) and KEEP_Z == 2.5


def test_evaluate_pools_out_of_fold_predictions_per_seed(y):
    signal = y + np.random.default_rng(0).normal(0, y.std(), len(y))
    ev = evaluate("noisy", signal_fit_predict(signal), y)
    assert ev.oof.shape == (3, len(y)) and len(ev.seed_scores) == 3 and len(ev.fold_scores) == 15
    np.testing.assert_allclose(ev.oof[0], signal)
    assert ev.mean == pytest.approx(np.mean(ev.seed_scores))
    assert ev.worst_fold == min(ev.fold_scores)


def test_compare_keeps_only_a_clear_improvement(y):
    rng = np.random.default_rng(1)
    good = evaluate("good", signal_fit_predict(y + rng.normal(0, 0.5 * y.std(), len(y))), y)
    bad = evaluate("bad", signal_fit_predict(y + rng.normal(0, 3 * y.std(), len(y))), y)
    assert compare(good, bad, y).keep
    assert not compare(bad, good, y).keep
    same = compare(good, good, y)
    assert same.delta == 0 and not same.keep


def test_paired_bootstrap_se_shrinks_for_similar_predictions(y):
    rng = np.random.default_rng(2)
    a = y + rng.normal(0, y.std(), len(y))
    near = a + rng.normal(0, 0.01 * y.std(), len(y))
    far = rng.normal(0, y.std(), len(y))
    assert paired_bootstrap_se(a, near, y) < paired_bootstrap_se(a, far, y)


def test_compare_requires_matching_seeds(y):
    sig = signal_fit_predict(y)
    with pytest.raises(ValueError, match="seeds"):
        compare(evaluate("a", sig, y), evaluate("b", sig, y, seeds=(1, 2, 3)), y)
