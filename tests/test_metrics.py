import math

import numpy as np
import pandas as pd
import pytest

from arena.config import DATATHON_2026
from arena.metrics import bootstrap_sharpe, score, sharpe, sharpe_standard_error, summary


@pytest.fixture()
def pred_ret():
    rng = np.random.default_rng(0)
    pred = rng.normal(size=2000)
    ret = rng.normal(size=2000) * 0.01 + pred * 0.001
    return pred, ret


def test_score_matches_hrt_arcade_formula(pred_ret):
    pred, ret = pred_ret
    pnl = pd.Series(pred * ret)
    expected = pnl.mean() / (pnl.std() + 1e-12) * np.sqrt(252)
    assert score(pred, ret) == pytest.approx(expected)


def test_score_matches_datathon_formula(pred_ret):
    pred, ret = pred_ret
    pnl = pred * ret
    assert score(pred, ret, DATATHON_2026) == pytest.approx(np.mean(pnl) / np.std(pnl) * 16)


def test_sharpe_is_scale_invariant(pred_ret):
    pred, ret = pred_ret
    assert score(pred * 7.3, ret) == pytest.approx(score(pred, ret))


def test_nonfinite_predictions_count_as_zero(pred_ret):
    pred, ret = pred_ret
    broken = pred.copy()
    broken[:10] = np.nan
    fixed = pred.copy()
    fixed[:10] = 0.0
    assert score(broken, ret) == pytest.approx(score(fixed, ret))


def test_shape_mismatch_raises():
    with pytest.raises(ValueError):
        score([1.0, 2.0], [1.0])


def test_sharpe_of_single_value_is_zero():
    assert sharpe([1.0]) == 0.0


def test_summary_perfect_direction():
    ret = np.array([0.01, -0.02, 0.03, -0.01])
    out = summary(np.sign(ret), ret)
    assert out["hit_rate"] == 1.0
    assert out["mean_return_long"] > 0 > out["mean_return_short"]
    assert out["frac_nonzero"] == 1.0
    assert out["correlation"] > 0.9


def test_summary_handles_constant_prediction():
    out = summary(np.zeros(3), np.array([0.1, -0.1, 0.2]))
    assert out["correlation"] == 0.0
    assert math.isnan(out["mean_return_long"])


def test_bootstrap_quantiles_are_ordered(pred_ret):
    pred, ret = pred_ret
    lo, mid, hi = bootstrap_sharpe(pred, ret, n_boot=300)
    assert lo <= mid <= hi
    assert lo < score(pred, ret) < hi


def test_standard_error_rule_of_thumb():
    assert sharpe_standard_error(0.0, 10_000) == pytest.approx(math.sqrt(252) / 100)


def test_score_by_unit_sums_pnl_within_units():
    from arena.metrics import score_by_unit
    pred, ret = np.array([1.0, 1.0, -1.0, 2.0]), np.array([0.01, 0.02, 0.01, -0.01])
    assert score_by_unit(pred, ret, np.array([1, 1, 2, 3])) == pytest.approx(sharpe(np.array([0.03, -0.01, -0.02])))
