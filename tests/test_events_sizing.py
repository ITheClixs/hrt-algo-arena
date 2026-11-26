import numpy as np
import pandas as pd
import pytest

from arena.data import from_long, truncate, with_column
from arena.events import (
    _overlap,
    event_study,
    fit_drift_rates,
    realized_drift,
    recent_count,
    remaining_drift,
)
from arena.sizing import clip_sd, normalize, sharpe_optimal, vol_scale


def test_overlap_counts_bars_inside_window():
    # headline at bar 45 drifts over bars 46..55; window 50..59 -> 6 bars
    assert _overlap(np.array([45]), 10, 50, 59).tolist() == [6]
    assert _overlap(np.array([30]), 10, 50, 59).tolist() == [0]


def _tiny_sessions():
    bars = pd.DataFrame({
        "session": np.repeat([7, 8], 60), "bar_ix": np.tile(np.arange(60), 2),
        "open": 1.0, "high": 1.0, "low": 1.0, "close": 1.0,
    })
    hl = pd.DataFrame({"session": [7, 7, 8], "bar_ix": [45, 30, 49], "headline": ["a", "b", "a"]})
    return with_column(from_long(bars, hl), "key", ["b", "a", "a"])  # rows sorted by bar


def test_remaining_and_realized_drift_hand_example():
    s = _tiny_sessions()
    rates = pd.Series({"a": 1.0, "b": 10.0})
    rem = remaining_drift(s, rates, halfway=49, horizon=10)
    # session 7: 'b'@30 owes 0, 'a'@45 owes 6 bars; session 8: 'a'@49 owes 10 bars
    assert rem.tolist() == [6.0, 10.0]
    real = realized_drift(s, rates, halfway=49)
    # session 7: 'b'@30 realized 10 bars * 10 + 'a'@45 realized 4 bars
    assert real.tolist() == [104.0, 0.0]
    assert recent_count(s, 40, 49).tolist() == [1.0, 1.0]


def test_missing_key_raises(sessions):
    with pytest.raises(KeyError):
        fit_drift_rates(sessions, key="nope")


def test_fit_drift_rates_recover_true_direction(keyed):
    rates = fit_drift_rates(keyed)
    truth = keyed.headlines.groupby("key")["direction"].first()
    assert np.corrcoef(rates.reindex(truth.index), truth)[0, 1] > 0.8


def test_fit_rates_on_truncated_sessions_and_shrinkage(keyed):
    rates = fit_drift_rates(truncate(keyed, 49))
    shrunk = fit_drift_rates(truncate(keyed, 49), shrinkage=1e6)
    assert rates.abs().sum() > 0
    assert (shrunk.abs() < rates.abs() + 1e-15).all()


def test_event_study_shape(keyed):
    es = event_study(keyed, horizon=12, min_count=5)
    assert list(es.columns) == ["n", *range(13)]
    assert (es["n"] >= 5).all()


def test_clip_sd():
    x = np.array([0.0, 1.0, -1.0, 100.0])
    assert clip_sd(x, 1.0).max() == pytest.approx(x.std())
    assert (clip_sd(np.zeros(3)) == 0).all()


def test_vol_scale_floors_tiny_vol():
    out = vol_scale(np.ones(4), np.array([1e-9, 1.0, 1.0, 1.0]), floor_quantile=0.5)
    assert np.isfinite(out).all() and out.max() == pytest.approx(1.0)


def test_sharpe_optimal_and_normalize():
    w = sharpe_optimal(np.array([0.01, -0.01]), np.array([0.1, 0.2]))
    assert w[0] > 0 > w[1] and abs(w[0]) > abs(w[1])
    assert np.abs(normalize(np.array([2.0, -4.0]))).mean() == pytest.approx(1.0)
    assert (normalize(np.zeros(2)) == 0).all()


def test_risk_model_tracks_variance():
    from arena.sizing import fit_risk_model
    rng = np.random.default_rng(0)
    sd = rng.uniform(0.5, 2.0, 5000)
    y = rng.normal(0, sd)
    model = fit_risk_model(np.column_stack([sd**2, rng.normal(size=5000) ** 2]), y)
    assert np.corrcoef(model.predict(np.column_stack([sd**2, np.ones(5000)])), sd**2)[0, 1] > 0.95


def test_sharpe_position_is_scale_free_and_floored():
    from arena.sizing import sharpe_position
    mu, m2 = np.array([1.0, -2.0, 3.0]), np.array([1.0, 0.0, 2.0])
    p = sharpe_position(mu, m2)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(sharpe_position(3 * mu, 3 * m2), p)


def test_generator_risk_model_sizing_is_finite_and_leak_free(arcade_dir):
    from arena.config import ARCADE_ALGO_ARENA as SPEC
    from arena.data import subset, truncate
    from arena.generator import GeneratorModel
    from arena.universe import load_universe
    u = load_universe(arcade_dir)
    fitted = GeneratorModel(SPEC, sizing="risk_model").fit(subset(u.ins, np.arange(150)), u.y[:150], unlabeled=u.oos)
    test = subset(u.ins, np.arange(150, u.ins.n))
    p = fitted.predict(test)
    assert np.isfinite(p).all()
    np.testing.assert_allclose(p, fitted.predict(truncate(test, 49)))


def test_second_moment_options_default_to_the_edinburgh_form(arcade_dir):
    from arena.config import ARCADE_ALGO_ARENA as SPEC
    from arena.generator import GeneratorModel
    from arena.universe import load_universe
    u = load_universe(arcade_dir)
    base = GeneratorModel(SPEC).fit(u.ins, u.y, unlabeled=u.oos).predict(u.oos)
    same = GeneratorModel(SPEC, m2_weight=1.0, residual_var=False).fit(u.ins, u.y, unlabeled=u.oos).predict(u.oos)
    other = GeneratorModel(SPEC, m2_weight=2.0, residual_var=True).fit(u.ins, u.y, unlabeled=u.oos).predict(u.oos)
    np.testing.assert_allclose(same, base)
    assert np.isfinite(other).all() and not np.allclose(other, base)
