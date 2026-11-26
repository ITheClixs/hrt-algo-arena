import numpy as np
import pandas as pd

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.data import subset, truncate, with_column
from arena.generator import (
    KERNEL_EDINBURGH,
    KERNEL_FLAT10,
    GeneratorModel,
    PushRates,
    done_weight,
    fit_push_rates,
    news_signal,
    owed_weight,
)
from arena.universe import load_universe


def test_owed_weight_edinburgh_kernel():
    np.testing.assert_allclose(owed_weight(np.array([39, 40, 45, 49]), KERNEL_EDINBURGH, 49, 10),
                               [0.0, 0.5, 5.5, 9.5])


def test_done_weight_edinburgh_kernel():
    np.testing.assert_allclose(done_weight(np.array([30, 45, 49]), KERNEL_EDINBURGH, 49), [10.0, 4.5, 0.5])


def test_fit_push_rates_recovers_planted_rates(keyed):
    fit = fit_push_rates([(keyed, keyed.n_bars - 1)], key="key", kernel=KERNEL_FLAT10, lam=1.0)
    planted = keyed.headlines.groupby("key")["direction"].mean()
    common = planted.index.intersection(fit.rates.index)
    assert np.corrcoef(fit.rates[common], planted[common])[0, 1] > 0.95


def test_unseen_type_contributes_nothing(keyed):
    rates = PushRates(rates=pd.Series({"never seen": 1.0}), kernel=KERNEL_EDINBURGH, key="key")
    assert np.all(news_signal(truncate(keyed, 49), rates, 49, 10) == 0.0)


def test_generator_model_never_peeks_past_halfway(arcade_dir):
    u = load_universe(arcade_dir)
    fitted = GeneratorModel(SPEC).fit(subset(u.ins, np.arange(150)), u.y[:150], unlabeled=u.oos)
    test = subset(u.ins, np.arange(150, u.ins.n))
    np.testing.assert_allclose(fitted.predict(test), fitted.predict(truncate(test, 49)))


def test_sessions_without_main_headlines_get_finite_predictions(arcade_dir):
    u = load_universe(arcade_dir)
    fitted = GeneratorModel(SPEC).fit(u.ins, u.y, unlabeled=u.oos)
    silent = with_column(u.oos, "in_main", np.zeros(len(u.oos.headlines), dtype=bool))
    assert np.isfinite(fitted.predict(silent)).all()


def test_coarse_types_merges_slot_variants():
    from arena.text import coarse_types
    a = pd.Series(["announces partnership with a global retailer",
                   "announces partnership with a major logistics provider",
                   "files routine patent applications"])
    out = coarse_types(a)
    assert out[0] == out[1] and out[2] == a[2]


def test_lag_profile_peaks_at_the_planted_duration(keyed):
    from arena.generator import lag_profile
    prof = lag_profile(keyed, key="key", max_lag=15)
    steps = np.diff(prof)
    assert steps[:10].mean() > 5 * abs(steps[11:].mean())


def test_extra_features_add_finite_columns(arcade_dir):
    u = load_universe(arcade_dir)
    names = ("news_count", "live_count", "conflict", "repeat_push", "peer_push")
    plain = GeneratorModel(SPEC)
    extra = GeneratorModel(SPEC, extra_features=names)
    rates = plain.fit(u.ins, u.y).rates
    x0, x1 = plain.features(plain.prepare(u.oos), rates), extra.features(extra.prepare(u.oos), rates)
    assert x1.shape[1] == x0.shape[1] + 7 and np.isfinite(x1).all()


def test_outsider_and_surprise_features_are_finite_and_leak_free(arcade_dir):
    u = load_universe(arcade_dir)
    model = GeneratorModel(SPEC, extra_features=("surprise", "outsider_push"))
    fitted = model.fit(subset(u.ins, np.arange(150)), u.y[:150], unlabeled=u.oos)
    test = subset(u.ins, np.arange(150, u.ins.n))
    p = fitted.predict(test)
    assert np.isfinite(p).all() and len(fitted.mu) == 1 + 7 + 2
    np.testing.assert_allclose(p, fitted.predict(truncate(test, 49)))


def test_inv_vol_feature_is_one_over_visible_vol_and_leak_free(arcade_dir):
    from arena.generator import gk_vol

    u = load_universe(arcade_dir)
    model = GeneratorModel(SPEC, extra_features=("inv_vol",))
    fitted = model.fit(subset(u.ins, np.arange(150)), u.y[:150], unlabeled=u.oos)
    visible = model.prepare(u.oos)
    x = model.features(visible, fitted.rates)
    np.testing.assert_allclose(x[:, -1], 1.0 / gk_vol(visible, 49))
    test = subset(u.ins, np.arange(150, u.ins.n))
    np.testing.assert_allclose(fitted.predict(test), fitted.predict(truncate(test, 49)))
