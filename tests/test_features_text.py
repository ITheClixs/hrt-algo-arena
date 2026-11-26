import numpy as np
import pandas as pd
import pytest

from arena.features import (
    bar_log_returns,
    garman_klass_vol,
    momentum,
    parkinson_vol,
    price_features,
)
from arena.text import detect_entities, keyword_score, mask_numbers, session_documents, templates


def test_bar_log_returns_sum_to_total(sessions):
    r = bar_log_returns(sessions.open, sessions.close)
    total = np.log(sessions.close[:, -1] / sessions.open[:, 0])
    np.testing.assert_allclose(r.sum(axis=1), total)


def test_parkinson_known_value():
    high, low = np.full((1, 5), np.exp(0.1)), np.ones((1, 5))
    assert parkinson_vol(high, low)[0] == pytest.approx(np.sqrt(0.01 / (4 * np.log(2))))


def test_garman_klass_is_nonnegative(sessions):
    s = sessions
    assert (garman_klass_vol(s.open, s.high, s.low, s.close, 10) >= 0).all()


def test_momentum_bounds(sessions):
    assert momentum(sessions.close, 5).shape == (sessions.n,)
    with pytest.raises(ValueError):
        momentum(sessions.close, 100)


def test_price_features_complete(sessions):
    feats = price_features(sessions)
    assert len(feats) == sessions.n
    assert np.isfinite(feats.to_numpy()).all()


def test_mask_numbers():
    out = mask_numbers(pd.Series(["secures $120M deal", "up 25% in 2026", "$1.2B buyback"]))
    assert out.tolist() == ["secures $X deal", "up N% in N", "$X buyback"]


def test_detect_entities_handles_three_word_names():
    hl = pd.Series([
        "Jovik Asset Group secures deal", "Jovik Asset Group misses estimates",
        "Alvex Systems wins award", "Alvex Systems loses contract",
    ])
    assert detect_entities(hl).tolist() == ["Jovik Asset Group"] * 2 + ["Alvex Systems"] * 2


def test_templates_mask_company_and_numbers():
    hl = pd.Series([
        "Alvex Systems secures $50M contract", "Borin Energy secures $70M contract",
        "Alvex Systems misses estimates by 4%", "Borin Energy misses estimates by 9%",
    ])
    assert templates(hl).nunique() == 2


def test_keyword_score_signs():
    s = keyword_score(pd.Series(["X secures record deal", "X misses, faces lawsuit", "X opens office"]))
    assert s.tolist() == [2.0, -2.0, 0.0]


def test_session_documents(sessions):
    docs = session_documents(sessions, first_bar=40, last_bar=49)
    assert len(docs) == sessions.n
    assert all(isinstance(d, str) for d in docs)


def test_tfidf_ridge_learns_dated_headlines():
    from arena.cv import cross_validate, walk_forward_by_time
    from arena.models import make_tfidf_ridge
    from arena.synth import make_dated_headlines
    df = make_dated_headlines(n_days=200, per_day=20)
    assert list(df.columns) == ["time", "ticker", "text", "target"] and df["time"].is_monotonic_increasing
    y = df["target"].to_numpy()

    def fp(tr, te):
        pipe = make_tfidf_ridge(alpha=1.0, min_df=1)
        return pipe.fit(df["text"].iloc[tr], y[tr]).predict(df["text"].iloc[te])

    assert cross_validate(fp, y, walk_forward_by_time(df["time"], k=4)).mean > 0.5
