"""Event studies and headline-drift features.

Observed in the public Algo Arena data: each headline template moves the price at a roughly
constant per-bar rate for ~10 bars after publication, then stops. So the expected return
over the holding window is the drift still "owed" by recent headlines.

All functions read the headline key from `s.headlines[key]` (add it with
`arena.data.with_column(s, "key", templates(...))`).
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from arena.data import Sessions
from arena.features import bar_log_returns


def _arrays(s: Sessions, key: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    if key not in s.headlines.columns:
        raise KeyError(f"headlines have no {key!r} column; add one with with_column()")
    hl = s.headlines
    return hl["sidx"].to_numpy(), hl["bar_ix"].to_numpy(), hl[key].to_numpy()


def event_study(s: Sessions, key: str = "key", horizon: int = 20, min_count: int = 30) -> pd.DataFrame:
    """Mean cumulative log return from close(t-1) to close(t+lag), lag = 0..horizon, per key.

    Needs bars after the headline, so run it on in-sample (full) sessions.
    """
    sidx, bar, keys = _arrays(s, key)
    ok = bar + horizon < s.n_bars
    sidx, bar, keys = sidx[ok], bar[ok], keys[ok]
    lc = s.log_close()
    prev = np.where(bar > 0, lc[sidx, np.maximum(bar - 1, 0)], np.log(s.open[sidx, 0]))
    cum = np.stack([lc[sidx, bar + lag] - prev for lag in range(horizon + 1)], axis=1)
    frame = pd.DataFrame(cum, columns=list(range(horizon + 1)))
    frame["key"] = keys
    grouped = frame.groupby("key")
    out = grouped.mean()
    out.insert(0, "n", grouped.size())
    out = out[out["n"] >= min_count]
    return out.sort_values(min(10, horizon))


def fit_drift_rates(
    s: Sessions, key: str = "key", duration: int = 10, shrinkage: float = 0.0
) -> pd.Series:
    """Average per-bar log return over the `duration` bars after each headline, per key.

    Only uses bars present in `s`, so truncated (first-half) sessions work too: you can add
    out-of-sample first halves to the estimate without touching any label.
    `shrinkage` pulls rare keys toward 0: rate * n / (n + shrinkage).
    """
    sidx, bar, keys = _arrays(s, key)
    r = bar_log_returns(s.open, s.close)
    total, count = np.zeros(len(bar)), np.zeros(len(bar))
    for lag in range(1, duration + 1):
        t = bar + lag
        m = t < s.n_bars
        total[m] += r[sidx[m], t[m]]
        count[m] += 1
    has = count > 0
    per_headline = pd.Series(total[has] / count[has])
    grouped = per_headline.groupby(keys[has])
    n = grouped.size()
    return (grouped.mean() * n / (n + shrinkage)).rename("rate")


def _overlap(bar: np.ndarray, duration: int, lo: int, hi: int) -> np.ndarray:
    """How many of the bars t+1..t+duration fall inside [lo, hi]."""
    start = np.maximum(bar + 1, lo)
    end = np.minimum(bar + duration, hi)
    return np.clip(end - start + 1, 0, None)


def _per_session(s: Sessions, rates: pd.Series, key: str, last_bar: int, lo: int, hi: int,
                 duration: int) -> np.ndarray:
    sidx, bar, keys = _arrays(s, key)
    m = bar <= last_bar
    r = pd.Series(keys[m]).map(rates).fillna(0.0).to_numpy()
    w = r * _overlap(bar[m], duration, lo, hi)
    return np.bincount(sidx[m], weights=w, minlength=s.n)


def remaining_drift(s: Sessions, rates: pd.Series, halfway: int, horizon: int,
                    key: str = "key", duration: int = 10) -> np.ndarray:
    """Expected log drift over bars halfway+1..halfway+horizon from headlines seen so far."""
    return _per_session(s, rates, key, halfway, halfway + 1, halfway + horizon, duration)


def realized_drift(s: Sessions, rates: pd.Series, halfway: int, key: str = "key",
                   duration: int = 10) -> np.ndarray:
    """Expected log drift that should already be in the price by the close of `halfway`."""
    return _per_session(s, rates, key, halfway, 0, halfway, duration)


def recent_count(s: Sessions, first_bar: int, last_bar: int) -> np.ndarray:
    """Number of headlines per session in [first_bar, last_bar]."""
    hl = s.headlines
    m = (hl["bar_ix"] >= first_bar) & (hl["bar_ix"] <= last_bar)
    return np.bincount(hl.loc[m, "sidx"].to_numpy(), minlength=s.n).astype(float)
