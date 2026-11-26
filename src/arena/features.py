"""Price features from the visible bars. All functions take (n_sessions, n_bars) arrays."""

from __future__ import annotations

import numpy as np
import pandas as pd

from arena.data import Sessions

EPS = 1e-12
LN2 = np.log(2.0)


def bar_log_returns(open_: np.ndarray, close: np.ndarray) -> np.ndarray:
    """Per-bar log return; bar 0 is measured from its open."""
    lc = np.log(close)
    first = lc[:, :1] - np.log(open_[:, :1])
    return np.concatenate([first, np.diff(lc, axis=1)], axis=1)


def _tail_mean(x: np.ndarray, window: int | None) -> np.ndarray:
    return x.mean(axis=1) if window is None else x[:, -window:].mean(axis=1)


def realized_vol(open_: np.ndarray, close: np.ndarray, window: int | None = None) -> np.ndarray:
    """Per-bar close-to-close volatility."""
    return np.sqrt(_tail_mean(bar_log_returns(open_, close) ** 2, window))


def parkinson_vol(high: np.ndarray, low: np.ndarray, window: int | None = None) -> np.ndarray:
    """Per-bar high-low range volatility. Less noisy than close-to-close on short windows."""
    return np.sqrt(_tail_mean(np.log(high / low) ** 2 / (4 * LN2), window))


def garman_klass_vol(o, h, lo, c, window: int | None = None) -> np.ndarray:
    v = 0.5 * np.log(h / lo) ** 2 - (2 * LN2 - 1) * np.log(c / o) ** 2
    return np.sqrt(np.maximum(_tail_mean(v, window), 0.0))


def momentum(close: np.ndarray, lookback: int) -> np.ndarray:
    """Log return over the last `lookback` bars."""
    if not 0 < lookback < close.shape[1]:
        raise ValueError(f"lookback={lookback} needs 1..{close.shape[1] - 1}")
    return np.log(close[:, -1] / close[:, -1 - lookback])


def up_ratio(close: np.ndarray) -> np.ndarray:
    return (np.diff(close, axis=1) > 0).mean(axis=1)


def price_features(s: Sessions) -> pd.DataFrame:
    """Standard feature table, one row per session (index = session id)."""
    o, h, lo, c = s.open, s.high, s.low, s.close
    n_bars = s.n_bars
    lb = {k: min(k, n_bars - 1) for k in (5, 10, 20)}
    pk, pk10 = parkinson_vol(h, lo), parkinson_vol(h, lo, lb[10])
    ret_all = np.log(c[:, -1] / o[:, 0])
    mom = {k: momentum(c, v) for k, v in lb.items()}
    feats = {
        "ret_all": ret_all,
        "mom_5": mom[5],
        "mom_10": mom[10],
        "mom_20": mom[20],
        "rv_all": realized_vol(o, c),
        "rv_10": realized_vol(o, c, lb[10]),
        "pk_all": pk,
        "pk_10": pk10,
        "gk_all": garman_klass_vol(o, h, lo, c),
        "range_trend": pk10 / (pk + EPS),
        "up_ratio": up_ratio(c),
        "ret_all_z": ret_all / (pk * np.sqrt(n_bars) + EPS),
        "mom_10_z": mom[10] / (pk * np.sqrt(lb[10]) + EPS),
        "dist_from_high": np.log(c[:, -1] / h.max(axis=1)),
        "dist_from_low": np.log(c[:, -1] / lo.min(axis=1)),
    }
    return pd.DataFrame(feats, index=pd.Index(s.ids, name="session"))
