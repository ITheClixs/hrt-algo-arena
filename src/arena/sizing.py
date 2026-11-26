"""Turn forecasts into positions. Sharpe ignores overall scale, so only RELATIVE size matters.

For independent sessions with expected return mu_i and risk sigma_i, the Sharpe-maximising
position is w_i ~ mu_i / (sigma_i^2 + mu_i^2) ~ mu_i / sigma_i^2. In words: bet more where
the edge is large relative to the noise; shrink bets in volatile sessions.
"""

from __future__ import annotations

import numpy as np

EPS = 1e-12


def clip_sd(pred, n_sd: float = 2.0, ref=None) -> np.ndarray:
    """Cap |pred| at n_sd standard deviations (of `ref`, default pred). Tames outliers."""
    p = np.asarray(pred, dtype=float)
    base = p if ref is None else np.asarray(ref, dtype=float)
    cap = n_sd * base.std()
    return np.clip(p, -cap, cap) if cap > 0 else p


def vol_scale(pred, vol, power: float = 1.0, floor_quantile: float = 0.05) -> np.ndarray:
    """pred / vol**power with vol floored at a low quantile (avoids blow-ups)."""
    v = np.asarray(vol, dtype=float)
    floor = np.quantile(v, floor_quantile) if v.size else 0.0
    return np.asarray(pred, dtype=float) / np.maximum(v, max(floor, EPS)) ** power


def sharpe_optimal(mu, sigma) -> np.ndarray:
    """Position maximising Sharpe across independent bets: mu / (sigma^2 + mu^2)."""
    mu, sigma = np.asarray(mu, dtype=float), np.asarray(sigma, dtype=float)
    return mu / (sigma**2 + mu**2 + EPS)


def normalize(pred, target_abs_mean: float = 1.0) -> np.ndarray:
    """Cosmetic rescale (Sharpe unchanged). Handy if the platform limits position size."""
    p = np.asarray(pred, dtype=float)
    scale = np.abs(p).mean()
    return p * (target_abs_mean / scale) if scale > 0 else p


def fit_risk_model(x, y):
    """E[y^2 | x] with non-negative weights (a variance model must not go negative)."""
    from sklearn.linear_model import LinearRegression

    return LinearRegression(positive=True).fit(np.asarray(x, dtype=float), np.asarray(y, dtype=float) ** 2)


def sharpe_position(mu, m2, floor_quantile: float = 0.05) -> np.ndarray:
    """mu / E[r^2] with E[r^2] floored at a low quantile of itself (scale-free)."""
    m2 = np.asarray(m2, dtype=float)
    floor = max(float(np.quantile(m2, floor_quantile)), EPS)
    return np.asarray(mu, dtype=float) / np.maximum(m2, floor)


def risk_features(s, owed: np.ndarray, halfway: int = 49) -> np.ndarray:
    """Per session: GK variance (all, last 10 bars), squared gaps, close-close variance,
    headline counts (all, last 10 bars), distinct companies, owed push squared."""
    sl = slice(0, halfway + 1)
    o, h, lo, c = (np.log(a[:, sl]) for a in (s.open, s.high, s.low, s.close))
    prev = np.concatenate([np.zeros((s.n, 1)), c[:, :-1]], axis=1)
    gk = 0.5 * (h - lo) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    hl = s.headlines[s.headlines["bar_ix"] <= halfway]
    sidx = hl["sidx"].to_numpy()
    recent = (hl["bar_ix"].to_numpy() >= halfway - 9).astype(float)
    companies = hl.groupby("sidx")["entity"].nunique().reindex(range(s.n), fill_value=0).to_numpy()
    return np.column_stack([
        gk.mean(axis=1), gk[:, -10:].mean(axis=1), ((o - prev) ** 2).mean(axis=1),
        ((c - prev) ** 2).mean(axis=1), np.bincount(sidx, minlength=s.n).astype(float),
        np.bincount(sidx, weights=recent, minlength=s.n), companies.astype(float),
        np.asarray(owed, dtype=float) ** 2,
    ])
