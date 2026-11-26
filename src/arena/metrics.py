"""Scoring that mirrors HRT's official evaluation.

Arcade scorer (sessions.py): pnl = pred * returns; sharpe = mean(pnl) / (std(pnl) + 1e-12) * sqrt(252)
with pandas std (ddof=1). Datathon scorer used numpy std (ddof=0) and a factor of 16.
"""

from __future__ import annotations

import math

import numpy as np
import pandas as pd

from arena.config import ARCADE_ALGO_ARENA, TaskSpec

EPS = 1e-12


def clean(pred) -> np.ndarray:
    """Non-finite predictions count as 0, exactly like the official scorer."""
    p = np.asarray(pred, dtype=float)
    return np.where(np.isfinite(p), p, 0.0)


def sharpe(pnl, annualization: float = math.sqrt(252), ddof: int = 1) -> float:
    x = np.asarray(pnl, dtype=float)
    if x.size < 2:
        return 0.0
    return float(x.mean() / (x.std(ddof=ddof) + EPS) * annualization)


def score(pred, returns, spec: TaskSpec = ARCADE_ALGO_ARENA) -> float:
    """The number the leaderboard shows."""
    p, r = clean(pred), np.asarray(returns, dtype=float)
    if p.shape != r.shape:
        raise ValueError(f"pred shape {p.shape} != returns shape {r.shape}")
    return sharpe(p * r, spec.annualization, spec.ddof)


def _mean_or_nan(x: np.ndarray) -> float:
    return float(x.mean()) if x.size else float("nan")


def summary(pred, returns, spec: TaskSpec = ARCADE_ALGO_ARENA) -> dict[str, float]:
    """Sharpe plus the diagnostics HRT's notebook prints."""
    p, r = clean(pred), np.asarray(returns, dtype=float)
    pnl, size = p * r, np.abs(p)
    active = p != 0
    corr = float(np.corrcoef(p, r)[0, 1]) if p.std() > 0 and r.std() > 0 else 0.0
    return {
        "sharpe": score(p, r, spec),
        "total_pnl": float(pnl.sum()),
        "correlation": corr,
        "hit_rate": _mean_or_nan((np.sign(p[active]) == np.sign(r[active])).astype(float)),
        "frac_nonzero": float(active.mean()),
        "mean_return_long": _mean_or_nan(r[p > 0]),
        "mean_return_short": _mean_or_nan(r[p < 0]),
        "avg_abs_position": float(size.mean()),
        "return_on_capital": float(pnl.sum() / (size.sum() + EPS)),
    }


def bootstrap_sharpe(
    pred,
    returns,
    spec: TaskSpec = ARCADE_ALGO_ARENA,
    n_boot: int = 1000,
    seed: int = 0,
    quantiles: tuple[float, ...] = (0.05, 0.5, 0.95),
) -> tuple[float, ...]:
    """Resample sessions with replacement -> quantiles of Sharpe. Wide band = noisy idea."""
    pnl = clean(pred) * np.asarray(returns, dtype=float)
    rng = np.random.default_rng(seed)
    out = []
    for start in range(0, n_boot, 100):
        idx = rng.integers(0, pnl.size, size=(min(100, n_boot - start), pnl.size))
        sample = pnl[idx]
        sd = sample.std(axis=1, ddof=spec.ddof) + EPS
        out.append(sample.mean(axis=1) / sd * spec.annualization)
    return tuple(float(q) for q in np.quantile(np.concatenate(out), quantiles))


def sharpe_standard_error(
    annual_sharpe: float, n_sessions: int, annualization: float = math.sqrt(252)
) -> float:
    """Lo (2002) iid standard error of an annualised Sharpe measured on n sessions.

    Rule of thumb for 10k sessions: SE ~ 0.16, so a 0.1 improvement is noise.
    """
    per_period = annual_sharpe / annualization
    return math.sqrt((1 + 0.5 * per_period**2) / n_sessions) * annualization


def score_by_unit(pred, returns, unit, spec: TaskSpec = ARCADE_ALGO_ARENA) -> float:
    """Sharpe of PnL summed per scoring unit (e.g. per day or per stock-day) before scoring."""
    pnl = clean(pred) * np.asarray(returns, dtype=float)
    per_unit = pd.Series(pnl).groupby(np.asarray(unit)).sum().to_numpy()
    return sharpe(per_unit, spec.annualization, spec.ddof)
