"""Cross-validation harness. Every idea gets the same folds so numbers are comparable.

A `FitPredict` is any function (train_idx, test_idx) -> predictions for test_idx. Fit
EVERYTHING that looks at labels or future bars (rates, scalers, vectorizers) inside it.
"""

from __future__ import annotations

from collections.abc import Callable, Sequence
from dataclasses import dataclass

import numpy as np
import pandas as pd

from arena.config import ARCADE_ALGO_ARENA, TaskSpec
from arena.metrics import score

FitPredict = Callable[[np.ndarray, np.ndarray], np.ndarray]
Folds = list[tuple[np.ndarray, np.ndarray]]


def kfold(n: int, k: int = 5, seed: int = 0, shuffle: bool = True) -> Folds:
    order = np.random.default_rng(seed).permutation(n) if shuffle else np.arange(n)
    chunks = np.array_split(order, k)
    return [
        (np.sort(np.concatenate([c for j, c in enumerate(chunks) if j != i])), np.sort(chunk))
        for i, chunk in enumerate(chunks)
    ]


def blocked(n: int, k: int = 5) -> Folds:
    """Contiguous folds in id order (use when ids may encode time)."""
    return kfold(n, k, shuffle=False)


def walk_forward(n: int, k: int = 5, min_train_frac: float = 0.5) -> Folds:
    """Expanding window: train on everything before each test block. For real, dated data."""
    start = int(n * min_train_frac)
    edges = np.linspace(start, n, k + 1).astype(int)
    return [(np.arange(0, a), np.arange(a, b)) for a, b in zip(edges[:-1], edges[1:], strict=True)]


@dataclass(frozen=True)
class CVResult:
    fold_sharpes: tuple[float, ...]
    oof_sharpe: float  # Sharpe of pooled out-of-fold predictions (averaged over repeats)

    @property
    def mean(self) -> float:
        return float(np.mean(self.fold_sharpes))

    @property
    def std(self) -> float:
        return float(np.std(self.fold_sharpes))

    @property
    def worst(self) -> float:
        return float(np.min(self.fold_sharpes))

    def __str__(self) -> str:
        return (
            f"CV Sharpe mean {self.mean:.3f} | std {self.std:.3f} | worst {self.worst:.3f} "
            f"| pooled OOF {self.oof_sharpe:.3f} | folds {len(self.fold_sharpes)}"
        )


def _run_folds(fit_predict: FitPredict, y: np.ndarray, folds: Folds, spec: TaskSpec):
    oof = np.zeros(len(y))
    sharpes = []
    for train_idx, test_idx in folds:
        pred = np.asarray(fit_predict(train_idx, test_idx), dtype=float)
        if pred.shape != test_idx.shape:
            raise ValueError(f"fit_predict returned shape {pred.shape}, expected {test_idx.shape}")
        oof[test_idx] = pred
        sharpes.append(score(pred, y[test_idx], spec))
    covered = np.zeros(len(y), dtype=bool)
    for _, test_idx in folds:
        covered[test_idx] = True
    return sharpes, score(oof[covered], y[covered], spec)


def cross_validate(
    fit_predict: FitPredict,
    y,
    folds: Folds | None = None,
    spec: TaskSpec = ARCADE_ALGO_ARENA,
) -> CVResult:
    y = np.asarray(y, dtype=float)
    folds = kfold(len(y)) if folds is None else folds
    sharpes, oof = _run_folds(fit_predict, y, folds, spec)
    return CVResult(fold_sharpes=tuple(sharpes), oof_sharpe=oof)


def repeated_kfold(
    fit_predict: FitPredict,
    y,
    k: int = 5,
    seeds: Sequence[int] = (0, 1, 2),
    spec: TaskSpec = ARCADE_ALGO_ARENA,
) -> CVResult:
    """k-fold repeated over several shuffles: the team's default yardstick."""
    y = np.asarray(y, dtype=float)
    all_sharpes, oofs = [], []
    for seed in seeds:
        sharpes, oof = _run_folds(fit_predict, y, kfold(len(y), k, seed), spec)
        all_sharpes += sharpes
        oofs.append(oof)
    return CVResult(fold_sharpes=tuple(all_sharpes), oof_sharpe=float(np.mean(oofs)))


def out_of_fold(fit_predict: FitPredict, n: int, folds: Folds, allow_partial: bool = False) -> np.ndarray:
    """Out-of-fold predictions for all n sessions (what you export as a team signal).

    Walk-forward folds never test the earliest rows: pass allow_partial=True to leave them NaN.
    """
    oof = np.full(n, np.nan)
    for train_idx, test_idx in folds:
        oof[test_idx] = np.asarray(fit_predict(train_idx, test_idx), dtype=float)
    if np.isnan(oof).any() and not allow_partial:
        raise ValueError("folds do not cover every session")
    return oof


def signal_fit_predict(signal) -> FitPredict:
    """Wrap a fixed, label-free signal (e.g. keyword score) so it fits the CV harness."""
    signal = np.asarray(signal, dtype=float)
    return lambda _train_idx, test_idx: signal[test_idx]


def walk_forward_by_time(times, k: int = 5, min_train_frac: float = 0.5, gap=0) -> Folds:
    """Expanding-window folds on time values (real, dated data).

    Test blocks are cut on unique sorted times, so rows sharing a timestamp never straddle a fold
    boundary, and rows may come in any order. Training rows within `gap` (a Timedelta for dates,
    a number otherwise) before a test block's first time are dropped, so overlapping return
    windows cannot leak. Rows without a time (NaT/NaN) are in no fold.
    """
    t = pd.Series(np.asarray(times)).reset_index(drop=True)
    uniq = np.sort(t.dropna().unique())
    start = int(len(uniq) * min_train_frac)
    edges = np.linspace(start, len(uniq), k + 1).astype(int)
    folds = []
    for a, b in zip(edges[:-1], edges[1:], strict=True):
        first, last = uniq[a], uniq[b - 1]
        test = np.flatnonzero(((t >= first) & (t <= last)).to_numpy())
        cutoff = first - gap if gap else first
        folds.append((np.flatnonzero((t < cutoff).to_numpy()), test))
    return folds
