"""The yardstick every idea is measured with.

* `evaluate` runs 5-fold CV for each development seed and pools each seed's out-of-fold
  predictions into one prediction per session. The CV score is the mean of the seed scores.
* `compare` decides whether a candidate beats the current best. Fold-level spreads would be
  wrong here: the 15 folds are three reshuffles of the same sessions, so they are strongly
  correlated. The uncertainty of a comparison therefore comes from a paired session bootstrap.
* Keep a change when the mean paired gain is at least KEEP_Z bootstrap standard errors and
  the candidate wins for every seed. Then confirm it on fresh targets with
  `arena.confirm.later_window_check` (held-out sessions' later windows, never selected on).
"""

from __future__ import annotations

import time
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from arena.config import ARCADE_ALGO_ARENA, TaskSpec
from arena.cv import FitPredict, kfold
from arena.metrics import EPS, clean, score

DEV_SEEDS: tuple[int, ...] = (42, 43, 44)
CONFIRM_SEEDS: tuple[int, ...] = (1042, 1043, 1044)
N_FOLDS = 5
KEEP_Z = 2.5  # was 1.5 until the 25 Nov review: ~30 trials were selected on the same dev targets
N_BOOT = 2000
_CHUNK = 200


@dataclass(frozen=True)
class Evaluation:
    name: str
    seeds: tuple[int, ...]
    seed_scores: tuple[float, ...]  # Sharpe of each seed's pooled out-of-fold predictions
    fold_scores: tuple[float, ...]  # len(seeds) * k single-fold Sharpes
    oof: np.ndarray  # (len(seeds), n) out-of-fold predictions
    runtime_s: float

    @property
    def mean(self) -> float:
        return float(np.mean(self.seed_scores))

    @property
    def worst_fold(self) -> float:
        return float(np.min(self.fold_scores))

    def __str__(self) -> str:
        seeds = " ".join(f"{s:.3f}" for s in self.seed_scores)
        return (f"{self.name}: CV {self.mean:.3f} | worst fold {self.worst_fold:.3f} "
                f"| seeds {seeds} | {self.runtime_s:.0f}s")


def evaluate(name: str, fit_predict: FitPredict, y, seeds: Sequence[int] = DEV_SEEDS,
             spec: TaskSpec = ARCADE_ALGO_ARENA, k: int = N_FOLDS) -> Evaluation:
    y = np.asarray(y, dtype=float)
    start = time.perf_counter()
    oof = np.full((len(seeds), len(y)), np.nan)
    seed_scores, fold_scores = [], []
    for row, seed in enumerate(seeds):
        for train_idx, test_idx in kfold(len(y), k, seed):
            pred = np.asarray(fit_predict(train_idx, test_idx), dtype=float)
            if pred.shape != test_idx.shape:
                raise ValueError(f"{name}: fit_predict returned {pred.shape}, expected {test_idx.shape}")
            oof[row, test_idx] = pred
            fold_scores.append(score(pred, y[test_idx], spec))
        seed_scores.append(score(oof[row], y, spec))
    return Evaluation(name=name, seeds=tuple(seeds), seed_scores=tuple(seed_scores),
                      fold_scores=tuple(fold_scores), oof=oof,
                      runtime_s=time.perf_counter() - start)


def _row_sharpes(pnl: np.ndarray, spec: TaskSpec) -> np.ndarray:
    return pnl.mean(axis=1) / (pnl.std(axis=1, ddof=spec.ddof) + EPS) * spec.annualization


def paired_bootstrap_se(pred_a, pred_b, y, spec: TaskSpec = ARCADE_ALGO_ARENA,
                        n_boot: int = N_BOOT, seed: int = 0) -> float:
    """Std over session resamples of Sharpe(a) - Sharpe(b); the same resample for both."""
    y = np.asarray(y, dtype=float)
    pnl_a, pnl_b = clean(pred_a) * y, clean(pred_b) * y
    rng = np.random.default_rng(seed)
    deltas = []
    for start in range(0, n_boot, _CHUNK):
        idx = rng.integers(0, len(y), size=(min(_CHUNK, n_boot - start), len(y)))
        deltas.append(_row_sharpes(pnl_a[idx], spec) - _row_sharpes(pnl_b[idx], spec))
    return float(np.std(np.concatenate(deltas), ddof=1))


@dataclass(frozen=True)
class Comparison:
    delta_by_seed: tuple[float, ...]
    delta: float  # mean over seeds of candidate minus best
    se: float  # paired session-bootstrap standard error

    @property
    def z(self) -> float:
        if self.se > 0:
            return self.delta / self.se
        return float(np.sign(self.delta) * np.inf) if self.delta else 0.0

    @property
    def keep(self) -> bool:
        return self.delta >= KEEP_Z * self.se and all(d > 0 for d in self.delta_by_seed)


def compare(candidate: Evaluation, best: Evaluation, y, spec: TaskSpec = ARCADE_ALGO_ARENA,
            n_boot: int = N_BOOT) -> Comparison:
    if candidate.seeds != best.seeds or candidate.oof.shape != best.oof.shape:
        raise ValueError(
            f"compare needs the same seeds and sessions: {candidate.name} {candidate.seeds} "
            f"{candidate.oof.shape} vs {best.name} {best.seeds} {best.oof.shape}")
    by_seed = tuple(float(a - b) for a, b in zip(candidate.seed_scores, best.seed_scores, strict=True))
    se = paired_bootstrap_se(candidate.oof[0], best.oof[0], y, spec, n_boot)
    return Comparison(delta_by_seed=by_seed, delta=float(np.mean(by_seed)), se=se)
