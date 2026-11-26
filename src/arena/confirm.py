"""Fresh-target confirmation: score fold models on the held-out sessions' LATER windows.

The development folds score every idea on bars 49 -> 59 of the held-out sessions, and about 30
ideas were selected on exactly those targets, so a 1.5-s.e. dev-fold gain can be selection
noise. A held-out session's later bars were never used to select anything: refit each model on
the training fold, show it bars offset .. offset+49 of the held-out sessions and score
close(offset+49) -> close(offset+59). A real gain should show up here too, unless it is specific
to bars 49 -> 59 (then it needs its own mechanism check, e.g. on the 1,000 datathon sessions).

Caveat: news density falls in late bars, so windows at offset >= 30 are less like the real task.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np

from arena.augment import shift_window
from arena.config import TaskSpec
from arena.cv import kfold
from arena.data import concat_sessions, disjoint_from, forward_returns, subset, truncate
from arena.modelapi import ArcadeModel
from arena.universe import Universe
from arena.validation import N_FOLDS

LATER_OFFSETS: tuple[int, ...] = (10, 20, 30, 40)
LATER_SEEDS: tuple[int, ...] = (42, 43)


@dataclass(frozen=True)
class LaterWindows:
    offsets: tuple[int, ...]
    delta_by_seed: tuple[float, ...]  # Sharpe(candidate) - Sharpe(base), all windows pooled
    delta_by_offset: tuple[float, ...]  # seed-mean delta per offset
    se: float  # paired session-bootstrap s.e. of the pooled delta (first seed)

    @property
    def delta(self) -> float:
        return float(np.mean(self.delta_by_seed))

    @property
    def z(self) -> float:
        return self.delta / self.se if self.se > 0 else 0.0

    def __str__(self) -> str:
        per = " ".join(f"{o}:{d:+.3f}" for o, d in zip(self.offsets, self.delta_by_offset, strict=True))
        return f"later windows Δ {self.delta:+.4f} (se {self.se:.4f}, z {self.z:.2f}) | per offset {per}"


def _sharpe(pnl: np.ndarray, spec: TaskSpec) -> float:
    return float(pnl.mean() / (pnl.std(ddof=spec.ddof) + 1e-12) * spec.annualization)


def _window_pnl(models: Sequence[ArcadeModel], u: Universe, spec: TaskSpec, seed: int, k: int,
                offsets: tuple[int, ...], ys: dict[int, np.ndarray]) -> list[np.ndarray]:
    """(len(offsets), n) PnL per model: fold models on the held-out sessions' later windows."""
    h, n_visible = spec.halfway_bar, spec.halfway_bar + 1
    pred = [np.zeros((len(offsets), u.ins.n)) for _ in models]
    oos_seen = disjoint_from(truncate(u.oos, h), u.ins)
    for tr, te in kfold(u.ins.n, k, seed):
        test = subset(u.ins, te)
        pool = concat_sessions(truncate(test, h), oos_seen)
        windows = [shift_window(test, o, n_visible, spec.horizon) for o in offsets]
        for m, model in enumerate(models):
            fitted = model.fit(subset(u.ins, tr), u.y[tr], unlabeled=pool)
            for j, w in enumerate(windows):
                pred[m][j, te] = fitted.predict(w)
    return [p * np.stack([ys[o] for o in offsets]) for p in pred]


def later_window_check(candidate: ArcadeModel, base: ArcadeModel, u: Universe, spec: TaskSpec,
                       seeds: Sequence[int] = LATER_SEEDS, k: int = N_FOLDS,
                       offsets: Sequence[int] = LATER_OFFSETS, n_boot: int = 1000) -> LaterWindows:
    """Paired comparison of two models on targets no idea was ever selected on."""
    n_visible = spec.halfway_bar + 1
    fit = tuple(o for o in offsets if o + n_visible + spec.horizon <= u.ins.n_bars)
    if not fit:
        raise ValueError(f"no offset in {tuple(offsets)} fits {u.ins.n_bars} bars")
    ys = {o: forward_returns(shift_window(u.ins, o, n_visible, spec.horizon), spec) for o in fit}
    by_seed, by_offset, first = [], [], None
    for seed in seeds:
        a, b = _window_pnl((candidate, base), u, spec, seed, k, fit, ys)
        first = (a, b) if first is None else first
        by_seed.append(_sharpe(a.ravel(), spec) - _sharpe(b.ravel(), spec))
        by_offset.append([_sharpe(a[j], spec) - _sharpe(b[j], spec) for j in range(len(fit))])
    a, b = first
    rng = np.random.default_rng(0)
    boot = []
    for _ in range(n_boot):
        idx = rng.integers(0, u.ins.n, u.ins.n)
        boot.append(_sharpe(a[:, idx].ravel(), spec) - _sharpe(b[:, idx].ravel(), spec))
    return LaterWindows(offsets=fit, delta_by_seed=tuple(by_seed),
                        delta_by_offset=tuple(float(v) for v in np.mean(by_offset, axis=0)),
                        se=float(np.std(boot, ddof=1)))
