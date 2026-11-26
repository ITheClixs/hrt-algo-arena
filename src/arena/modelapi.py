"""The interface every arcade model implements, and the CV adapter that feeds it.

`fit(full, y, unlabeled)` gets labelled training sessions with all their bars, plus optional
unlabeled sessions that carry only their visible bars (label-free information such as the
out-of-sample first halves). `predict(seen)` must only use bars <= halfway of `seen`.
"""

from __future__ import annotations

from typing import Protocol

import numpy as np

from arena.cv import FitPredict
from arena.data import Sessions, concat_sessions, disjoint_from, subset, truncate


class Fitted(Protocol):
    def predict(self, seen: Sessions) -> np.ndarray: ...


class ArcadeModel(Protocol):
    def fit(self, full: Sessions, y: np.ndarray, unlabeled: Sessions | None = None) -> Fitted: ...


def as_fit_predict_with_unlabeled(model: ArcadeModel, full: Sessions, y,
                                  unlabeled: Sessions | None = None,
                                  halfway: int = 49) -> FitPredict:
    """CV adapter: each fold fits on its training sessions and may also use, label-free, the
    test fold's visible bars and `unlabeled`, exactly like the out-of-sample first halves are
    available when the real model is fitted."""
    y = np.asarray(y, dtype=float)
    # label-free sessions whose ids collide with the labelled ones are renumbered for the pool
    unlabeled_seen = None if unlabeled is None else disjoint_from(truncate(unlabeled, halfway), full)

    def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        test = subset(full, test_idx)
        visible = truncate(test, halfway)
        pool = visible if unlabeled_seen is None else concat_sessions(visible, unlabeled_seen)
        fitted = model.fit(subset(full, train_idx), y[train_idx], unlabeled=pool)
        return np.asarray(fitted.predict(test), dtype=float)

    return fit_predict
