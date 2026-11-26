"""Practice tools: a mock leaderboard with an upload budget, and holdout splits.

Use it for timed drills on the public practice data: hide 30% of the labelled sessions,
give the team 10 uploads, and see whether CV numbers predict the "leaderboard".
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd

from arena.config import ARCADE_ALGO_ARENA, TaskSpec
from arena.metrics import score


def holdout_split(n: int, frac: float = 0.3, seed: int = 0) -> tuple[np.ndarray, np.ndarray]:
    """(train_idx, holdout_idx), both sorted."""
    if not 0 < frac < 1:
        raise ValueError("frac must be in (0, 1)")
    order = np.random.default_rng(seed).permutation(n)
    cut = int(round(n * (1 - frac)))
    return np.sort(order[:cut]), np.sort(order[cut:])


@dataclass(frozen=True)
class Upload:
    label: str
    public: float
    private: float | None


class MockLeaderboard:
    """Scores uploads against hidden returns. Optional public/private split like Kaggle."""

    def __init__(self, ids, returns, spec: TaskSpec = ARCADE_ALGO_ARENA, budget: int = 10,
                 private_frac: float = 0.0, seed: int = 0) -> None:
        self._ids = np.asarray(ids)
        self._returns = pd.Series(np.asarray(returns, dtype=float), index=self._ids)
        self._spec = spec
        self._budget = budget
        self._uploads: list[Upload] = []
        rng = np.random.default_rng(seed)
        self._private = rng.random(len(self._ids)) < private_frac

    @property
    def remaining(self) -> int:
        return self._budget - len(self._uploads)

    def _score(self, pred: pd.Series, mask: np.ndarray) -> float:
        return score(pred.to_numpy()[mask], self._returns.to_numpy()[mask], self._spec)

    def submit(self, ids, pred, label: str = "") -> float:
        """Spend one upload; returns the public score."""
        if self.remaining <= 0:
            raise RuntimeError("no uploads left")
        series = pd.Series(np.asarray(pred, dtype=float), index=np.asarray(ids))
        if series.index.duplicated().any() or set(series.index) != set(self._ids):
            raise ValueError("upload must contain every hidden session exactly once")
        series = series.reindex(self._ids)
        public = self._score(series, ~self._private)
        private = self._score(series, self._private) if self._private.any() else None
        self._uploads.append(Upload(label=label or f"upload {len(self._uploads) + 1}",
                                    public=public, private=private))
        return public

    def history(self, reveal_private: bool = False) -> pd.DataFrame:
        rows = [
            {"label": u.label, "public": u.public}
            | ({"private": u.private} if reveal_private else {})
            for u in self._uploads
        ]
        return pd.DataFrame(rows)
