"""R5: fixed-weight blend of fitted members.

Each member's predictions are divided by their own standard deviation over the sessions being
predicted (label-free), so the weights compare like with like. Weights come from a short grid
declared in advance and are kept only if the blend passes `arena.validation.compare`.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from arena.data import Sessions
from arena.modelapi import ArcadeModel, Fitted


@dataclass(frozen=True)
class FittedBlend:
    members: tuple[Fitted, ...]
    weights: tuple[float, ...]

    def predict(self, seen: Sessions) -> np.ndarray:
        total = np.zeros(seen.n)
        for member, weight in zip(self.members, self.weights, strict=True):
            p = np.asarray(member.predict(seen), dtype=float)
            sd = p.std()
            total += weight * (p / sd if sd > 0 else p)
        return total


@dataclass(frozen=True)
class BlendModel:
    members: tuple[ArcadeModel, ...]
    weights: tuple[float, ...]

    def __post_init__(self) -> None:
        if len(self.members) != len(self.weights) or not self.members:
            raise ValueError(f"{len(self.members)} members but {len(self.weights)} weights")

    def fit(self, full: Sessions, y, unlabeled: Sessions | None = None) -> FittedBlend:
        fitted = tuple(m.fit(full, y, unlabeled=unlabeled) for m in self.members)
        return FittedBlend(members=fitted, weights=tuple(float(w) for w in self.weights))
