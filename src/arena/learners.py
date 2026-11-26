"""R4: learners that read the same headlines differently, for diversity in the blend.

* `OwedTfidfModel`: TF-IDF of each live headline's template, weighted by the push it still
  owes, summed per session, then ridge on r/vol. It shares strength across similar templates.
* `TemplateAgeMLP`: counts per (template, age) for headlines of the last 11 bars, plus log vol,
  then a small MLP on r/vol. It can learn the push shape and interactions without a kernel.

Both size positions like the generator model: E[r] / E[r^2] = m / (vol * (var + m^2)).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import scipy.sparse as sp
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.neural_network import MLPRegressor

from arena.augment import augmented
from arena.config import TaskSpec
from arena.data import Sessions, truncate
from arena.generator import KERNEL_EDINBURGH, gk_vol, owed_weight
from arena.universe import main_only


def _visible(s: Sessions, halfway: int) -> Sessions:
    return truncate(main_only(s), halfway)


def _training_set(full: Sessions, y, offsets: tuple[int, ...], spec: TaskSpec) -> tuple[Sessions, np.ndarray]:
    if not offsets:
        return full, np.asarray(y, dtype=float)
    windows, y_aug, _ = augmented(full, offsets, spec)
    return windows, y_aug


def _second_moment(m: np.ndarray, vol: np.ndarray, var: float) -> np.ndarray:
    return m / (vol * (var + m**2))


@dataclass(frozen=True)
class FittedOwedTfidf:
    config: OwedTfidfModel
    vectorizer: TfidfVectorizer
    ridge: Ridge
    var: float

    def predict(self, seen: Sessions) -> np.ndarray:
        visible = _visible(seen, self.config.spec.halfway_bar)
        x = self.config.design(visible, self.vectorizer)
        return _second_moment(self.ridge.predict(x), gk_vol(visible, self.config.spec.halfway_bar), self.var)


@dataclass(frozen=True)
class OwedTfidfModel:
    spec: TaskSpec
    alpha: float = 3000.0
    min_df: int = 5
    kernel: tuple[float, ...] = KERNEL_EDINBURGH
    augment_offsets: tuple[int, ...] = ()

    def _live(self, visible: Sessions):
        hl = visible.headlines
        w = owed_weight(hl["bar_ix"].to_numpy(), self.kernel, self.spec.halfway_bar, self.spec.horizon)
        keep = w > 0
        return hl.loc[keep, "sidx"].to_numpy(), w[keep], hl.loc[keep, "key"].astype(str).tolist()

    def design(self, visible: Sessions, vectorizer: TfidfVectorizer) -> sp.csr_matrix:
        sidx, w, texts = self._live(visible)
        weights = sp.csr_matrix((w, (sidx, np.arange(len(w)))), shape=(visible.n, len(w)))
        return (weights @ vectorizer.transform(texts)).tocsr()

    def fit(self, full: Sessions, y, unlabeled: Sessions | None = None) -> FittedOwedTfidf:
        train, y = _training_set(full, y, self.augment_offsets, self.spec)
        visible = _visible(train, self.spec.halfway_bar)
        _, _, texts = self._live(visible)
        vectorizer = TfidfVectorizer(ngram_range=(1, 2), min_df=self.min_df, sublinear_tf=True).fit(texts)
        target = y / gk_vol(visible, self.spec.halfway_bar)
        ridge = Ridge(alpha=self.alpha, fit_intercept=False).fit(self.design(visible, vectorizer), target)
        return FittedOwedTfidf(config=self, vectorizer=vectorizer, ridge=ridge, var=float(target.var()))


@dataclass(frozen=True)
class FittedTemplateAge:
    config: TemplateAgeMLP
    col_of: dict
    vol_centre: float
    vol_scale: float
    mlp: MLPRegressor
    var: float

    def predict(self, seen: Sessions) -> np.ndarray:
        visible = _visible(seen, self.config.spec.halfway_bar)
        x = self.config.design(visible, self.col_of, self.vol_centre, self.vol_scale)
        return _second_moment(self.mlp.predict(x), gk_vol(visible, self.config.spec.halfway_bar), self.var)


@dataclass(frozen=True)
class TemplateAgeMLP:
    spec: TaskSpec
    hidden: tuple[int, ...] = (64,)
    alpha: float = 1e-3
    max_iter: int = 50
    augment_offsets: tuple[int, ...] = tuple(range(0, 41, 2))
    seed: int = 0
    max_age: int = 10

    def design(self, visible: Sessions, col_of: dict, vol_centre: float, vol_scale: float) -> sp.csr_matrix:
        hl = visible.headlines
        age = self.spec.halfway_bar - hl["bar_ix"].to_numpy()
        col = hl["key"].map(col_of).to_numpy(dtype=float)
        keep = (age >= 0) & (age <= self.max_age) & ~np.isnan(col)
        cols = col[keep].astype(int) * (self.max_age + 1) + age[keep]
        counts = sp.csr_matrix((np.ones(keep.sum()), (hl["sidx"].to_numpy()[keep], cols)),
                               shape=(visible.n, len(col_of) * (self.max_age + 1)))
        log_vol = (np.log(gk_vol(visible, self.spec.halfway_bar)) - vol_centre) / vol_scale
        return sp.hstack([counts, sp.csr_matrix(log_vol[:, None])]).tocsr()

    def fit(self, full: Sessions, y, unlabeled: Sessions | None = None) -> FittedTemplateAge:
        train, y = _training_set(full, y, self.augment_offsets, self.spec)
        visible = _visible(train, self.spec.halfway_bar)
        col_of = {k: j for j, k in enumerate(sorted(visible.headlines["key"].unique()))}
        log_vol = np.log(gk_vol(visible, self.spec.halfway_bar))
        centre, scale = float(log_vol.mean()), float(log_vol.std() + 1e-12)
        target = y / gk_vol(visible, self.spec.halfway_bar)
        mlp = MLPRegressor(hidden_layer_sizes=self.hidden, alpha=self.alpha, max_iter=self.max_iter,
                           early_stopping=True, random_state=self.seed)
        mlp.fit(self.design(visible, col_of, centre, scale), target)
        return FittedTemplateAge(config=self, col_of=col_of, vol_centre=centre, vol_scale=scale, mlp=mlp,
                                 var=float(target.var()))
