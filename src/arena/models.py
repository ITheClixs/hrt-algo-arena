"""Ready-to-run models. Each has .fit(full_sessions, y) -> fitted with .predict(seen_sessions).

`full_sessions` may contain bars past the halfway point (in-sample research data); the fitted
model only ever looks at `seen_sessions` truncated to the halfway bar when predicting.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import Ridge
from sklearn.pipeline import Pipeline, make_pipeline
from sklearn.preprocessing import StandardScaler

from arena.config import TaskSpec
from arena.cv import FitPredict
from arena.data import Sessions, subset, truncate, with_column
from arena.events import fit_drift_rates, realized_drift, recent_count, remaining_drift
from arena.features import parkinson_vol, price_features
from arena.sizing import clip_sd, vol_scale
from arena.text import session_documents, templates

DEFAULT_PRICE_COLS = ("ret_all_z", "mom_10_z", "range_trend")


def fit_ridge(X, y, alpha: float = 1.0, sample_weight=None) -> Pipeline:
    model = make_pipeline(StandardScaler(), Ridge(alpha=alpha))
    model.fit(X, y, ridge__sample_weight=sample_weight)
    return model


def make_tfidf_ridge(alpha: float = 10.0, ngram_range=(1, 2), min_df: int = 3,
                     max_features: int | None = 200_000) -> Pipeline:
    """Text -> return regression. The EPFL (real Bloomberg headlines) winner used this shape."""
    return make_pipeline(
        TfidfVectorizer(ngram_range=ngram_range, min_df=min_df, max_features=max_features,
                        sublinear_tf=True, strip_accents="unicode"),
        Ridge(alpha=alpha),
    )


def ensure_keys(s: Sessions, key: str = "key") -> Sessions:
    """Add the template column if it is missing."""
    if key in s.headlines.columns:
        return s
    return with_column(s, key, templates(s.headlines["headline"]).to_numpy())


@dataclass(frozen=True)
class DriftModel:
    """Template drift still owed + price features -> ridge -> optional vol sizing + clipping."""

    spec: TaskSpec
    duration: int = 10
    alpha: float = 10.0
    price_cols: tuple[str, ...] = DEFAULT_PRICE_COLS
    vol_power: float = 0.0
    clip_n_sd: float = 3.0
    shrinkage: float = 0.0

    def design(self, seen: Sessions, rates: pd.Series) -> np.ndarray:
        h, H = self.spec.halfway_bar, self.spec.horizon
        seen = ensure_keys(seen)
        X = np.column_stack(
            [
                remaining_drift(seen, rates, h, H, duration=self.duration),
                np.log(seen.close[:, -1] / seen.open[:, 0])
                - realized_drift(seen, rates, h, duration=self.duration),
                recent_count(seen, h - self.duration + 1, h),
            ]
        )
        if self.price_cols:
            X = np.column_stack([X, price_features(seen)[list(self.price_cols)].to_numpy()])
        return X

    def fit(self, full: Sessions, y) -> FittedDrift:
        full = ensure_keys(full)
        rates = fit_drift_rates(full, duration=self.duration, shrinkage=self.shrinkage)
        seen = truncate(full, self.spec.halfway_bar)
        model = fit_ridge(self.design(seen, rates), np.asarray(y, dtype=float), self.alpha)
        fitted = FittedDrift(config=self, rates=rates, model=model)
        return fitted.with_reference(fitted.raw_predict(seen))


@dataclass(frozen=True)
class FittedDrift:
    config: DriftModel
    rates: pd.Series
    model: Pipeline
    reference: np.ndarray = field(default_factory=lambda: np.zeros(0))

    def with_reference(self, ref: np.ndarray) -> FittedDrift:
        return FittedDrift(config=self.config, rates=self.rates, model=self.model, reference=ref)

    def raw_predict(self, seen: Sessions) -> np.ndarray:
        seen = truncate(seen, self.config.spec.halfway_bar)
        pred = self.model.predict(self.config.design(seen, self.rates))
        if self.config.vol_power:
            pred = vol_scale(pred, parkinson_vol(seen.high, seen.low), self.config.vol_power)
        return pred

    def predict(self, seen: Sessions) -> np.ndarray:
        ref = self.reference if self.reference.size else None
        return clip_sd(self.raw_predict(seen), self.config.clip_n_sd, ref=ref)


@dataclass(frozen=True)
class TfidfModel:
    """Session documents (all visible headlines) -> TF-IDF -> ridge."""

    spec: TaskSpec
    alpha: float = 10.0
    first_bar: int = 0
    min_df: int = 3

    def fit(self, full: Sessions, y) -> FittedTfidf:
        seen = truncate(full, self.spec.halfway_bar)
        pipe = make_tfidf_ridge(alpha=self.alpha, min_df=self.min_df)
        pipe.fit(session_documents(seen, self.first_bar), np.asarray(y, dtype=float))
        return FittedTfidf(config=self, pipe=pipe)


@dataclass(frozen=True)
class FittedTfidf:
    config: TfidfModel
    pipe: Pipeline

    def predict(self, seen: Sessions) -> np.ndarray:
        seen = truncate(seen, self.config.spec.halfway_bar)
        return self.pipe.predict(session_documents(seen, self.config.first_bar))


def as_fit_predict(model, full: Sessions, y) -> FitPredict:
    """Adapt any `.fit(full, y).predict(seen)` model to the CV harness without leakage."""
    y = np.asarray(y, dtype=float)
    full = ensure_keys(full)

    def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        fitted = model.fit(subset(full, train_idx), y[train_idx])
        return fitted.predict(subset(full, test_idx))  # predict() truncates to halfway

    return fit_predict
