"""R7: a nonlinear calibrator on the structural features (never tried: public GBM tests used raw features).

Same push-rate fit, augmentation, sizing and clip as the primary; the calibration ridge on
[owed push, 7 decayed residual sums] is replaced by gradient boosting on those features plus vol, 1/vol and the
visible headline count, so saturation and vol interactions can show up. One pre-declared configuration.

    uv run python research/r7_gbm_calibrator.py
"""

from __future__ import annotations

from dataclasses import dataclass, replace

import numpy as np
from common import SPEC, run_trial
from sklearn.ensemble import HistGradientBoostingRegressor

from arena.augment import augmented
from arena.generator import FittedGenerator, GeneratorModel, fit_push_rates, gk_vol
from arena.pipeline import PRIMARY
from arena.universe import main_only


class _Centred:
    """GBM with an `intercept_`, so FittedGenerator.raw_predict centres it like the ridge."""

    def __init__(self, model, intercept):
        self.model, self.intercept_ = model, intercept

    def predict(self, x):
        return self.model.predict(x)


@dataclass(frozen=True)
class GBMGenerator(GeneratorModel):
    def features(self, visible, rates, outsiders=None):
        x = super().features(visible, rates, outsiders)
        vol = gk_vol(visible, self.spec.halfway_bar)
        count = np.bincount(visible.headlines["sidx"], minlength=visible.n).astype(float)
        return np.column_stack([x, vol, 1.0 / vol, count])

    def fit(self, full, y, unlabeled=None):
        h = self.spec.halfway_bar
        parts = [(main_only(full), full.n_bars - 1)]
        if self.use_unlabeled and unlabeled is not None:
            parts.append((self.prepare(unlabeled), h))
        rates = fit_push_rates(parts, key=self.key, kernel=self.kernel, lam=self.lam, units=self.units)
        train, ya, _ = augmented(full, self.augment_offsets, self.spec)
        visible = self.prepare(train)
        x = self.features(visible, rates)
        target = ya / gk_vol(visible, h)
        gbm = HistGradientBoostingRegressor(max_iter=200, learning_rate=0.05, max_leaf_nodes=15,
                                            min_samples_leaf=400, l2_regularization=1.0, random_state=0)
        gbm.fit(x, target)
        model = _Centred(gbm, float(gbm.predict(x).mean()))
        mu, sd = np.zeros(x.shape[1]), np.ones(x.shape[1])  # GBM needs no scaling
        fitted = FittedGenerator(config=self, rates=rates, ridge=model, mu=mu, sd=sd, var_y=float(target.var()))
        cap = self.clip_sd * float(fitted.raw_predict(full).std())
        return replace(fitted, cap=cap)


def main() -> None:
    run_trial("r7_gbm_calibrator", lambda: GBMGenerator(SPEC, **PRIMARY), {**PRIMARY, "calibrator": "hgb200"},
              vs="r3_bundle_aug_clip2")


if __name__ == "__main__":
    main()
