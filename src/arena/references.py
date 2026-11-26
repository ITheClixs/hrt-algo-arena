"""R0 reference models: what already existed or was public before 25 Nov, re-implemented on
our own interface so every idea is compared on the same folds."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression

from arena.config import TaskSpec
from arena.data import Sessions, truncate
from arena.features import bar_log_returns
from arena.generator import owed_weight
from arena.models import DriftModel, FittedDrift
from arena.universe import main_only


@dataclass(frozen=True)
class FittedBaseline:
    inner: FittedDrift

    def predict(self, seen: Sessions) -> np.ndarray:
        return self.inner.predict(main_only(seen))


@dataclass(frozen=True)
class BaselineDrift:
    """R0(a): this repo's quickstart model (template drift + main industry, divided by vol)."""

    spec: TaskSpec

    def fit(self, full: Sessions, y, unlabeled: Sessions | None = None) -> FittedBaseline:
        return FittedBaseline(DriftModel(self.spec, vol_power=1.0).fit(main_only(full), y))


# --- R0(c): the 7 Nov winner's ideas (re-implemented from their write-up, not their code) ---

WINNER_KERNEL = (0.5,) + (1.0,) * 9 + (0.5,)  # push weight at lags 0..10
WINNER_LAGS = 9  # rates = mean per-bar log return over lags 1..9
PEER_COMPANY_COUNTS = (1, 2, 4)  # sessions with 3 main-industry companies are the reference


def _drift_observations(s: Sessions, key: str = "key", lags: int = WINNER_LAGS) -> pd.DataFrame:
    """Per headline: mean log return over lags 1..lags that fall inside the session's bars."""
    hl = s.headlines
    sidx, bar = hl["sidx"].to_numpy(), hl["bar_ix"].to_numpy()
    r = bar_log_returns(s.open, s.close)
    total, count = np.zeros(len(hl)), np.zeros(len(hl))
    for lag in range(1, lags + 1):
        t = bar + lag
        ok = t < s.n_bars
        total[ok] += r[sidx[ok], t[ok]]
        count[ok] += 1
    has = count > 0
    return pd.DataFrame({"key": hl[key].to_numpy()[has], "obs": total[has] / count[has]})


def _winner_rates(parts: list[Sessions]) -> pd.Series:
    obs = pd.concat([_drift_observations(p) for p in parts], ignore_index=True)
    return obs.groupby("key")["obs"].mean()


def _per_session(sidx: np.ndarray, values: np.ndarray, n: int) -> np.ndarray:
    return np.bincount(sidx, weights=values, minlength=n)


def _winner_features(visible: Sessions, rates: pd.Series, h: int, horizon: int) -> np.ndarray:
    """mu, surprise, unexplained, count, repeat, live, conflict and 3 peer pushes per session."""
    hl, n = visible.headlines, visible.n
    sidx, t = hl["sidx"].to_numpy(), hl["bar_ix"].to_numpy()
    r = hl["key"].map(rates).fillna(0.0).to_numpy()
    rem = owed_weight(t, WINNER_KERNEL, h, horizon)
    done = np.cumsum(WINNER_KERNEL)[np.clip(h - t, 0, len(WINNER_KERNEL) - 1)]
    log_c = np.log(visible.close)
    before = np.where(t > 0, log_c[sidx, np.maximum(t - 1, 0)], np.log(visible.open[sidx, 0]))
    obs = log_c[sidx, h] - before
    recent = (t >= h - 9) & (t <= h - 1) & (np.abs(r) > 1e-4)
    push = r * rem
    prev = pd.Series(t).groupby([sidx, hl["entity"].to_numpy()]).shift()
    repeat = (t - prev.fillna(-99).to_numpy()) <= 10
    pos = _per_session(sidx, np.where(r > 0, push, 0.0), n)
    neg = _per_session(sidx, np.where(r < 0, push, 0.0), n)
    companies = hl.groupby("sidx")["entity"].nunique().reindex(range(n), fill_value=0).to_numpy()
    cols = [
        pos + neg,
        _per_session(sidx, np.where(recent, rem * (obs - done * r) / done, 0.0), n),
        np.log(visible.close[:, h] / visible.open[:, 0]) - _per_session(sidx, r * done, n),
        _per_session(sidx, np.ones(len(t)), n),
        _per_session(sidx, np.where(repeat, push, 0.0), n),
        _per_session(sidx, (rem > 0).astype(float), n),
        (pos + neg) * ((pos > 0) & (neg < 0)),
    ]
    cols += [_per_session(sidx, push * (companies[sidx] == k), n) for k in PEER_COMPANY_COUNTS]
    return np.column_stack(cols)


def _risk_inputs(visible: Sessions, h: int) -> np.ndarray:
    """Garman-Klass variance (50 and 10 bars), squared gaps, close-close variance, news counts."""
    o, hi, lo, c = (np.log(a[:, : h + 1]) for a in (visible.open, visible.high, visible.low, visible.close))
    prev = np.concatenate([np.zeros((visible.n, 1)), c[:, :-1]], axis=1)
    gk = 0.5 * (hi - lo) ** 2 - (2 * np.log(2) - 1) * (c - o) ** 2
    hl = visible.headlines
    sidx = hl["sidx"].to_numpy()
    companies = hl.groupby("sidx")["entity"].nunique().reindex(range(visible.n), fill_value=0)
    return np.column_stack([
        gk.mean(axis=1), gk[:, -10:].mean(axis=1), ((o - prev) ** 2).mean(axis=1), ((c - prev) ** 2).mean(axis=1),
        _per_session(sidx, np.ones(len(hl)), visible.n),
        _per_session(sidx, (hl["bar_ix"].to_numpy() >= h - 9).astype(float), visible.n),
        companies.to_numpy(dtype=float),
    ])


@dataclass(frozen=True)
class FittedWinner:
    config: WinnerRef
    rates: pd.Series
    coef: np.ndarray
    centre: np.ndarray
    cap: float
    risk: LinearRegression
    v_floor: float
    v_mid: float

    def predict(self, seen: Sessions) -> np.ndarray:
        h, horizon = self.config.spec.halfway_bar, self.config.spec.horizon
        visible = truncate(main_only(seen), h)
        p = (_winner_features(visible, self.rates, h, horizon) - self.centre) @ self.coef
        v = np.maximum(self.risk.predict(_risk_inputs(visible, h)), self.v_floor)
        return np.clip(p, -self.cap, self.cap) * (self.v_mid / v) ** self.config.power


@dataclass(frozen=True)
class WinnerRef:
    """R0(c): drift + news counts + refined drift + peers + bar-based risk sizing (7 Nov winner)."""

    spec: TaskSpec
    clip_sd: float = 2.0
    power: float = 0.5
    use_unlabeled: bool = True

    def fit(self, full: Sessions, y, unlabeled: Sessions | None = None) -> FittedWinner:
        h, horizon = self.spec.halfway_bar, self.spec.horizon
        y = np.asarray(y, dtype=float)
        main = main_only(full)
        parts = [main] + ([main_only(unlabeled)] if self.use_unlabeled and unlabeled is not None else [])
        rates = _winner_rates(parts)
        visible = truncate(main, h)
        risk_x = _risk_inputs(visible, h)
        risk = LinearRegression().fit(risk_x, y**2)
        v = risk.predict(risk_x)
        v_floor = max(float(np.quantile(v, 0.05)), 1e-12)
        w = 1.0 / np.maximum(v, v_floor)
        x = _winner_features(visible, rates, h, horizon)
        centre = np.average(x, axis=0, weights=w)
        sw = np.sqrt(w)
        coef = np.linalg.lstsq((x - centre) * sw[:, None], y * sw, rcond=None)[0]
        cap = self.clip_sd * float(((x - centre) @ coef).std())
        return FittedWinner(config=self, rates=rates, coef=coef, centre=centre, cap=cap, risk=risk,
                            v_floor=v_floor, v_mid=float(np.median(v)))
