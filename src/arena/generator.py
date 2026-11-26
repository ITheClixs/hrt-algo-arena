"""Fit the synthetic generator directly: one push rate per headline type, estimated jointly.

Model of a session's per-bar log returns (market-adjusted, i.e. minus the cross-sectional mean
of that bar):

    r[i, t] = vol[i] * sum over headlines h of session i:  rate[type(h)] * kernel[t - bar(h)]  + noise

`kernel[k]` is the push on the return of bar `bar(h) + k` (k = 0 is the headline's own bar).
Public research on this data found 0.5, 1 x 9, 0.5 (`KERNEL_EDINBURGH`). Rates are fitted by
ridge least squares over every visible bar: labelled sessions (all bars) plus, label-free, the
visible bars of unlabeled sessions. The forecast is the push still owed in bars 50..59.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, replace

import numpy as np
import pandas as pd
import scipy.sparse as sp
from sklearn.linear_model import Ridge

from arena.augment import augmented, fitting_offsets
from arena.config import TaskSpec
from arena.data import Sessions, truncate
from arena.sizing import fit_risk_model, risk_features, sharpe_position
from arena.structure import keep_headlines
from arena.universe import main_only

KERNEL_EDINBURGH: tuple[float, ...] = (0.5,) + (1.0,) * 9 + (0.5,)
KERNEL_FLAT10: tuple[float, ...] = (0.0,) + (1.0,) * 10
LN2 = np.log(2.0)
VOL_FLOOR = 1e-5


def owed_weight(bar, kernel: Sequence[float], halfway: int, horizon: int) -> np.ndarray:
    """Kernel weight of a headline at `bar` that lands on the returns of bars halfway+1..halfway+horizon."""
    bar = np.asarray(bar)
    w = np.zeros(bar.shape, dtype=float)
    for k, wk in enumerate(kernel):
        t = bar + k
        w += wk * ((t > halfway) & (t <= halfway + horizon))
    return w


def done_weight(bar, kernel: Sequence[float], halfway: int) -> np.ndarray:
    """Kernel weight already realised in the returns of bars 1..halfway."""
    bar = np.asarray(bar)
    w = np.zeros(bar.shape, dtype=float)
    for k, wk in enumerate(kernel):
        t = bar + k
        w += wk * ((t >= 1) & (t <= halfway))
    return w


def gk_vol(s: Sessions, last_bar: int = 49) -> np.ndarray:
    """Garman-Klass per-bar volatility over bars 0..last_bar, floored."""
    sl = slice(0, last_bar + 1)
    o, h, lo, c = (np.log(a[:, sl]) for a in (s.open, s.high, s.low, s.close))
    var = (0.5 * (h - lo) ** 2 - (2 * LN2 - 1) * (c - o) ** 2).mean(axis=1)
    return np.sqrt(np.maximum(var, VOL_FLOOR**2))


def market_adjusted_returns(s: Sessions, last_bar: int) -> np.ndarray:
    """(n, last_bar) log returns of bars 1..last_bar minus each bar's cross-sectional mean."""
    r = np.diff(np.log(s.close[:, : last_bar + 1]), axis=1)
    return r - r.mean(axis=0, keepdims=True)


@dataclass(frozen=True)
class PushRates:
    rates: pd.Series  # per headline type; per-bar push per unit of kernel weight
    kernel: tuple[float, ...]
    key: str
    units: str = "vol"  # "vol": push = vol * rate * kernel; "raw": push = rate * kernel

    def lookup(self, keys: pd.Series) -> np.ndarray:
        return keys.map(self.rates).fillna(0.0).to_numpy(dtype=float)


def _design(s: Sessions, last_bar: int, key: str, kernel: Sequence[float], col_of: dict,
            units: str) -> tuple[sp.csr_matrix, np.ndarray]:
    """Rows = (session, bar 1..last_bar); X = kernel exposure per type, target = returns / vol."""
    vol = gk_vol(s, min(last_bar, 49))
    hl = s.headlines[s.headlines["bar_ix"] <= last_bar]
    sidx, bar = hl["sidx"].to_numpy(), hl["bar_ix"].to_numpy()
    cols = hl[key].map(col_of).to_numpy()
    rows, cc, vals = [], [], []
    for k, wk in enumerate(kernel):
        t = bar + k
        ok = (t >= 1) & (t <= last_bar) & (wk != 0)
        rows.append(sidx[ok] * last_bar + t[ok] - 1)
        cc.append(cols[ok])
        scale = 1.0 / vol[sidx[ok]] if units == "raw" else np.ones(ok.sum())
        vals.append(wk * scale)
    x = sp.csr_matrix((np.concatenate(vals), (np.concatenate(rows), np.concatenate(cc))),
                      shape=(s.n * last_bar, len(col_of)))
    target = market_adjusted_returns(s, last_bar) / vol[:, None]
    return x, target.ravel()


def fit_push_rates(parts: Sequence[tuple[Sessions, int]], key: str = "key",
                   kernel: Sequence[float] = KERNEL_EDINBURGH, lam: float = 200.0,
                   units: str = "vol") -> PushRates:
    """Joint ridge fit of one rate per type over (sessions, last usable bar) parts."""
    if units not in ("vol", "raw"):
        raise ValueError(f"units must be 'vol' or 'raw', got {units!r}")
    types = sorted(set().union(*(set(s.headlines[key].dropna()) for s, _ in parts)))
    col_of = {t: j for j, t in enumerate(types)}
    xtx, xty = np.zeros((len(types), len(types))), np.zeros(len(types))
    for s, last_bar in parts:
        x, target = _design(s, last_bar, key, kernel, col_of, units)
        xtx += (x.T @ x).toarray()
        xty += x.T @ target
    scale = 1.0 if units == "vol" else 1.0 / np.median(gk_vol(parts[0][0])) ** 2
    coef = np.linalg.solve(xtx + lam * scale * np.eye(len(types)), xty)
    return PushRates(rates=pd.Series(coef, index=types), kernel=tuple(kernel), key=key, units=units)


def _to_vol_units(values: np.ndarray, rates: PushRates, vol: np.ndarray) -> np.ndarray:
    return values / vol if rates.units == "raw" else values


def news_signal(s: Sessions, rates: PushRates, halfway: int, horizon: int) -> np.ndarray:
    """Push still owed over bars halfway+1..halfway+horizon, in units of session vol."""
    hl = s.headlines[s.headlines["bar_ix"] <= halfway]
    w = rates.lookup(hl[rates.key]) * owed_weight(hl["bar_ix"].to_numpy(), rates.kernel, halfway, horizon)
    owed = np.bincount(hl["sidx"].to_numpy(), weights=w, minlength=s.n)
    return _to_vol_units(owed, rates, gk_vol(s, halfway))


def residual_returns(s: Sessions, rates: PushRates, halfway: int) -> np.ndarray:
    """(n, halfway) vol-normalised returns of bars 1..halfway minus the push headlines explain."""
    vol = gk_vol(s, halfway)
    z = market_adjusted_returns(s, halfway) / vol[:, None]
    hl = s.headlines[s.headlines["bar_ix"] <= halfway]
    sidx, bar = hl["sidx"].to_numpy(), hl["bar_ix"].to_numpy()
    rate = rates.lookup(hl[rates.key])
    rate = rate / vol[sidx] if rates.units == "raw" else rate
    for k, wk in enumerate(rates.kernel):
        t = bar + k
        ok = (t >= 1) & (t <= halfway)
        np.subtract.at(z, (sidx[ok], t[ok] - 1), wk * rate[ok])
    return z


def decayed_sums(z: np.ndarray, half_lives: Sequence[float]) -> np.ndarray:
    """(n, len(half_lives)): sum_t z_t * exp(-age_t / h), age 0 at the last column."""
    age = np.arange(z.shape[1])[::-1]
    return np.column_stack([z @ np.exp(-age / h) for h in half_lives]) if half_lives else np.zeros((len(z), 0))


SIZINGS = ("second_moment", "vol", "risk_model")
EXTRA_FEATURES = ("news_count", "live_count", "conflict", "repeat_push", "peer_push", "surprise",
                  "outsider_push", "inv_vol")
PEER_COMPANY_COUNTS = (1, 2, 4)  # sessions with 3 main-industry companies are the reference


def extra_columns(visible: Sessions, rates: PushRates, names: Sequence[str], halfway: int,
                  horizon: int, outsiders: Sessions | None = None) -> np.ndarray:
    """Optional session features (each in vol units or counts); peer_push adds 3 columns."""
    hl, n = visible.headlines, visible.n
    sidx, bar = hl["sidx"].to_numpy(), hl["bar_ix"].to_numpy()
    vol = gk_vol(visible, halfway)
    rate = rates.lookup(hl[rates.key])
    rate = rate / vol[sidx] if rates.units == "raw" else rate
    owed = owed_weight(bar, rates.kernel, halfway, horizon)
    push = rate * owed

    def per_session(values: np.ndarray) -> np.ndarray:
        return np.bincount(sidx, weights=values, minlength=n)

    columns: list[np.ndarray] = []
    for name in names:
        if name == "news_count":
            columns.append(per_session(np.ones(len(bar))))
        elif name == "live_count":
            columns.append(per_session((owed > 0).astype(float)))
        elif name == "conflict":
            pos, neg = per_session(np.maximum(push, 0.0)), per_session(np.minimum(push, 0.0))
            columns.append((pos + neg) * ((pos > 0) & (neg < 0)))
        elif name == "surprise":
            columns.append(per_session(_surprise(visible, rate, bar, sidx, owed, rates.kernel, halfway, vol)))
        elif name == "inv_vol":  # 1 / visible vol: the low-vol drift deficit in bars 49-59
            columns.append(1.0 / vol)
        elif name == "outsider_push":
            columns.append(np.zeros(n) if outsiders is None else news_signal(outsiders, rates, halfway, horizon))
        elif name == "repeat_push":
            prev = pd.Series(bar).groupby([sidx, hl["entity"].to_numpy()]).shift()
            repeat = (bar - prev.fillna(-99).to_numpy()) <= 10
            columns.append(per_session(np.where(repeat, push, 0.0)))
        else:  # peer_push
            companies = hl.groupby("sidx")["entity"].nunique().reindex(range(n), fill_value=0).to_numpy()
            columns += [per_session(push * (companies[sidx] == k)) for k in PEER_COMPANY_COUNTS]
    return np.column_stack(columns) if columns else np.zeros((n, 0))


def _surprise(visible: Sessions, rate: np.ndarray, bar: np.ndarray, sidx: np.ndarray, owed: np.ndarray,
              kernel: Sequence[float], halfway: int, vol: np.ndarray) -> np.ndarray:
    """Per headline at bars halfway-9..halfway-1: owed x (realised move - expected done push) / done."""
    done = done_weight(bar, kernel, halfway)
    log_c = np.log(visible.close)
    before = np.where(bar > 0, log_c[sidx, np.maximum(bar - 1, 0)], np.log(visible.open[sidx, 0]))
    moved = (log_c[sidx, halfway] - before) / vol[sidx]
    recent = (bar >= halfway - 9) & (bar <= halfway - 1) & (np.abs(rate) > 0) & (done > 0)
    return np.where(recent, owed * (moved - done * rate) / np.maximum(done, 1e-12), 0.0)


def lag_profile(s: Sessions, key: str = "key", max_lag: int = 15, sign_lags: int = 11) -> np.ndarray:
    """Cumulative sign-adjusted mean log response at lags 0..max_lag after a headline.

    Label-free when `s` holds only visible bars. Each type's sign comes from its mean response
    over lags 0..sign_lags-1. The increments of the profile are the measured push kernel.
    """
    r = np.concatenate([np.log(s.close[:, :1] / s.open[:, :1]), np.diff(np.log(s.close), axis=1)], axis=1)
    hl = s.headlines
    sidx, bar = hl["sidx"].to_numpy(), hl["bar_ix"].to_numpy()
    resp = np.full((len(hl), max_lag + 1), np.nan)
    for lag in range(max_lag + 1):
        t = bar + lag
        ok = t < s.n_bars
        resp[ok, lag] = r[sidx[ok], t[ok]]
    total = pd.Series(np.nansum(resp[:, :sign_lags], axis=1))
    sign = np.sign(total.groupby(hl[key].to_numpy()).transform("mean").to_numpy())
    return np.cumsum(np.nanmean(resp * sign[:, None], axis=0))


def measured_kernel(profile: np.ndarray, length: int = 11) -> tuple[float, ...]:
    """Kernel from a lag profile: increments at lags 0..length-1, clipped at 0, scaled to sum 10."""
    inc = np.clip(np.diff(np.concatenate([[0.0], profile]))[:length], 0.0, None)
    return tuple(float(v) for v in inc * 10.0 / inc.sum())


@dataclass(frozen=True)
class FittedGenerator:
    config: GeneratorModel
    rates: PushRates
    ridge: Ridge
    mu: np.ndarray
    sd: np.ndarray
    var_y: float
    cap: float | None = None
    risk: object | None = None

    def raw_predict(self, seen: Sessions) -> np.ndarray:
        cfg = self.config
        visible = cfg.prepare(seen)
        x = cfg.features(visible, self.rates, cfg.outsiders(seen))
        m = self.ridge.predict((x - self.mu) / self.sd) - self.ridge.intercept_
        vol = gk_vol(visible, cfg.spec.halfway_bar)
        if cfg.sizing == "second_moment":
            return m / (vol * (self.var_y + cfg.m2_weight * m**2))
        if cfg.sizing == "risk_model":
            mu = m * vol
            return sharpe_position(mu, self.risk.predict(risk_features(visible, mu, cfg.spec.halfway_bar)))
        return m * vol ** (-cfg.vol_power)

    def predict(self, seen: Sessions) -> np.ndarray:
        p = self.raw_predict(seen)
        return p if self.cap is None else np.clip(p, -self.cap, self.cap)


@dataclass(frozen=True)
class GeneratorModel:
    """Joint push-rate fit -> [owed push, decayed residual momentum] -> ridge on r/vol -> sizing."""

    spec: TaskSpec
    key: str = "key"
    kernel: tuple[float, ...] = KERNEL_EDINBURGH
    lam: float = 200.0
    alpha: float = 50.0
    half_lives: tuple[float, ...] = (1, 2, 3, 5, 10, 25, 1000)
    use_unlabeled: bool = True
    extra_features: tuple[str, ...] = ()
    sizing: str = "second_moment"
    vol_power: float = 1.0
    clip_sd: float | None = None
    augment_offsets: tuple[int, ...] = ()
    units: str = "vol"
    main_industry: bool = True
    m2_weight: float = 1.0  # position = m / (vol * (var + m2_weight * m^2))
    residual_var: bool = False  # var = Var(y/vol - m) instead of Var(y/vol)
    var_target: bool = False  # sizing variance from the real target window only (not the augmented ones)
    target_weight: float = 1.0  # ridge weight of the real target window's rows (offset 0) vs augmented rows

    def __post_init__(self) -> None:
        if self.sizing not in SIZINGS:
            raise ValueError(f"sizing must be one of {SIZINGS}, got {self.sizing!r}")
        unknown = set(self.extra_features) - set(EXTRA_FEATURES)
        if unknown:
            raise ValueError(f"unknown extra_features {sorted(unknown)}; choose from {EXTRA_FEATURES}")

    def prepare(self, s: Sessions) -> Sessions:
        """The part of a session the model may look at: visible bars, main-industry headlines."""
        s = main_only(s) if self.main_industry else s
        return truncate(s, self.spec.halfway_bar)

    def outsiders(self, s: Sessions) -> Sessions | None:
        """Visible headlines from outside the main industry (only built when a feature needs them)."""
        if "outsider_push" not in self.extra_features:
            return None
        off = replace(s, headlines=s.headlines.assign(off_main=~s.headlines["in_main"].astype(bool)))
        return truncate(keep_headlines(off, "off_main"), self.spec.halfway_bar)

    def features(self, visible: Sessions, rates: PushRates, outsiders: Sessions | None = None) -> np.ndarray:
        h = self.spec.halfway_bar
        news = news_signal(visible, rates, h, self.spec.horizon)
        momentum = decayed_sums(residual_returns(visible, rates, h), self.half_lives)
        extra = extra_columns(visible, rates, self.extra_features, h, self.spec.horizon, outsiders)
        return np.column_stack([news, momentum, extra])

    def fit(self, full: Sessions, y, unlabeled: Sessions | None = None) -> FittedGenerator:
        h = self.spec.halfway_bar
        labelled = main_only(full) if self.main_industry else full
        parts = [(labelled, labelled.n_bars - 1)]
        if self.use_unlabeled and unlabeled is not None:
            parts.append((self.prepare(unlabeled), h))
        rates = fit_push_rates(parts, key=self.key, kernel=self.kernel, lam=self.lam, units=self.units)
        y0 = np.asarray(y, dtype=float)
        train, y, weight = full, y0, None
        if self.augment_offsets:
            train, y, _ = augmented(full, self.augment_offsets, self.spec)
            row_offset = np.repeat(fitting_offsets(full.n_bars, self.augment_offsets, self.spec), full.n)
            if self.target_weight != 1.0:  # None keeps the default fit bit-identical
                weight = np.where(row_offset == 0, self.target_weight, 1.0)
        visible = self.prepare(train)
        x = self.features(visible, rates, self.outsiders(train))
        vol = gk_vol(visible, h)
        target = y / vol
        mu, sd = x.mean(axis=0), x.std(axis=0) + 1e-12
        ridge = Ridge(alpha=self.alpha).fit((x - mu) / sd, target, sample_weight=weight)
        m = ridge.predict((x - mu) / sd) - ridge.intercept_
        risk = fit_risk_model(risk_features(visible, m * vol, h), y) if self.sizing == "risk_model" else None
        if self.var_target:  # the variance of the window that is actually traded, not of the augmented mix
            target = y0 / gk_vol(self.prepare(full), h)
            m = m[: full.n] if self.augment_offsets and row_offset[0] == 0 else m[:0]
        var_y = float((target - m).var()) if self.residual_var and len(m) == len(target) else float(target.var())
        fitted = FittedGenerator(config=self, rates=rates, ridge=ridge, mu=mu, sd=sd, var_y=var_y, risk=risk)
        if self.clip_sd is None:
            return fitted
        cap = self.clip_sd * float(fitted.raw_predict(full).std())
        return replace(fitted, cap=cap)


SECTOR_REF: dict = dict(kernel=KERNEL_EDINBURGH, lam=200.0, alpha=50.0,
                        half_lives=(1, 2, 3, 5, 10, 25, 1000), use_unlabeled=True,
                        sizing="second_moment", units="raw")
