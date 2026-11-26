"""Synthetic market in HRT's format, for tests and offline drills (no internet needed).

Mechanism mirrors what the public Algo Arena data shows: each event template pushes the
log price at a constant per-bar rate for `drift_len` bars after it is published.
"""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np
import pandas as pd

from arena.config import ARCADE_ALGO_ARENA, TaskSpec

COMPANIES = (
    "Alvex Systems", "Borin Energy", "Corvel Labs", "Dravik Holdings", "Elnor Devices",
    "Fennic Foods", "Gorvan Capital", "Hestra Retail Group", "Ivrel Asset Partners",
    "Jorvik Software", "Kelmar Power", "Lunex Genomics",
)
REGIONS = ("Europe", "Asia Pacific", "Latin America", "North America")
EVENTS = (  # (template with slots, direction)
    ("secures ${amt}M contract with a global retailer", 1.0),
    ("reports record quarterly revenue, up {pct}% year-over-year", 1.2),
    ("announces ${amt}M share buyback program", 0.8),
    ("announces breakthrough in digital payments", 1.5),
    ("expands operations into {region} market", 0.6),
    ("misses quarterly revenue estimates by {pct}%", -1.2),
    ("faces regulatory review of cloud infrastructure practices", -1.4),
    ("recalls products in data analytics line due to quality concerns", -1.5),
    ("loses key contract in {region} to competitor", -0.9),
    ("opens new office in {region}", 0.0),
    ("files routine patent applications", 0.0),
    ("to host investor day focused on strategy", 0.1),
)


@dataclass(frozen=True)
class SyntheticMarket:
    bars: pd.DataFrame  # session, bar_ix, open, high, low, close
    headlines: pd.DataFrame  # session, bar_ix, headline, direction


def _headline(rng: np.random.Generator) -> tuple[str, float]:
    template, direction = EVENTS[rng.integers(len(EVENTS))]
    text = template.format(amt=int(rng.integers(5, 90)) * 10, pct=int(rng.integers(2, 40)),
                           region=REGIONS[rng.integers(len(REGIONS))])
    return f"{COMPANIES[rng.integers(len(COMPANIES))]} {text}", direction


def _session(rng, sid, n_bars, drift_len, rate, vol, intensity):
    drift = np.zeros(n_bars)
    rows = []
    for bar in range(n_bars):
        for _ in range(rng.poisson(intensity)):
            text, direction = _headline(rng)
            drift[bar + 1 : bar + 1 + drift_len] += direction * rate
            rows.append((sid, bar, text, direction))
    sigma = vol * rng.lognormal(0.0, 0.3)
    return _bars(rng, sid, n_bars, drift, sigma), rows


def _industry_session(rng, sid, n_bars, drift_len, rate, vol, intensity, industries, vol_scaled):
    """Session built around one industry: 80% of headlines are about it, and only those push."""
    main = int(rng.integers(len(industries)))
    sigma = vol * rng.lognormal(0.0, 0.3)
    scale = sigma / vol if vol_scaled else 1.0
    drift = np.zeros(n_bars)
    rows = []
    for bar in range(n_bars):
        for _ in range(rng.poisson(intensity)):
            group = main if rng.random() < 0.8 else int(rng.integers(len(industries)))
            company = industries[group][rng.integers(len(industries[group]))]
            template, direction = EVENTS[rng.integers(len(EVENTS))]
            text = template.format(amt=int(rng.integers(5, 90)) * 10, pct=int(rng.integers(2, 40)),
                                   region=REGIONS[rng.integers(len(REGIONS))])
            effect = direction if group == main else 0.0
            drift[bar + 1 : bar + 1 + drift_len] += effect * rate * scale
            rows.append((sid, bar, f"{company} {text}", effect))
    return _bars(rng, sid, n_bars, drift, sigma), rows


def _bars(rng, sid, n_bars, drift, sigma):
    log_close = np.cumsum(drift + rng.normal(0.0, sigma, n_bars))
    close = np.exp(log_close)
    open_ = np.concatenate([[1.0], close[:-1]])
    wick = np.abs(rng.normal(0.0, sigma * 0.5, (2, n_bars)))
    high = np.maximum(open_, close) * np.exp(wick[0])
    low = np.minimum(open_, close) * np.exp(-wick[1])
    return pd.DataFrame({"session": sid, "bar_ix": np.arange(n_bars), "open": open_,
                         "high": high, "low": low, "close": close})


def industry_companies(n_industries: int, per_industry: int = 16) -> list[list[str]]:
    """Programmatic company names; the first word names the industry ('Indus2 Corp07')."""
    return [[f"Indus{g} Corp{j:02d}" for j in range(per_industry)] for g in range(n_industries)]


def make_synthetic(n_sessions: int = 400, n_bars: int = 100, seed: int = 0, drift_len: int = 10,
                   rate: float = 2e-4, vol: float = 1e-3, intensity: float = 0.17,
                   start_id: int = 1000, n_industries: int = 0,
                   vol_scaled: bool = False) -> SyntheticMarket:
    """Synthetic sessions. n_industries > 0 adds the arcade data's industry structure (each session
    built around one industry whose headlines alone push); vol_scaled makes the push ∝ volatility."""
    rng = np.random.default_rng(seed)
    industries = industry_companies(n_industries) if n_industries > 0 else None
    bar_frames, rows = [], []
    for sid in range(start_id, start_id + n_sessions):
        if industries is None:
            bars, hl_rows = _session(rng, sid, n_bars, drift_len, rate, vol, intensity)
        else:
            bars, hl_rows = _industry_session(rng, sid, n_bars, drift_len, rate, vol, intensity,
                                              industries, vol_scaled)
        bar_frames.append(bars)
        rows += hl_rows
    headlines = pd.DataFrame(rows, columns=["session", "bar_ix", "headline", "direction"])
    return SyntheticMarket(bars=pd.concat(bar_frames, ignore_index=True), headlines=headlines)


def returns_frame(bars: pd.DataFrame, spec: TaskSpec = ARCADE_ALGO_ARENA) -> pd.DataFrame:
    close = bars.pivot(index="session", columns="bar_ix", values="close")
    r = close[spec.exit_bar] / close[spec.halfway_bar] - 1.0
    r = r - r.mean() if spec.demean_returns else r
    return pd.DataFrame({"session": close.index.to_numpy(), "returns": r.to_numpy()})


def write_arcade_files(market: SyntheticMarket, out_dir: str | Path, oos_frac: float = 0.5,
                       spec: TaskSpec = ARCADE_ALGO_ARENA) -> Path:
    """Write ins_*/oos_* parquet files exactly like the arcade challenge ships them."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    ids = np.sort(market.bars["session"].unique())
    oos_ids = set(ids[int(len(ids) * (1 - oos_frac)) :].tolist())
    is_oos_bar = market.bars["session"].isin(oos_ids)
    is_oos_hl = market.headlines["session"].isin(oos_ids)
    hl = market.headlines[["session", "bar_ix", "headline"]]
    ins_bars = market.bars[~is_oos_bar]
    ins_bars.to_parquet(out / "ins_bars.parquet", index=False)
    hl[~is_oos_hl].to_parquet(out / "ins_headlines.parquet", index=False)
    returns_frame(ins_bars, spec).to_parquet(out / "ins_returns.parquet", index=False)
    last = spec.halfway_bar
    market.bars[is_oos_bar & (market.bars["bar_ix"] <= last)].to_parquet(
        out / "oos_bars.parquet", index=False)
    hl[is_oos_hl & (hl["bar_ix"] <= last)].to_parquet(out / "oos_headlines.parquet", index=False)
    return out


def make_dated_headlines(n_days: int = 400, per_day: int = 30, seed: int = 0,
                         start: str = "2010-01-04") -> pd.DataFrame:
    """Real-headline-style rehearsal table (EPFL format): one row per headline with its date,
    ticker, text and the return that follows. target = direction * 1% + noise (2%)."""
    rng = np.random.default_rng(seed)
    days = pd.bdate_range(start, periods=n_days)
    rows = []
    for day in days:
        for _ in range(per_day):
            text, direction = _headline(rng)
            ticker = text.split()[0][:4].upper()
            rows.append((day, ticker, text, direction * 0.01 + rng.normal(0.0, 0.02)))
    return pd.DataFrame(rows, columns=["time", "ticker", "text", "target"])
