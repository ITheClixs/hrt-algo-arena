"""Build an EPFL-like rehearsal dataset from public data (never commit the output).

Real Bloomberg headlines (Ding et al. 2014 corpus, HF `danidanou/Bloomberg_Financial_News`)
+ daily US prices (FNSPID, HF `Zihan1004/FNSPID`, Stock_price/full_history.zip):

    one row per (headline, ticker): time, entry_date, ticker, headline,
    past-only price features, and ret = next-day market-adjusted return.

Entry rule (no look-ahead): a headline published before 16:00 is traded at that day's close,
otherwise at the next trading day's close. ret = adjclose(entry+1)/adjclose(entry) - 1, minus the
same for SPY. Train = entries before 2013, test = 2013 (answers in a separate file), mirroring
EPFL's "2016 holdout". Bloomberg timestamps are taken as US Eastern time (an assumption).

    uv run python scripts/build_news_rehearsal.py --raw <dir with the downloads> --out <dir>
"""

from __future__ import annotations

import argparse
import re
import sys
from pathlib import Path

import numpy as np
import pandas as pd

TICKER = re.compile(r"\(([A-Z]{1,5})\)")
NOT_TICKERS = {
    "CEO", "CFO", "GDP", "EU", "IMF", "ECB", "US", "UK", "FDA", "SEC", "NYSE", "IPO", "OPEC", "UN",
    "NATO", "WTO", "AP", "PDF", "EPA", "FED", "FOMC", "CPI", "PMI", "ETF", "LLC", "PLC", "AG", "SA",
}
CUTOFF_HOUR = 16
TEST_START = pd.Timestamp("2013-01-01")


def log(msg: str) -> None:
    sys.stdout.write(msg + "\n")
    sys.stdout.flush()


def load_headlines(raw: Path) -> pd.DataFrame:
    b = pd.read_parquet(raw / "bloomberg_financial_data.parquet.gzip", columns=["Headline", "Date", "Article"])
    b = b.dropna(subset=["Headline", "Date"])
    first = b["Article"].fillna("").str.slice(0, 600)
    b["ticker"] = first.str.extract(TICKER)[0]
    b = b[b["ticker"].notna() & ~b["ticker"].isin(NOT_TICKERS)]
    b = b.rename(columns={"Headline": "headline", "Date": "time"})[["time", "ticker", "headline"]]
    b["headline"] = b["headline"].str.strip()
    b = b[b["headline"].str.len() > 10]
    return b.drop_duplicates(["ticker", "headline"]).reset_index(drop=True)


def load_prices(raw: Path, tickers) -> dict[str, pd.DataFrame]:
    out = {}
    for t in sorted(set(tickers) | {"SPY"}):
        path = raw / "full_history" / f"{t}.csv"
        if not path.exists():
            continue
        p = pd.read_csv(path, parse_dates=["date"]).sort_values("date").set_index("date")
        p = p[["adj close", "close", "volume"]].rename(columns={"adj close": "adj"})
        p = p[(p.index >= "2006-01-01") & (p.index <= "2014-01-31")]
        if len(p) > 30:
            out[t] = p
    return out


def price_features(p: pd.DataFrame) -> pd.DataFrame:
    r = p["adj"].pct_change()
    return pd.DataFrame(
        {
            "vol20": r.rolling(20, min_periods=15).std(),
            "ret1": r,
            "ret5": p["adj"].pct_change(5),
            "logdv20": np.log((p["close"] * p["volume"]).rolling(20, min_periods=15).mean() + 1.0),
            "price": p["close"],
            "fwd": p["adj"].shift(-1) / p["adj"] - 1.0,
        },
        index=p.index,
    )


def entry_dates(times: pd.Series, calendar: pd.DatetimeIndex) -> pd.Series:
    """First trading-day close after the headline: same day before 16:00, else the next one."""
    day = times.dt.normalize().to_numpy()
    pos = calendar.searchsorted(day, side="left")  # first trading day >= the headline's day
    same_day = calendar.to_numpy()[np.minimum(pos, len(calendar) - 1)] == day
    after_close = (times.dt.hour >= CUTOFF_HOUR).to_numpy()
    pos = np.where(same_day & after_close, pos + 1, pos)
    out = np.full(len(day), np.datetime64("NaT", "ns"), dtype="datetime64[ns]")
    ok = pos < len(calendar)
    out[ok] = calendar.to_numpy()[pos[ok]]
    return pd.Series(out, index=times.index)


def build(raw: Path) -> pd.DataFrame:
    news = load_headlines(raw)
    log(f"headlines with a ticker tag: {len(news):,} ({news['ticker'].nunique():,} tickers)")
    prices = load_prices(raw, news["ticker"].unique())
    log(f"tickers with US prices: {len(prices) - 1:,}")
    spy = price_features(prices.pop("SPY"))
    calendar = spy.index
    news = news[news["ticker"].isin(prices)].copy()
    news["entry_date"] = entry_dates(news["time"], calendar)
    news = news.dropna(subset=["entry_date"])
    feats = pd.concat({t: price_features(p) for t, p in prices.items()}, names=["ticker", "entry_date"])
    feats = feats.reset_index()
    news = news.merge(feats, on=["ticker", "entry_date"], how="inner")
    news = news.merge(spy[["fwd"]].rename(columns={"fwd": "spy_fwd"}), left_on="entry_date", right_index=True)
    news["ret"] = news["fwd"] - news["spy_fwd"]
    news = news.dropna(subset=["ret", "vol20", "logdv20"])
    news = news[(news["price"] >= 2.0) & (news["ret"].abs() < 0.5)]
    news = news.sort_values(["time", "ticker"]).reset_index(drop=True)
    news.insert(0, "id", np.arange(len(news), dtype=np.int64))
    cols = ["id", "time", "entry_date", "ticker", "headline", "vol20", "ret1", "ret5", "logdv20", "price", "ret"]
    return news[cols]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--raw", type=Path, default=Path("data/rehearsal/raw"))
    ap.add_argument("--out", type=Path, default=Path("data/rehearsal/news"))
    args = ap.parse_args()
    df = build(args.raw)
    args.out.mkdir(parents=True, exist_ok=True)
    train, test = df[df["entry_date"] < TEST_START], df[df["entry_date"] >= TEST_START]
    train.to_parquet(args.out / "news_train.parquet", index=False)
    test.drop(columns="ret").to_parquet(args.out / "news_test.parquet", index=False)
    test[["id", "ret"]].to_parquet(args.out / "news_test_answers.parquet", index=False)
    log(f"rows: {len(df):,}  train {len(train):,}  test {len(test):,}  tickers {df['ticker'].nunique():,}")
    log(f"per year: {df['entry_date'].dt.year.value_counts().sort_index().to_dict()}")
    log(f"ret sd {df['ret'].std():.4f}  mean {df['ret'].mean():.5f}  rows/day {df.groupby('entry_date').size().mean():.1f}")
    log(f"wrote {args.out}")


if __name__ == "__main__":
    main()
