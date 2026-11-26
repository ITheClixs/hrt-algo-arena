"""Load HRT-style session data into dense arrays.

HRT ships long-format parquet files:
  bars:      session, bar_ix, open, high, low, close   (prices start at 1.0)
  headlines: session, bar_ix, headline
  returns:   session, returns                          (arcade variant only)

`Sessions` holds one row per session as (n_sessions, n_bars) numpy arrays plus a long
headline table with a `sidx` column (row number into the arrays). Everything is immutable:
helpers return new objects.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq

from arena.config import ARCADE_ALGO_ARENA, TaskSpec

PRICE_COLS = ("open", "high", "low", "close")
BAR_COLS = ("session", "bar_ix", *PRICE_COLS)
HEADLINE_COLS = ("session", "bar_ix", "headline")


@dataclass(frozen=True)
class Sessions:
    ids: np.ndarray  # (n,) session ids, sorted
    open: np.ndarray  # (n, n_bars)
    high: np.ndarray
    low: np.ndarray
    close: np.ndarray
    headlines: pd.DataFrame  # session, sidx, bar_ix, headline (+ any extra columns), sorted

    @property
    def n(self) -> int:
        return len(self.ids)

    @property
    def n_bars(self) -> int:
        return self.close.shape[1]

    def log_close(self) -> np.ndarray:
        return np.log(self.close)


def _require(df: pd.DataFrame, cols: tuple[str, ...], what: str) -> None:
    missing = [c for c in cols if c not in df.columns]
    if missing:
        raise ValueError(f"{what} is missing columns {missing}; got {list(df.columns)}")


def _empty_headlines() -> pd.DataFrame:
    return pd.DataFrame(
        {
            "session": pd.Series(dtype="int64"),
            "sidx": pd.Series(dtype="int64"),
            "bar_ix": pd.Series(dtype="int64"),
            "headline": pd.Series(dtype="object"),
        }
    )


def _attach_sidx(headlines: pd.DataFrame | None, ids: np.ndarray) -> pd.DataFrame:
    if headlines is None or len(headlines) == 0:
        return _empty_headlines()
    _require(headlines, HEADLINE_COLS, "headlines")
    pos = pd.Series(np.arange(len(ids)), index=ids)
    hl = headlines.loc[headlines["session"].isin(ids)].copy()
    hl.insert(1, "sidx", pos.reindex(hl["session"]).to_numpy())
    return hl.sort_values(["sidx", "bar_ix"], kind="stable").reset_index(drop=True)


def from_long(bars: pd.DataFrame, headlines: pd.DataFrame | None = None) -> Sessions:
    """Build `Sessions` from HRT's long-format bars/headlines tables."""
    _require(bars, BAR_COLS, "bars")
    ids = np.sort(bars["session"].unique())
    arrays = {}
    for col in PRICE_COLS:
        wide = bars.pivot(index="session", columns="bar_ix", values=col).reindex(ids)
        if wide.isna().to_numpy().any():
            raise ValueError(f"bars are ragged: some sessions are missing bars for {col!r}")
        arrays[col] = wide.sort_index(axis=1).to_numpy(dtype=float)
    return Sessions(ids=ids, headlines=_attach_sidx(headlines, ids), **arrays)


def with_column(s: Sessions, name: str, values) -> Sessions:
    """Return a copy whose headline table has an extra column (aligned to row order)."""
    values = np.asarray(values)
    if len(values) != len(s.headlines):
        raise ValueError(f"{name}: got {len(values)} values for {len(s.headlines)} headlines")
    return replace(s, headlines=s.headlines.assign(**{name: values}))


def truncate(s: Sessions, last_bar: int) -> Sessions:
    """What you may see at the close of `last_bar` (inclusive). Use this to avoid leakage."""
    if not 0 <= last_bar < s.n_bars:
        raise ValueError(f"last_bar={last_bar} outside 0..{s.n_bars - 1}")
    k = last_bar + 1
    hl = s.headlines[s.headlines["bar_ix"] <= last_bar].reset_index(drop=True)
    return replace(
        s, open=s.open[:, :k], high=s.high[:, :k], low=s.low[:, :k], close=s.close[:, :k],
        headlines=hl,
    )


def subset(s: Sessions, idx) -> Sessions:
    """Rows `idx` (positions or boolean mask) as a new `Sessions` with re-numbered sidx."""
    idx = np.asarray(idx)
    if idx.dtype == bool:
        idx = np.flatnonzero(idx)
    if len(np.unique(idx)) != len(idx):
        raise ValueError("subset() needs unique indices; bootstrap on predictions instead")
    remap = pd.Series(np.arange(len(idx)), index=idx)
    hl = s.headlines[s.headlines["sidx"].isin(idx)].copy()
    hl["sidx"] = remap.reindex(hl["sidx"]).to_numpy()
    hl = hl.sort_values(["sidx", "bar_ix"], kind="stable").reset_index(drop=True)
    return Sessions(
        ids=s.ids[idx], open=s.open[idx], high=s.high[idx], low=s.low[idx],
        close=s.close[idx], headlines=hl,
    )


def renumbered(s: Sessions, start: int) -> Sessions:
    """Copy with session ids start, start+1, ... (for label-free or extra sessions whose ids
    collide with another part; the ids of these sessions never reach an upload)."""
    ids = np.arange(start, start + s.n, dtype=np.int64)
    headlines = s.headlines.assign(session=ids[s.headlines["sidx"].to_numpy()])
    return replace(s, ids=ids, headlines=headlines)


def disjoint_from(s: Sessions, *others: Sessions) -> Sessions:
    """`s` unchanged if its ids are free, else renumbered after the largest id in `others`."""
    taken = np.concatenate([o.ids for o in others])
    if not np.intersect1d(s.ids, taken).size:
        return s
    return renumbered(s, int(max(taken.max(), s.ids.max())) + 1)


def concat_sessions(*parts: Sessions) -> Sessions:
    """Stack sessions from several sources into one `Sessions` (parts keep their order)."""
    if not parts:
        raise ValueError("concat_sessions needs at least one part")
    n_bars = {p.n_bars for p in parts}
    if len(n_bars) > 1:
        raise ValueError(f"every part needs the same number of bars; got {sorted(n_bars)}")
    ids = np.concatenate([p.ids for p in parts])
    if len(np.unique(ids)) != len(ids):
        raise ValueError("duplicate session ids across parts")
    offsets = np.cumsum([0, *(p.n for p in parts[:-1])])
    headlines = pd.concat(
        [p.headlines.assign(sidx=p.headlines["sidx"].to_numpy() + off)
         for p, off in zip(parts, offsets, strict=True)],
        ignore_index=True,
    )
    arrays = {col: np.concatenate([getattr(p, col) for p in parts]) for col in PRICE_COLS}
    return Sessions(ids=ids, headlines=headlines, **arrays)


def forward_returns(s: Sessions, spec: TaskSpec) -> np.ndarray:
    """Target return close(halfway) -> close(exit); needs full (research) sessions."""
    if s.n_bars <= spec.exit_bar:
        raise ValueError(f"need bars up to {spec.exit_bar}; these sessions have {s.n_bars}")
    r = s.close[:, spec.exit_bar] / s.close[:, spec.halfway_bar] - 1.0
    return r - r.mean() if spec.demean_returns else r


def _read(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(
            f"{path} not found. Download the data first (scripts/fetch_gdrive.sh or "
            "scripts/fetch_practice_data.sh) and point DATA_DIR at it."
        )
    return pd.read_parquet(path)


@dataclass(frozen=True)
class ArcadeData:
    ins: Sessions  # all bars of the in-sample sessions (research only past halfway!)
    y: np.ndarray  # official in-sample returns aligned to ins.ids
    oos: Sessions  # out-of-sample sessions, first half only


def load_arcade(data_dir: str | Path) -> ArcadeData:
    """Load the Algo Arena (arcade) files: ins_bars/headlines/returns + oos_bars/headlines."""
    d = Path(data_dir)
    ins = from_long(_read(d / "ins_bars.parquet"), _read(d / "ins_headlines.parquet"))
    rets = _read(d / "ins_returns.parquet")
    _require(rets, ("session", "returns"), "returns")
    y = rets.set_index("session")["returns"].reindex(ins.ids).to_numpy(dtype=float)
    if np.isnan(y).any():
        raise ValueError("some in-sample sessions have no return")
    oos = from_long(_read(d / "oos_bars.parquet"), _read(d / "oos_headlines.parquet"))
    return ArcadeData(ins=ins, y=y, oos=oos)


@dataclass(frozen=True)
class DatathonData:
    train: Sessions  # seen + unseen halves joined
    public_test: Sessions
    private_test: Sessions


def load_datathon(data_dir: str | Path) -> DatathonData:
    """Load ETH Datathon 2026 files: {bars,headlines}_{seen,unseen}_{split}.parquet."""
    d = Path(data_dir)

    def part(split: str, kinds: tuple[str, ...]) -> Sessions:
        bars = pd.concat([_read(d / f"bars_{k}_{split}.parquet") for k in kinds])
        hl = pd.concat([_read(d / f"headlines_{k}_{split}.parquet") for k in kinds])
        return from_long(bars, hl)

    return DatathonData(
        train=part("train", ("seen", "unseen")),
        public_test=part("public_test", ("seen",)),
        private_test=part("private_test", ("seen",)),
    )


EXTRA_TRAIN_FILES = (
    "bars_seen_train.parquet",
    "bars_unseen_train.parquet",
    "headlines_seen_train.parquet",
    "headlines_unseen_train.parquet",
)


def load_extra_sessions(data_dir: str | Path,
                        spec: TaskSpec = ARCADE_ALGO_ARENA) -> tuple[Sessions, np.ndarray]:
    """Labelled sessions from the ETH Datathon train set (same generator draw as the arcade data).

    Returns all 100 bars plus the close(halfway) -> close(exit) return, demeaned within these
    sessions the way HRT demeans the in-sample returns.
    """
    d = Path(data_dir)
    frames = [_read(d / name) for name in EXTRA_TRAIN_FILES]
    bars = pd.concat(frames[:2], ignore_index=True)
    headlines = pd.concat(frames[2:], ignore_index=True)
    sessions = from_long(bars, headlines)
    return sessions, forward_returns(sessions, spec)


def list_data_files(data_dir: str | Path) -> pd.DataFrame:
    """Inventory of a data folder: size, rows and columns of every parquet/csv file."""
    rows = []
    for path in sorted(Path(data_dir).rglob("*")):
        if path.suffix not in {".parquet", ".csv"}:
            continue
        info = {"file": str(path.relative_to(data_dir)), "mb": round(path.stat().st_size / 1e6, 2)}
        if path.suffix == ".parquet":
            meta = pq.read_metadata(path)
            info |= {"rows": meta.num_rows, "columns": ", ".join(meta.schema.names)}
        else:
            head = pd.read_csv(path, nrows=5)
            info |= {"rows": None, "columns": ", ".join(head.columns)}
        rows.append(info)
    return pd.DataFrame(rows, columns=["file", "mb", "rows", "columns"])
