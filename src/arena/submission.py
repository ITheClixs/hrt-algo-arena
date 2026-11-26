"""Write, validate and log leaderboard uploads. Uploads are scarce (10 at the arcade events)."""

from __future__ import annotations

import csv
import re
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from arena.config import ARCADE_ALGO_ARENA, TaskSpec

LEDGER_COLUMNS = ("time", "author", "file", "label", "cv_sharpe", "leaderboard", "notes")
RUN_STAMP = ".arena_run"  # which data, horizon and model code the runner's files in a folder came from
_RUNNER_UPLOAD = re.compile(r"\d{2}_[a-z0-9_]+_(hrtonly|withextra)\.csv")


def build_submission(ids, pred, spec: TaskSpec = ARCADE_ALGO_ARENA) -> pd.DataFrame:
    """Upload frame. Non-finite predictions become 0 (like HRT's own writer), with a warning:
    the day-of runner refuses them instead (`pipeline.write_version`)."""
    p = np.asarray(pred, dtype=float)
    ids = np.asarray(ids)
    if p.shape != ids.shape:
        raise ValueError(f"{len(ids)} ids but {len(p)} predictions")
    bad = int((~np.isfinite(p)).sum())
    if bad:
        warnings.warn(f"{bad} non-finite prediction(s) set to 0: check the model", stacklevel=2)
    return pd.DataFrame({spec.id_col: _id_values(ids), spec.pred_col: np.where(np.isfinite(p), p, 0.0)})


def _id_values(ids: np.ndarray) -> np.ndarray:
    """Integer-valued numeric ids as int64 (arcade session numbers); anything else unchanged,
    so string or zero-padded ids (real-headline formats) survive."""
    if np.issubdtype(ids.dtype, np.number) and np.all(np.mod(ids, 1) == 0):
        return ids.astype(np.int64)
    return ids


def validate_submission(df: pd.DataFrame, expected_ids, spec: TaskSpec = ARCADE_ALGO_ARENA) -> None:
    """Raise ValueError with a clear message if the file would be rejected or mis-scored."""
    for col in (spec.id_col, spec.pred_col):
        if col not in df.columns:
            raise ValueError(f"missing column {col!r}; need exactly {spec.id_col},{spec.pred_col}")
    ids = df[spec.id_col]
    if ids.duplicated().any():
        raise ValueError(f"{int(ids.duplicated().sum())} duplicate ids")
    expected, got = set(np.asarray(expected_ids).tolist()), set(ids.tolist())
    if expected != got:
        raise ValueError(f"{len(expected - got)} ids missing, {len(got - expected)} unexpected")
    values = pd.to_numeric(df[spec.pred_col], errors="coerce")
    if not np.isfinite(values).all():
        raise ValueError("predictions contain NaN/inf or non-numbers")
    if values.abs().sum() == 0:
        raise ValueError("all predictions are zero: Sharpe would be 0")


def write_submission(ids, pred, path: str | Path, spec: TaskSpec = ARCADE_ALGO_ARENA,
                     expected_ids=None) -> Path:
    """Build, validate and write the upload CSV. Writing does NOT spend an upload."""
    df = build_submission(ids, pred, spec)
    validate_submission(df, ids if expected_ids is None else expected_ids, spec)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(path, index=False, lineterminator="\n")
    return path


def log_submission(ledger: str | Path, *, file: str, label: str, cv_sharpe: float, author: str,
                   leaderboard: float | None = None, notes: str = "") -> pd.DataFrame:
    """Append one row to the team's upload ledger and return the whole ledger."""
    ledger = Path(ledger)
    ledger.parent.mkdir(parents=True, exist_ok=True)
    is_new = not ledger.exists() or ledger.stat().st_size == 0
    row = {
        "time": datetime.now().strftime("%H:%M:%S"),
        "author": author,
        "file": file,
        "label": label,
        "cv_sharpe": round(float(cv_sharpe), 4),
        "leaderboard": "" if leaderboard is None else round(float(leaderboard), 4),
        "notes": notes,
    }
    with ledger.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=LEDGER_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return pd.read_csv(ledger)


def retire_stale_uploads(out_dir: str | Path, stamp: str) -> list[Path]:
    """Move the runner's upload files out of the way when they came from other data, horizon or code.

    They go to out_dir/old/<time>/ (nothing is deleted), so the folder only ever holds files that
    match the current run: a practice `_withextra` file can never sit next to NEW DATA files.
    Returns the moved files' new paths."""
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    marker = out / RUN_STAMP
    previous = marker.read_text().strip() if marker.is_file() else None
    files = sorted(p for p in out.glob("*.csv") if _RUNNER_UPLOAD.fullmatch(p.name))
    moved: list[Path] = []
    if files and previous != stamp:
        dest = out / "old" / datetime.now().strftime("%Y%m%d-%H%M%S-%f")
        dest.mkdir(parents=True)
        moved = [p.rename(dest / p.name) for p in files]
    marker.write_text(stamp + "\n")
    return moved
