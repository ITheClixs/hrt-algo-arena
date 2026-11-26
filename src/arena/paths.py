"""Where things live. Notebooks call `default_data_dir()` so they work from any folder."""

from __future__ import annotations

import os
import warnings
from collections.abc import Sequence
from pathlib import Path

REPO = Path(__file__).resolve().parents[2]
DATA = REPO / "data"
SIGNALS = REPO / "signals"
SUBMISSIONS = REPO / "submissions"

# Checked in order. data/raw = the real competition data you download on the day.
CANDIDATES = (DATA / "raw", DATA / "practice" / "arcade", DATA / "synthetic")


def has_parquet(path: Path) -> bool:
    return path.is_dir() and any(path.rglob("*.parquet"))


def default_data_dir() -> Path:
    """$ARENA_DATA if set, else the first folder in CANDIDATES that holds parquet files."""
    env = os.environ.get("ARENA_DATA")
    if env:
        return Path(env).expanduser().resolve()
    for rank, path in enumerate(CANDIDATES):
        if has_parquet(path):
            if rank > 0:
                warnings.warn(f"{CANDIDATES[0]} has no parquet files: using {path} (NOT the competition "
                              "data). Set ARENA_DATA or DATA_DIR if that is wrong.", stacklevel=2)
            return path
    raise FileNotFoundError(
        "No data found. On the day: scripts/fetch_gdrive.sh <drive-folder-url>. "
        "Before: scripts/fetch_practice_data.sh or uv run python scripts/make_synthetic_data.py"
    )


def find_dir_with(root: str | Path, names: Sequence[str]) -> Path | None:
    """`root` if it holds every file in `names`, else the one folder below it that does
    (Drive downloads often land in a subfolder). None if no folder does."""
    root = Path(root)
    if all((root / name).is_file() for name in names):
        return root
    if not root.is_dir():
        return None
    hits = sorted({p.parent for p in root.rglob(names[0])
                   if all((p.parent / name).is_file() for name in names)})
    if len(hits) > 1:
        raise ValueError(f"several folders under {root} hold the files: "
                         + ", ".join(str(h) for h in hits) + ". Pass --data <folder>.")
    return hits[0] if hits else None
