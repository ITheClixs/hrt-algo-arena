"""Is this the Algo Arena dataset we already know?

HRT shipped byte-identical arcade files at three events in Nov 2026 (7, 21-22 and 24 Nov). The
runner hashes the downloaded files on the day: SAME DATA means every pre-validated number
applies; NEW DATA means refit everything from scratch and re-check the generator facts.

Two levels: the bytes of each file, and, when the bytes differ, its content (the same rows and
values, whatever the row order, dtype width or compression), so a re-saved copy of the known
files is still recognised. HRT's scorer `sessions.py` is checked separately: if it changed, the
scoring may have changed.
"""

from __future__ import annotations

import hashlib
from collections.abc import Mapping
from pathlib import Path

import pandas as pd

ARCADE_FILES: tuple[str, ...] = (
    "ins_bars.parquet",
    "ins_headlines.parquet",
    "ins_returns.parquet",
    "oos_bars.parquet",
    "oos_headlines.parquet",
)
SCORER_FILE = "sessions.py"

# SHA-256 of data/practice/arcade/ (computed 25 Nov 2026).
KNOWN_ARCADE_SHA256: dict[str, str] = {
    "ins_bars.parquet": "32dc426a6078c77d60fcf4b403677c3ba4030247e4573784f7345e80f8e25960",
    "ins_headlines.parquet": "36a86dd2c82c8d252f3020e5f7a2bd248c6112200f5ceb26f872232ae560a4c0",
    "ins_returns.parquet": "91090a33d6b4d360fa856d71bec39f73ecefd88bdc4250a0f5b47f602abdda3c",
    "oos_bars.parquet": "71a5714f20ac75b86480ca625e1a4f6f1b48d0f7e978bbd08fd84bb861c37662",
    "oos_headlines.parquet": "4f99213116881d6cf15b774aa70456704cab47fdccfaf405e17aac44b992afa1",
}
# content_sha256 of the same files (computed 25 Nov 2026 with `content_fingerprint`).
KNOWN_ARCADE_CONTENT: dict[str, str] = {
    "ins_bars.parquet": "57448bfe329682b122cf35b82805c3af1fb321e7d4adfc21bffe8b75d18b73e2",
    "ins_headlines.parquet": "cf1ac12e103fc3a4ea4b7fb84d3006ff7de4ec50dc4aac33bcaa5e3770ba4605",
    "ins_returns.parquet": "caabd704b1a4a65a4838815f6e9229e9c5c7a995d7ef542e3f6a594ae2844cbc",
    "oos_bars.parquet": "131261942ac48671fe3dafc40af98a4c77b0c14ceb4320ad60deb618e7e2a78d",
    "oos_headlines.parquet": "f68175854074fe192c42411147dece9387bbb028e9b21e19d08e418f4c30dc2c",
}
KNOWN_SESSIONS_PY_SHA256 = "c536970137868707402f0969b32ddd83e90ac929f065990f45a6a983c8f1c22f"

_CHUNK = 1 << 20


def file_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as fh:
        for block in iter(lambda: fh.read(_CHUNK), b""):
            digest.update(block)
    return digest.hexdigest()


def fingerprint(data_dir: str | Path) -> dict[str, str | None]:
    """SHA-256 of each arcade file in `data_dir` (None when the file is missing)."""
    d = Path(data_dir)
    return {name: file_sha256(d / name) if (d / name).is_file() else None for name in ARCADE_FILES}


def _canonical(df: pd.DataFrame) -> pd.DataFrame:
    """Same rows and values -> same frame: sorted columns and rows, int64/float64/str dtypes."""
    df = df[sorted(df.columns)]
    kinds = {}
    for col in df.columns:
        if pd.api.types.is_integer_dtype(df[col]):
            kinds[col] = "int64"
        elif pd.api.types.is_float_dtype(df[col]):
            kinds[col] = "float64"
        else:
            kinds[col] = str
    df = df.astype(kinds)
    return df.sort_values(list(df.columns), kind="stable").reset_index(drop=True)


def content_sha256(path: Path) -> str:
    df = _canonical(pd.read_parquet(path))
    digest = hashlib.sha256(repr([(c, str(df[c].dtype)) for c in df.columns]).encode())
    digest.update(pd.util.hash_pandas_object(df, index=False).to_numpy().tobytes())
    return digest.hexdigest()


def content_fingerprint(data_dir: str | Path) -> dict[str, str | None]:
    d = Path(data_dir)
    return {name: content_sha256(d / name) if (d / name).is_file() else None for name in ARCADE_FILES}


def _all_match(found: Mapping[str, str | None], known: Mapping[str, str]) -> bool:
    return all(found[name] is not None and found[name] == known.get(name) for name in ARCADE_FILES)


def same_as_known(data_dir: str | Path, known: Mapping[str, str] = KNOWN_ARCADE_SHA256) -> bool:
    return _all_match(fingerprint(data_dir), known)


def match_known(data_dir: str | Path, known_bytes: Mapping[str, str] = KNOWN_ARCADE_SHA256,
                known_content: Mapping[str, str] = KNOWN_ARCADE_CONTENT) -> str | None:
    """"bytes" (byte-identical), "content" (same data, re-saved) or None (new data)."""
    if same_as_known(data_dir, known_bytes):
        return "bytes"
    if known_content and _all_match(content_fingerprint(data_dir), known_content):
        return "content"
    return None


def scorer_status(data_dir: str | Path, known: str = KNOWN_SESSIONS_PY_SHA256) -> str:
    """"same", "changed" or "missing": HRT's sessions.py next to the data vs the known one."""
    path = Path(data_dir) / SCORER_FILE
    if not path.is_file():
        return "missing"
    return "same" if file_sha256(path) == known else "changed"
