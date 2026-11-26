"""Trial log: every evaluation leaves a row, so a CV number can't hide how many tries it took.

`research/trials.csv` is committed. Out-of-fold predictions are saved next to it (in
`research/oof/`, not committed) so a later script can compare against an earlier best.
"""

from __future__ import annotations

import csv
import json
import subprocess
from collections.abc import Mapping
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

from arena.paths import REPO
from arena.validation import Comparison, Evaluation

TRIALS_CSV = REPO / "research" / "trials.csv"
OOF_DIR = REPO / "research" / "oof"
TRIAL_COLUMNS = (
    "time", "idea", "params", "seeds", "cv_mean", "worst_fold", "seed_scores", "vs", "delta",
    "se", "z", "keep", "runtime_s", "git_sha",
)


def git_sha() -> str:
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=REPO, check=True,
                             capture_output=True, text=True, timeout=10)
    except (OSError, subprocess.SubprocessError):
        return "unknown"
    return out.stdout.strip() or "unknown"


def _blank_or(comparison: Comparison | None, value) -> object:
    return "" if comparison is None else value


def log_trial(evaluation: Evaluation, *, idea: str, params: Mapping[str, object],
              comparison: Comparison | None = None, vs: str = "",
              path: Path = TRIALS_CSV) -> pd.DataFrame:
    """Append one row for `evaluation` and return the whole log."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    c = comparison
    row = {
        "time": datetime.now().isoformat(timespec="seconds"),
        "idea": idea,
        "params": json.dumps(dict(params), sort_keys=True, default=str),
        "seeds": " ".join(str(s) for s in evaluation.seeds),
        "cv_mean": round(evaluation.mean, 4),
        "worst_fold": round(evaluation.worst_fold, 4),
        "seed_scores": " ".join(f"{s:.4f}" for s in evaluation.seed_scores),
        "vs": vs,
        "delta": _blank_or(c, c and round(c.delta, 4)),
        "se": _blank_or(c, c and round(c.se, 4)),
        "z": _blank_or(c, c and round(c.z, 2)),
        "keep": _blank_or(c, c and c.keep),
        "runtime_s": round(evaluation.runtime_s, 1),
        "git_sha": git_sha(),
    }
    is_new = not path.exists() or path.stat().st_size == 0
    with path.open("a", newline="") as fh:
        writer = csv.DictWriter(fh, fieldnames=TRIAL_COLUMNS)
        if is_new:
            writer.writeheader()
        writer.writerow(row)
    return pd.read_csv(path)


def save_evaluation(ev: Evaluation, directory: Path = OOF_DIR) -> Path:
    directory = Path(directory)
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / f"{ev.name}.npz"
    np.savez_compressed(path, name=np.array(ev.name), seeds=np.array(ev.seeds),
                        seed_scores=np.array(ev.seed_scores), fold_scores=np.array(ev.fold_scores),
                        oof=ev.oof, runtime_s=np.array(ev.runtime_s))
    return path


def load_evaluation(name: str, directory: Path = OOF_DIR) -> Evaluation:
    path = Path(directory) / f"{name}.npz"
    if not path.exists():
        raise FileNotFoundError(f"no saved evaluation {name!r} in {directory}: run its research script first")
    with np.load(path) as z:
        return Evaluation(
            name=str(z["name"]),
            seeds=tuple(int(s) for s in z["seeds"]),
            seed_scores=tuple(float(s) for s in z["seed_scores"]),
            fold_scores=tuple(float(s) for s in z["fold_scores"]),
            oof=z["oof"],
            runtime_s=float(z["runtime_s"]),
        )
