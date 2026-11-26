"""Signal exchange: several people research in parallel without touching each other's files.

Every idea is exported as ONE file `signals/<author>/<name>.parquet` holding
  * out-of-fold predictions for every in-sample session, made with `team_folds` (same folds
    for everybody, so signals can be compared and blended fairly), and
  * predictions for every out-of-sample session (model refit on all in-sample data).
`load_signals`, `leaderboard` and `blend_cv` / `blend_predict` turn them into one upload.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import Ridge

from arena.config import ARCADE_ALGO_ARENA, TaskSpec
from arena.cv import CVResult, FitPredict, Folds, cross_validate, kfold, out_of_fold
from arena.metrics import score

TEAM_SEED = 42
TEAM_K = 5
SIGNAL_ROOT = Path("signals")
_SLUG = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def team_folds(n: int) -> Folds:
    """THE folds. Everyone uses these for OOF predictions and CV numbers."""
    return kfold(n, TEAM_K, TEAM_SEED)


def _check_slug(value: str, what: str) -> None:
    if not _SLUG.match(value):
        raise ValueError(f"{what} {value!r}: use lowercase letters, digits, '-' or '_'")


def save_signal(root: str | Path, author: str, name: str, ins_ids, oof, oos_ids, oos,
                allow_nan_ins: bool = False) -> Path:
    """Write one signal file. Overwrites an older version of the same signal.

    `allow_nan_ins` keeps NaN for in-sample rows no fold tested (walk-forward folds)."""
    _check_slug(author, "author")
    _check_slug(name, "signal name")
    parts = []
    for split, ids, values in (("ins", ins_ids, oof), ("oos", oos_ids, oos)):
        ids, values = np.asarray(ids), np.asarray(values, dtype=float)
        if ids.shape != values.shape:
            raise ValueError(f"{split}: {len(ids)} ids but {len(values)} values")
        checked = values[~np.isnan(values)] if (split == "ins" and allow_nan_ins) else values
        if not np.isfinite(checked).all():
            raise ValueError(f"{split}: values contain NaN/inf")
        parts.append(pd.DataFrame({"session": ids, "split": split, "value": values}))
    path = Path(root) / author / f"{name}.parquet"
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.concat(parts, ignore_index=True).to_parquet(path, index=False)
    return path


def export_signal(root: str | Path, author: str, name: str, fit_predict_all: FitPredict,
                  ins_ids, oos_ids, oos_pred, folds: Folds | None = None) -> Path:
    """OOF from `fit_predict_all` + the given OOS predictions -> signal file.

    Default folds are `team_folds`. Pass walk-forward folds for dated data: rows they never
    test are stored as NaN and skipped by the leaderboard and the blend."""
    ins_ids = np.asarray(ins_ids)
    partial = folds is not None
    folds = team_folds(len(ins_ids)) if folds is None else folds
    oof = out_of_fold(fit_predict_all, len(ins_ids), folds, allow_partial=partial)
    return save_signal(root, author, name, ins_ids, oof, oos_ids, oos_pred, allow_nan_ins=partial)


def load_signals(root: str | Path = SIGNAL_ROOT) -> tuple[pd.DataFrame, pd.DataFrame]:
    """All signal files -> (ins, oos) wide tables: index = session, column = 'author/name'."""
    frames = {}
    for path in sorted(Path(root).glob("*/*.parquet")):
        frames[f"{path.parent.name}/{path.stem}"] = pd.read_parquet(path)
    if not frames:
        raise FileNotFoundError(f"no signal files under {root}/<author>/<name>.parquet")

    def wide(split: str) -> pd.DataFrame:
        cols = {k: f[f["split"] == split].set_index("session")["value"] for k, f in frames.items()}
        return pd.DataFrame(cols).sort_index()

    return wide("ins"), wide("oos")


def leaderboard(ins: pd.DataFrame, y: pd.Series, spec: TaskSpec = ARCADE_ALGO_ARENA) -> pd.DataFrame:
    """CV Sharpe of every signal alone + its correlation with the best one."""
    y = y.reindex(ins.index)
    sharpes = {}
    for c in ins.columns:
        ok = ins[c].notna().to_numpy()
        sharpes[c] = score(ins[c].to_numpy()[ok], y.to_numpy()[ok], spec)
    table = pd.DataFrame({"oof_sharpe": sharpes}).sort_values("oof_sharpe", ascending=False)
    table["corr_with_best"] = ins.corr()[table.index[0]].reindex(table.index)
    return table


def _zscore(frame: pd.DataFrame, ref: pd.DataFrame) -> np.ndarray:
    return (frame / ref.std().replace(0, 1)).to_numpy()


def blend_weights(ins: pd.DataFrame, y, alpha: float = 1.0) -> pd.Series:
    """Ridge (no intercept) on std-scaled signals. Weights are per unit of signal std."""
    ok = ins.notna().all(axis=1).to_numpy()
    model = Ridge(alpha=alpha, fit_intercept=False)
    model.fit(_zscore(ins[ok], ins), np.asarray(y, dtype=float)[ok])
    return pd.Series(model.coef_, index=ins.columns, name="weight")


def blend_cv(ins: pd.DataFrame, y, alpha: float = 1.0, spec: TaskSpec = ARCADE_ALGO_ARENA,
             folds: Folds | None = None) -> CVResult:
    """Honest estimate of the blend: weights refit inside every fold (complete rows only).

    `folds` index the rows of `ins` as given (e.g. walk-forward folds on the training table);
    they are mapped onto the complete rows here."""
    ok = ins.notna().all(axis=1).to_numpy()
    if folds is not None:
        position = np.cumsum(ok) - 1  # row of `ins` -> row among the complete rows

        def complete(idx) -> np.ndarray:
            idx = np.asarray(idx)
            return position[idx[ok[idx]]]

        folds = [(complete(tr), complete(te)) for tr, te in folds]
        folds = [(tr, te) for tr, te in folds if len(te)]
    ins, y = ins[ok], np.asarray(y, dtype=float)[ok]

    def fit_predict(train_idx: np.ndarray, test_idx: np.ndarray) -> np.ndarray:
        train = ins.iloc[train_idx]
        w = blend_weights(train, y[train_idx], alpha).to_numpy()
        return _zscore(ins.iloc[test_idx], train) @ w

    return cross_validate(fit_predict, y, folds or kfold(len(y), TEAM_K, TEAM_SEED + 1), spec)


def blend_predict(ins: pd.DataFrame, oos: pd.DataFrame, y, alpha: float = 1.0) -> np.ndarray:
    """Final OOS positions from the blend fitted on all in-sample OOF signals."""
    w = blend_weights(ins, y, alpha)
    return _zscore(oos[ins.columns], ins) @ w.to_numpy()
