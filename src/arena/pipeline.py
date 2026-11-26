"""Upload candidates: the frozen models the day-of runner fits and writes.

Every candidate is written in two versions:
  * `_hrtonly`   - fitted on HRT's in-sample sessions only
  * `_withextra` - also fitted on the 1,000 ETH Datathon train sessions (same generator draw)
Upload the version that matches the organisers' answer on outside data.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from arena.blend import BlendModel
from arena.config import TaskSpec
from arena.data import concat_sessions
from arena.generator import SECTOR_REF, GeneratorModel
from arena.learners import OwedTfidfModel
from arena.modelapi import ArcadeModel
from arena.references import BaselineDrift, WinnerRef
from arena.submission import write_submission
from arena.universe import Universe


@dataclass(frozen=True)
class Candidate:
    label: str  # file-name slug
    build: Callable[[TaskSpec], ArcadeModel]
    note: str  # one line for the runner's table
    cv_known: float | None = None  # dev-fold CV on the known dataset (research/trials.csv)


EVERY4 = tuple(range(0, 41, 4))
PRIMARY = {**SECTOR_REF, "augment_offsets": EVERY4, "clip_sd": 2.0}


def _generator(**changes) -> Callable[[TaskSpec], ArcadeModel]:
    return lambda spec: GeneratorModel(spec, **{**SECTOR_REF, **changes})


def _blend_tfidf(spec: TaskSpec) -> ArcadeModel:
    return BlendModel((GeneratorModel(spec, **PRIMARY), OwedTfidfModel(spec, alpha=3000.0, augment_offsets=EVERY4)),
                      (1.0, 1.0))


# Frozen 25 Nov after R6 (research/RESULTS.md). Upload 1 = primary; the rest differ in one known way.
# cv_known = development-fold CV on the known dataset (confirmation seeds within 0.006).
# volinv and baseline were added after the 25 Nov review; new labels go at the end so file numbers stay.
CANDIDATES: tuple[Candidate, ...] = (
    Candidate("primary", _generator(augment_offsets=EVERY4, clip_sd=2.0),
              "generator fit + augmented calibration + 2-sd clip (R3 bundle)", cv_known=3.568),
    Candidate("clip2", _generator(clip_sd=2.0), "primary without augmentation", cv_known=3.552),
    Candidate("aug", _generator(augment_offsets=EVERY4), "primary without clipping", cv_known=3.554),
    Candidate("sector", _generator(), "Edinburgh-style sector model (R0b reference)", cv_known=3.537),
    Candidate("winner", WinnerRef, "7 Nov winner-style model (R0c), different family", cv_known=3.471),
    Candidate("blend_tfidf", _blend_tfidf, "primary + owed-push TF-IDF, equal weights (R5)", cv_known=3.532),
    Candidate("volinv", _generator(augment_offsets=EVERY4, clip_sd=2.0, extra_features=("inv_vol",)),
              "primary + 1/vol calibration feature (review idea, dev +0.016 but fresh windows +0.002)",
              cv_known=3.583),
    Candidate("baseline", BaselineDrift, "R0a: our own quickstart model, the clean fallback if public "
              "pipelines are not allowed", cv_known=3.400),
)


def fit_predict_oos(candidate: Candidate, u: Universe, spec: TaskSpec, with_extra: bool) -> np.ndarray:
    """Fit on all labelled sessions (optionally + extra) and predict the out-of-sample sessions."""
    if with_extra and u.extra is None:
        raise ValueError("with_extra=True but no extra sessions were loaded")
    labelled = concat_sessions(u.ins, u.extra) if with_extra else u.ins
    y = np.concatenate([u.y, u.y_extra]) if with_extra else u.y
    fitted = candidate.build(spec).fit(labelled, y, unlabeled=u.oos)
    return np.asarray(fitted.predict(u.oos), dtype=float)


def write_version(candidate: Candidate, u: Universe, spec: TaskSpec, out_dir: Path, index: int,
                  with_extra: bool) -> Path:
    suffix = "withextra" if with_extra else "hrtonly"
    path = Path(out_dir) / f"{index:02d}_{candidate.label}_{suffix}.csv"
    pred = fit_predict_oos(candidate, u, spec, with_extra)
    bad = int((~np.isfinite(pred)).sum())
    if bad:
        raise ValueError(f"{candidate.label}: {bad} non-finite predictions; refusing to write {path.name}")
    return write_submission(u.oos.ids, pred, path, spec, expected_ids=u.oos.ids)


def write_candidate(candidate: Candidate, u: Universe, spec: TaskSpec, out_dir: Path,
                    index: int) -> list[Path]:
    versions = (False, True) if u.extra is not None else (False,)
    return [write_version(candidate, u, spec, out_dir, index, v) for v in versions]
