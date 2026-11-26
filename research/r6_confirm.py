"""R6: confirm the shortlist before freezing (plan Task 16).

1. confirmation folds (fresh seeds 1042-1044)  2. the 1,000 datathon sessions, scored once
3. stress: 10% of test headlines dropped        4. half-split stability
5. new-draw rehearsal of the runner on synthetic data with industries and vol-scaled pushes

    uv run python research/r6_confirm.py
"""

from __future__ import annotations

import tempfile
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
from common import SPEC, dev_universe, run_trial
from r2_sizing import SECTOR_REF

from arena.data import subset
from arena.generator import GeneratorModel
from arena.metrics import score, sharpe_standard_error
from arena.modelapi import as_fit_predict_with_unlabeled
from arena.paths import DATA
from arena.references import WinnerRef
from arena.runner import main as run_runner
from arena.synth import make_synthetic, write_arcade_files
from arena.universe import load_universe
from arena.validation import CONFIRM_SEEDS, evaluate

EVERY4 = tuple(range(0, 41, 4))
SHORTLIST = {
    "r3_bundle_aug_clip2": lambda: GeneratorModel(SPEC, **{**SECTOR_REF, "augment_offsets": EVERY4, "clip_sd": 2.0}),
    "r3_aug_every4": lambda: GeneratorModel(SPEC, **{**SECTOR_REF, "augment_offsets": EVERY4}),
    "r2_clip2": lambda: GeneratorModel(SPEC, **{**SECTOR_REF, "clip_sd": 2.0}),
    "r0b_sector": lambda: GeneratorModel(SPEC, **SECTOR_REF),
    "r0c_winner": lambda: WinnerRef(SPEC),
}


@dataclass(frozen=True)
class _Dropping:
    inner: object
    frac: float
    seed: int

    def predict(self, seen):
        keep = np.random.default_rng(self.seed).random(len(seen.headlines)) >= self.frac
        return self.inner.predict(replace(seen, headlines=seen.headlines[keep].reset_index(drop=True)))


@dataclass(frozen=True)
class DropHeadlines:
    inner: object
    frac: float = 0.1
    seed: int = 0

    def fit(self, full, y, unlabeled=None):
        return _Dropping(self.inner.fit(full, y, unlabeled=unlabeled), self.frac, self.seed)


def confirm() -> None:
    for name, build in SHORTLIST.items():
        run_trial(f"{name}@confirm", build, {"confirm_of": name}, seeds=CONFIRM_SEEDS)


def check_set() -> None:
    u = load_universe(DATA / "practice" / "arcade", extra_dir=DATA / "practice" / "datathon")
    for name, build in SHORTLIST.items():
        pred = build().fit(u.ins, u.y, unlabeled=u.oos).predict(u.extra)
        s = score(pred, u.y_extra, SPEC)
        print(f"check set (1,000 datathon sessions) | {name} | Sharpe {s:.3f} ± {sharpe_standard_error(s, 1000):.2f}")


def stress() -> None:
    u = dev_universe()
    for name in ("r3_bundle_aug_clip2", "r0b_sector"):
        base = evaluate(name, as_fit_predict_with_unlabeled(SHORTLIST[name](), u.ins, u.y, unlabeled=u.oos), u.y,
                        seeds=(42,), spec=SPEC)
        hit = evaluate(name, as_fit_predict_with_unlabeled(DropHeadlines(SHORTLIST[name]()), u.ins, u.y,
                                                           unlabeled=u.oos), u.y, seeds=(42,), spec=SPEC)
        print(f"stress | {name} | seed-42 CV {base.mean:.3f} -> {hit.mean:.3f} with 10% of test headlines dropped")
        order = np.random.default_rng(0).permutation(u.ins.n)
        a, b = np.sort(order[: u.ins.n // 2]), np.sort(order[u.ins.n // 2:])
        halves = []
        for tr, te in ((a, b), (b, a)):
            fitted = SHORTLIST[name]().fit(subset(u.ins, tr), u.y[tr], unlabeled=u.oos)
            halves.append(score(fitted.predict(subset(u.ins, te)), u.y[te], SPEC))
        print(f"stress | {name} | half-split: A->B {halves[0]:.3f}, B->A {halves[1]:.3f}")


def new_draw() -> None:
    with tempfile.TemporaryDirectory() as tmp:
        d = write_arcade_files(make_synthetic(n_sessions=2000, seed=11, n_industries=5, vol_scaled=True),
                               Path(tmp) / "draw")
        code = run_runner(["--data", str(d), "--extra", "none", "--out", str(Path(tmp) / "out"), "--cv", "fast"])
        print("new-draw rehearsal exit code:", code)


if __name__ == "__main__":
    confirm()
    check_set()
    stress()
    new_draw()
