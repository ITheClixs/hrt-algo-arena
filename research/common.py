"""Shared helpers for the research scripts. Protocol: README, section Validation.

Run a rung from the repo root, e.g. `uv run python research/r0_reference.py`.
On new competition files add `--data <folder>` (e.g. `--data data/raw`): the horizon is read off
those files, the industry count is estimated, and results go to research/oof_<folder>/ and
research/trials_<folder>.csv, never mixed with the known dataset's. Start a new folder with the
references (r0_reference.py, r0b_sector.py), because each rung compares with an earlier one.
"""

from __future__ import annotations

import argparse
import functools
import re
from collections.abc import Callable, Mapping, Sequence
from pathlib import Path

from arena.config import ARCADE_ALGO_ARENA
from arena.fingerprint import ARCADE_FILES
from arena.paths import DATA, REPO, find_dir_with
from arena.trials import OOF_DIR, TRIALS_CSV, load_evaluation, log_trial, save_evaluation
from arena.validation import DEV_SEEDS, Evaluation, compare, evaluate

PRACTICE = DATA / "practice" / "arcade"


def _data_dir() -> Path:
    parser = argparse.ArgumentParser(add_help=False)
    parser.add_argument("--data", type=Path, default=None)
    args, _ = parser.parse_known_args()
    if args.data is None:
        return PRACTICE.resolve()
    root = args.data.expanduser()
    return (find_dir_with(root, ARCADE_FILES) or root).resolve()  # Drive downloads land in subfolders


DATA_DIR = _data_dir()
KNOWN = DATA_DIR == PRACTICE.resolve()
_TAG = re.sub(r"[^A-Za-z0-9]+", "_", DATA_DIR.name)
OOF = OOF_DIR if KNOWN else REPO / "research" / f"oof_{_TAG}"
TRIALS = TRIALS_CSV if KNOWN else REPO / "research" / f"trials_{_TAG}.csv"


def _spec():
    if KNOWN:
        return ARCADE_ALGO_ARENA
    from arena.data import load_arcade
    from arena.horizon import resolve_spec

    spec = resolve_spec(ARCADE_ALGO_ARENA, load_arcade(DATA_DIR))
    print(f"research on {DATA_DIR}: halfway {spec.halfway_bar}, exit {spec.exit_bar}; results -> {OOF.name}/",
          flush=True)
    return spec


SPEC = _spec()


@functools.cache
def dev_universe():
    """The research dataset (known data unless --data), preprocessed once per process (no extra sessions)."""
    from arena.structure import estimate_n_groups
    from arena.universe import WEAK_GAP, load_universe, without_industry_filter

    u = load_universe(DATA_DIR, spec=SPEC, n_groups=5 if KNOWN else None)
    if not KNOWN and estimate_n_groups(u.ins, u.oos, last_bar=SPEC.halfway_bar)[1] < WEAK_GAP:
        print("no clear industry structure: research without the industry filter (as the runner does)", flush=True)
        u = without_industry_filter(u)
    return u


def run_trial(idea: str, build: Callable[[], object], params: Mapping[str, object],
              vs: str | None = None, seeds: Sequence[int] = DEV_SEEDS) -> Evaluation:
    """Evaluate one idea on the development folds, compare it with `vs`, log and save it."""
    from arena.modelapi import as_fit_predict_with_unlabeled

    u = dev_universe()
    fp = as_fit_predict_with_unlabeled(build(), u.ins, u.y, unlabeled=u.oos,
                                       halfway=SPEC.halfway_bar)
    ev = evaluate(idea, fp, u.y, seeds=seeds, spec=SPEC)
    comparison = compare(ev, load_evaluation(vs, OOF), u.y, SPEC) if vs else None
    log_trial(ev, idea=idea, params=params, comparison=comparison, vs=vs or "", path=TRIALS)
    save_evaluation(ev, OOF)
    line = f"{idea} | CV {ev.mean:.3f} | worst {ev.worst_fold:.3f} | {ev.runtime_s:.0f}s"
    if comparison is not None:
        verdict = "KEEP" if comparison.keep else "drop"
        line += f" | Δ vs {vs} {comparison.delta:+.3f} (z {comparison.z:.1f}) | {verdict}"
    print(line, flush=True)
    return ev


State = tuple[str, dict]


def step(state: State, idea: str, change: Callable[[dict], dict], build: Callable[[dict], object]) -> State:
    """One ladder rung: evaluate `change` applied to the current best config; keep it only if
    `compare(...).keep`. Returns the (possibly unchanged) best name and config."""
    best, cfg = state
    new = {**cfg, **change(cfg)}
    ev = run_trial(idea, lambda: build(new), new, vs=best)
    keep = compare(ev, load_evaluation(best, OOF), dev_universe().y, SPEC).keep
    return (idea, new) if keep else state
