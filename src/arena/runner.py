"""Day-of command: competition files -> validated upload CSVs.

    uv run python scripts/arena_run.py                    # data/raw, all candidates
    uv run python scripts/arena_run.py --only primary     # just upload 1
    uv run python scripts/arena_run.py --practice         # rehearse on data/practice/arcade

What it prints, in order: the folder it reads; SAME DATA (the dataset HRT used at three earlier
events: every stored CV applies) or NEW DATA (everything is refitted, the extra sessions stay
out unless a folder is given, the industry count is estimated); the horizon read off the files;
the industries; one line per upload file; then the CVs. Every file is written before any CV
runs, so a failing CV never costs a file. Exit code 2 = nothing written (fix the input first),
1 = some file failed, 0 = all files written.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import time
from collections.abc import Sequence
from pathlib import Path

import numpy as np

from arena.config import ARCADE_ALGO_ARENA, TaskSpec
from arena.data import EXTRA_TRAIN_FILES, ArcadeData, load_arcade, load_extra_sessions
from arena.facts import generator_facts
from arena.fingerprint import (
    ARCADE_FILES,
    KNOWN_ARCADE_CONTENT,
    KNOWN_ARCADE_SHA256,
    SCORER_FILE,
    fingerprint,
    match_known,
    scorer_status,
)
from arena.horizon import detect_horizon, resolve_spec, scorer_constants
from arena.modelapi import as_fit_predict_with_unlabeled
from arena.paths import DATA, SUBMISSIONS, find_dir_with
from arena.pipeline import CANDIDATES, Candidate, write_version
from arena.structure import estimate_n_groups
from arena.submission import retire_stale_uploads
from arena.universe import (
    DEFAULT_N_GROUPS,
    WEAK_GAP,
    Universe,
    build_universe,
    without_industry_filter,
)
from arena.validation import evaluate

RAW = DATA / "raw"
PRACTICE_ARCADE = DATA / "practice" / "arcade"
DEFAULT_EXTRA = DATA / "practice" / "datathon"


class RunError(Exception):
    """Something to fix before any file is written (printed, exit code 2)."""


def _parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(prog="arena_run", description="Write the Algo Arena upload files.")
    p.add_argument("--data", type=Path, default=None,
                   help="folder with the arcade files (default: $ARENA_DATA, else data/raw; subfolders are searched)")
    p.add_argument("--practice", action="store_true",
                   help="allow data/practice/arcade when data/raw is empty (rehearsals only)")
    p.add_argument("--extra", default="auto",
                   help="datathon train folder, 'none', or 'auto' (practice datathon folder on SAME DATA only)")
    p.add_argument("--out", type=Path, default=SUBMISSIONS, help="where the CSVs go")
    p.add_argument("--only", nargs="+", default=None, metavar="LABEL", help="candidate labels to write")
    p.add_argument("--cv", choices=("auto", "none", "fast"), default="auto",
                   help="auto: stored CV on SAME DATA, quick CV of the primary on NEW DATA; fast: quick CV of each")
    p.add_argument("--groups", type=int, default=None,
                   help="number of industries (default: 5 on SAME DATA, estimated on NEW DATA)")
    p.add_argument("--no-industry", action="store_true", help="keep every headline (no industry filter)")
    p.add_argument("--halfway", type=int, default=None, help="trading bar (default: read off the files)")
    p.add_argument("--exit", type=int, default=None, help="exit bar (default: read off the files)")
    return p


def _selected(only: Sequence[str] | None) -> list[tuple[int, Candidate]]:
    numbered = list(enumerate(CANDIDATES, start=1))
    if only is None:
        return numbered
    unknown = sorted(set(only) - {c.label for c in CANDIDATES})
    if unknown:
        raise RunError(f"unknown label(s) {', '.join(unknown)}; choose from "
                       + ", ".join(c.label for c in CANDIDATES))
    return [(i, c) for i, c in numbered if c.label in only]


def _data_root(args: argparse.Namespace) -> tuple[Path, str]:
    env = os.environ.get("ARENA_DATA")
    if args.data is not None:
        return Path(args.data), "--data"
    if env:
        return Path(env).expanduser(), "$ARENA_DATA"
    if find_dir_with(RAW, ARCADE_FILES) is not None:
        return RAW, "data/raw"
    if args.practice:
        return PRACTICE_ARCADE, "--practice"
    raise RunError(f"no competition files in {RAW}. Download them (scripts/fetch_gdrive.sh "
                   "\"<url>\") or pass --data <folder>. To rehearse on the practice data: --practice")


def _data_dir(args: argparse.Namespace) -> Path:
    try:
        root, how = _data_root(args)
        found = find_dir_with(root, ARCADE_FILES)
    except ValueError as exc:  # several folders hold the files
        raise RunError(str(exc)) from None
    data_dir = (found or root).resolve()
    print(f"data: {data_dir}  ({how})")
    if data_dir.is_relative_to((DATA / "practice").resolve()) or how == "--practice":
        print("PRACTICE DATA: rehearsal only, these are NOT the competition files")
    return data_dir


def _identity(data_dir: Path) -> bool:
    match = match_known(data_dir, KNOWN_ARCADE_SHA256, KNOWN_ARCADE_CONTENT)
    if match is None:
        print("NEW DATA: refitting from scratch")
    else:
        how = "byte-identical" if match == "bytes" else "same content, re-saved files"
        print(f"SAME DATA (known Algo Arena dataset, {how})")
    status = scorer_status(data_dir)
    print({"same": f"scorer {SCORER_FILE}: same as the known one",
           "changed": f"scorer {SCORER_FILE}: CHANGED. Read it before uploading: the scoring may differ",
           "missing": f"scorer {SCORER_FILE}: not shipped"}[status])
    return match is not None


def _load(data_dir: Path) -> ArcadeData:
    try:
        return load_arcade(data_dir)
    except Exception as exc:  # unreadable or differently shaped files: a clear message, not a traceback
        raise RunError(f"cannot read the arcade files in {data_dir}: {type(exc).__name__}: {exc}") from None


def _spec(args: argparse.Namespace, d: ArcadeData, data_dir: Path) -> TaskSpec:
    try:
        spec = resolve_spec(ARCADE_ALGO_ARENA, d, halfway=args.halfway, exit_bar=args.exit)
    except ValueError as exc:
        raise RunError(str(exc)) from None
    source = "read off the files" if detect_horizon(d).ok else "from --halfway/--exit: the files do not determine it"
    print(f"horizon: trade at close of bar {spec.halfway_bar}, exit at close of bar {spec.exit_bar} "
          f"({'demeaned' if spec.demean_returns else 'raw'} returns, {source})")
    declared = scorer_constants(data_dir / SCORER_FILE)
    if declared and (declared.get("HALFWAY_BAR"), declared.get("HORIZON_BARS")) != (spec.halfway_bar, spec.horizon):
        print(f"WARNING: {SCORER_FILE} declares {declared}, the files say halfway {spec.halfway_bar}, "
              f"horizon {spec.horizon}. Ask the organisers which one is scored")
    if np.intersect1d(d.ins.ids, d.oos.ids).size:
        print("WARNING: in-sample and out-of-sample session ids overlap (handled; check the upload ids)")
    return spec


def _extra_dir(arg: str, same: bool) -> Path | None:
    if arg == "none":
        print("no extra sessions (--extra none): writing _hrtonly files only")
        return None
    if arg == "auto" and not same:
        print("NEW DATA: extra sessions off (they come from the known draw). Writing _hrtonly files only; "
              "pass --extra <folder> to force them in")
        return None
    folder = DEFAULT_EXTRA if arg == "auto" else Path(arg)
    if not all((folder / name).is_file() for name in EXTRA_TRAIN_FILES):
        print(f"no extra sessions in {folder}: writing _hrtonly files only")
        return None
    if not same:
        print(f"WARNING: extra sessions from {folder} on NEW DATA; use _withextra only if they come from this draw")
    return folder


def _extra_sessions(arg: str, same: bool, spec: TaskSpec):
    folder = _extra_dir(arg, same)
    if folder is None:
        return None, None
    try:
        return load_extra_sessions(folder, spec)
    except Exception as exc:  # optional input: it must never cost the _hrtonly files
        print(f"WARNING: cannot read the extra sessions in {folder} ({type(exc).__name__}: {exc}). "
              "Writing _hrtonly files only")
        return None, None


def _universe(args: argparse.Namespace, d: ArcadeData, spec: TaskSpec, same: bool) -> tuple[Universe, bool]:
    """The tagged universe and whether the files use the industry filter."""
    extra, y_extra = _extra_sessions(args.extra, same, spec)
    n_groups = args.groups if args.groups is not None else (DEFAULT_N_GROUPS if same else None)
    u = build_universe(d.ins, d.oos, d.y, extra, y_extra, n_groups=n_groups, halfway=spec.halfway_bar)
    estimate, gap = estimate_n_groups(u.ins, u.oos, last_bar=spec.halfway_bar)
    k = u.groups.nunique()
    sizes = " ".join(str(v) for v in sorted(u.groups.value_counts(), reverse=True))
    n_extra = f", {u.extra.n} extra" if u.extra is not None else ""
    print(f"{u.ins.n} in-sample, {u.oos.n} out-of-sample{n_extra} sessions | "
          f"{u.ins.headlines['key'].nunique()} templates")
    print(f"industries: {k} groups, sizes {sizes} | eigengap suggests {estimate} (gap {gap:.2f})")
    if args.no_industry:
        print("industry filter OFF (--no-industry): every headline counts")
        return u, False
    if gap < WEAK_GAP and args.groups is None and not same:
        print(f"no clear industry structure (gap below {WEAK_GAP}): industry filter OFF, every headline counts. "
              f"--groups {k} forces it on")
        return u, False
    if gap < WEAK_GAP:
        print("WARNING: no clear industry structure: compare the CV with --no-industry")
    elif estimate != k:
        print(f"WARNING: the industry count looks like {estimate}, not {k}: compare the CV with --groups {estimate}")
    return u, True


def _stamp(data_dir: Path, spec: TaskSpec) -> str:
    """Which data, horizon and model code the upload files come from (src/arena, not the git commit,
    so committing the ledger does not count as new code)."""
    code = hashlib.sha256()
    for path in sorted(Path(__file__).parent.glob("*.py")):
        code.update(path.name.encode() + path.read_bytes())
    parts = {"files": fingerprint(data_dir), "halfway": spec.halfway_bar, "exit": spec.exit_bar,
             "code": code.hexdigest()}
    return hashlib.sha256(json.dumps(parts, sort_keys=True).encode()).hexdigest()[:16]


def _print_facts(u: Universe, spec: TaskSpec) -> None:
    try:
        facts = generator_facts(u, spec.halfway_bar)
    except Exception as exc:  # informative only: never block the upload files
        print(f"generator facts n/a ({type(exc).__name__}: {exc})")
        return
    print("generator facts (known data: 302 templates, 5 industries, main_share 0.79, push_duration 11): "
          + ", ".join(f"{k}={v:.3g}" for k, v in facts.items()))


def _write_all(selected: list[tuple[int, Candidate]], u: Universe, spec: TaskSpec, out: Path) -> bool:
    ok = True
    for index, cand in selected:
        for with_extra in (False, True) if u.extra is not None else (False,):
            name = f"{index:02d}_{cand.label}_{'withextra' if with_extra else 'hrtonly'}"
            t = time.perf_counter()
            try:
                path = write_version(cand, u, spec, out, index, with_extra)
            except Exception as exc:  # report and go on: the other files may still be fine
                print(f"FAILED {name}: {type(exc).__name__}: {exc}")
                ok = False
                continue
            print(f"{path}  {u.oos.n} rows  {time.perf_counter() - t:.1f}s  [{cand.note}]")
    return ok


def _quick_cv(cand: Candidate, u: Universe, spec: TaskSpec) -> str:
    try:
        fp = as_fit_predict_with_unlabeled(cand.build(spec), u.ins, u.y, unlabeled=u.oos,
                                           halfway=spec.halfway_bar)
        ev = evaluate(cand.label, fp, u.y, seeds=(42,), spec=spec)
    except Exception as exc:  # the files are written already; a CV problem must not stop the run
        return f"CV n/a ({type(exc).__name__}: {exc})"
    return f"CV {ev.mean:.3f} (quick 5-fold), worst fold {ev.worst_fold:.3f}"


def _report_cv(selected: list[tuple[int, Candidate]], u: Universe, spec: TaskSpec, mode: str,
               stored_ok: bool, alt: tuple[Universe, str] | None) -> None:
    """stored_ok: the stored dev-fold CVs describe these files. alt: the other industry setting,
    whose quick CV is shown next to the primary's on NEW DATA."""
    if mode == "none":
        print("CV skipped (--cv none)")
        return
    extra_note = " | _withextra: CV n/a (no labelled hold-out)" if u.extra is not None else ""
    first = selected[0][1]
    quick = {c.label for _, c in selected} if mode == "fast" else (set() if stored_ok else {first.label})
    for index, cand in selected:
        if cand.label in quick:
            text = _quick_cv(cand, u, spec)
        elif stored_ok:
            text = "CV unknown" if cand.cv_known is None else f"CV {cand.cv_known:.3f} (stored, dev folds)"
        else:
            text = "CV not run (use --cv fast)"
        print(f"{index:02d} {cand.label:<12} _hrtonly: {text}{extra_note}")
    if alt is not None and first.label in quick:
        print(f"{first.label} {alt[1]}: {_quick_cv(first, alt[0], spec)}")


def _run(args: argparse.Namespace) -> int:
    selected = _selected(args.only)
    data_dir = _data_dir(args)
    missing = [name for name in ARCADE_FILES if not (data_dir / name).is_file()]
    if missing:
        raise RunError("\n".join(f"missing {name} in {data_dir}" for name in missing))
    same = _identity(data_dir)
    d = _load(data_dir)
    spec = _spec(args, d, data_dir)
    base, filtered = _universe(args, d, spec, same)
    u = base if filtered else without_industry_filter(base)
    if not same:
        _print_facts(base, spec)
    moved = retire_stale_uploads(args.out, _stamp(data_dir, spec))
    if moved:
        print(f"moved {len(moved)} upload file(s) from an earlier run on other data or code to {moved[0].parent}")
    ok = _write_all(selected, u, spec, args.out)
    alt = ((without_industry_filter(base), "without industry filter") if filtered
           else (base, f"with the {base.groups.nunique()}-group industry filter"))
    stored_ok = same and filtered and args.groups in (None, DEFAULT_N_GROUPS)
    _report_cv(selected, u, spec, args.cv, stored_ok, None if same else alt)
    return 0 if ok else 1


def main(argv: Sequence[str] | None = None) -> int:
    args = _parser().parse_args(argv)
    start = time.perf_counter()
    try:
        code = _run(args)
    except RunError as exc:
        print(f"ERROR: {exc}")
        return 2
    print(f"done in {time.perf_counter() - start:.1f}s")
    return code
