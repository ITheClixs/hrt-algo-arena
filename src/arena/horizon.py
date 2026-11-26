"""Read the task off the files: the trading bar, the exit bar and how the returns are defined.

HRT states the horizon in its notebook and in `sessions.py`, but the files are what gets scored.
`detect_horizon` takes the halfway bar from the out-of-sample sessions (the last bar they show)
and finds the exit bar whose close-to-close return reproduces `ins_returns`, demeaned or not.
`resolve_spec` turns that into a `TaskSpec` and refuses flags that contradict the files, so a
changed horizon can never silently produce a mis-timed upload.
"""

from __future__ import annotations

import re
from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np

from arena.config import TaskSpec
from arena.data import ArcadeData

MIN_CORR = 0.9999  # a matching exit bar reproduces ins_returns up to rounding
_CONSTANT = re.compile(r"^(HALFWAY_BAR|HORIZON_BARS)\s*=\s*(\d+)", re.MULTILINE)


@dataclass(frozen=True)
class Horizon:
    halfway: int  # last bar the out-of-sample sessions show
    exit: int | None  # bar whose close(halfway) -> close(exit) return is ins_returns; None if none is
    demeaned: bool  # ins_returns are demeaned across sessions
    corr: float  # correlation of the best candidate bar's return with ins_returns
    best_bar: int | None  # best candidate exit bar, even when it does not match

    @property
    def ok(self) -> bool:
        return self.exit is not None

    def describe(self) -> str:
        if self.ok:
            kind = "demeaned" if self.demeaned else "raw"
            return (f"trade at close of bar {self.halfway}, exit at close of bar {self.exit} "
                    f"(ins_returns = {kind} close-to-close return, corr {self.corr:.6f})")
        best = "" if self.best_bar is None else f"; best bar {self.best_bar} has corr {self.corr:.4f}"
        return (f"no exit bar reproduces ins_returns from close of bar {self.halfway}{best}. "
                "The target is defined differently: read the challenge page and sessions.py")


def detect_horizon(d: ArcadeData) -> Horizon:
    halfway = d.oos.n_bars - 1
    y = np.asarray(d.y, dtype=float)
    best = (-np.inf, None, True)
    for exit_bar in range(halfway + 1, d.ins.n_bars):
        r = d.ins.close[:, exit_bar] / d.ins.close[:, halfway] - 1.0
        if np.std(r) == 0:
            continue
        corr = float(np.corrcoef(r, y)[0, 1])
        if corr > best[0]:
            demeaned = np.abs(r - r.mean() - y).max() <= np.abs(r - y).max()
            best = (corr, exit_bar, bool(demeaned))
    corr, bar, demeaned = best
    if bar is None:
        return Horizon(halfway=halfway, exit=None, demeaned=True, corr=float("nan"), best_bar=None)
    matched = bar if corr >= MIN_CORR else None
    return Horizon(halfway=halfway, exit=matched, demeaned=demeaned, corr=corr, best_bar=bar)


def resolve_spec(base: TaskSpec, d: ArcadeData, halfway: int | None = None,
                 exit_bar: int | None = None) -> TaskSpec:
    """`base` with the horizon the files define; flags must agree with the files."""
    h = detect_horizon(d)
    if not h.ok:
        if halfway is None or exit_bar is None:
            raise ValueError(h.describe() + ". To override, pass both --halfway and --exit.")
        return replace(base, halfway_bar=halfway, exit_bar=exit_bar)
    for name, flag, found in (("halfway", halfway, h.halfway), ("exit", exit_bar, h.exit)):
        if flag is not None and flag != found:
            raise ValueError(f"--{name} {flag} contradicts the files ({h.describe()}). "
                             f"Drop the flag or check the data folder.")
    return replace(base, halfway_bar=h.halfway, exit_bar=h.exit, demean_returns=h.demeaned)


def scorer_constants(path: str | Path) -> dict[str, int]:
    """HALFWAY_BAR and HORIZON_BARS as HRT's sessions.py declares them ({} if absent)."""
    path = Path(path)
    if not path.is_file():
        return {}
    return {name: int(value) for name, value in _CONSTANT.findall(path.read_text(errors="replace"))}
