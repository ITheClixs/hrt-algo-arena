"""Time-shift augmentation: re-use the in-sample sessions' later bars as extra training windows.

A labelled session has all 100 bars, so the same problem (see 50 bars, predict the next 10)
can be posed at decision bars 49, 50, ..., 89. Window `offset` covers bars offset..offset+59,
re-indexed from 0, so models see exactly the shape they see at prediction time. Windows of a
session stay with that session (`groups`), which keeps them inside its CV fold.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import replace

import numpy as np

from arena.config import ARCADE_ALGO_ARENA, TaskSpec
from arena.data import Sessions, concat_sessions, forward_returns


def shift_window(s: Sessions, offset: int, n_visible: int = 50, horizon: int = 10) -> Sessions:
    """Bars offset..offset+n_visible+horizon-1 of every session, re-indexed from 0."""
    end = offset + n_visible + horizon
    if offset < 0 or end > s.n_bars:
        raise ValueError(f"window {offset}..{end - 1} does not fit in {s.n_bars} bars")
    hl = s.headlines
    inside = (hl["bar_ix"] >= offset) & (hl["bar_ix"] < end)
    moved = hl[inside].assign(bar_ix=hl.loc[inside, "bar_ix"] - offset).reset_index(drop=True)
    sl = slice(offset, end)
    return replace(s, ids=s.ids * 100 + offset, open=s.open[:, sl], high=s.high[:, sl],
                   low=s.low[:, sl], close=s.close[:, sl], headlines=moved)


def fitting_offsets(n_bars: int, offsets: Sequence[int], spec: TaskSpec = ARCADE_ALGO_ARENA) -> tuple[int, ...]:
    """The offsets whose window fits in n_bars, in the order `augmented` stacks them."""
    n_visible = spec.halfway_bar + 1
    return tuple(o for o in offsets if 0 <= o and o + n_visible + spec.horizon <= n_bars)


def augmented(s: Sessions, offsets: Sequence[int],
              spec: TaskSpec = ARCADE_ALGO_ARENA) -> tuple[Sessions, np.ndarray, np.ndarray]:
    """Stacked windows, their targets (demeaned within each offset) and each window's session row.

    Offsets whose window would run past the last bar (e.g. a longer horizon) are skipped."""
    n_visible = spec.halfway_bar + 1
    fitting = fitting_offsets(s.n_bars, offsets, spec)
    if not fitting:
        raise ValueError(f"no window offset in {tuple(offsets)} fits {s.n_bars} bars "
                         f"with {n_visible} visible + {spec.horizon} held")
    windows = [shift_window(s, o, n_visible, spec.horizon) for o in fitting]
    y = np.concatenate([forward_returns(w, spec) for w in windows])
    groups = np.tile(np.arange(s.n), len(windows))
    return concat_sessions(*windows), y, groups
