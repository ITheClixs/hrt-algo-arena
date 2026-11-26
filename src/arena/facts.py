"""Generator facts the research found on the known dataset, re-measured on whatever data arrives.

On NEW DATA the runner prints these so the team can see at a glance whether the plan's
assumptions still hold. Everything is label-free.
"""

from __future__ import annotations

import numpy as np

from arena.data import concat_sessions, disjoint_from, truncate
from arena.generator import gk_vol, lag_profile
from arena.universe import Universe, main_only


def _push_duration(increments: np.ndarray, share: float = 0.4) -> int:
    """Longest run of consecutive lags whose push is at least `share` of the peak (robust to noise)."""
    peak = increments.max()
    if peak <= 0:
        return 0
    best = run = 0
    for strong in increments >= share * peak:
        run = run + 1 if strong else 0
        best = max(best, run)
    return best


def generator_facts(u: Universe, halfway: int = 49) -> dict[str, float]:
    oos = disjoint_from(truncate(main_only(u.oos), halfway), u.ins)  # ids may overlap the in-sample ones
    visible = concat_sessions(truncate(main_only(u.ins), halfway), oos)
    increments = np.diff(np.concatenate([[0.0], lag_profile(visible, key="key", max_lag=15)]))
    duration = _push_duration(increments)
    vol = gk_vol(visible, halfway)
    hl = visible.headlines
    early = hl[hl["bar_ix"] <= halfway - 10]
    move = np.abs(np.log(visible.close[early["sidx"], np.minimum(early["bar_ix"] + 9, halfway)]
                         / visible.close[early["sidx"], np.maximum(early["bar_ix"] - 1, 0)]))
    q = np.quantile(vol, [0.2, 0.8])
    low, high = vol[early["sidx"]] <= q[0], vol[early["sidx"]] >= q[1]
    ratio = float(move[high].mean() / move[low].mean()) if low.any() and high.any() else float("nan")
    return {
        "n_templates": float(u.ins.headlines["key"].nunique()),
        "n_types": float(u.ins.headlines["type"].nunique()),
        "n_groups": float(u.groups.nunique()),
        "main_share": float(u.ins.headlines["in_main"].mean()),
        "push_duration": float(duration),
        "move_high_vs_low_vol": ratio,
    }
