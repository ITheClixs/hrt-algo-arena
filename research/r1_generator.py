"""R1: generator-fit ladder (plan Task 11). Each step is compared with the current best and kept
only if `compare(...).keep`; kept steps carry forward.

    uv run python research/r1_generator.py          # run the ladder
    uv run python research/r1_generator.py --vocab  # list slot phrases used by coarse_types
"""

from __future__ import annotations

import collections
import sys
from collections.abc import Callable

from common import OOF, SPEC, dev_universe, run_trial

from arena.data import concat_sessions, truncate
from arena.generator import KERNEL_FLAT10, SECTOR_REF, GeneratorModel, lag_profile, measured_kernel
from arena.trials import load_evaluation
from arena.universe import main_only
from arena.validation import compare

State = tuple[str, dict]


def print_vocab() -> None:
    """Phrases that fill the same slot inside otherwise identical templates."""
    shapes = collections.defaultdict(set)
    for k in dev_universe().ins.headlines["key"].unique():
        w = k.split()
        for i in range(len(w)):
            for j in range(i + 1, min(len(w), i + 5) + 1):
                shapes[(tuple(w[:i]), tuple(w[j:]))].add(" ".join(w[i:j]))
    counts = collections.Counter(tuple(sorted(f)) for (pre, suf), f in shapes.items() if len(f) >= 4 and (pre or suf))
    for fills, c in counts.most_common(12):
        print(c, "|", " ; ".join(fills))


def measured() -> tuple[float, ...]:
    """Kernel from the label-free lag profile (bars 0-49 of in-sample and out-of-sample sessions)."""
    u = dev_universe()
    visible = concat_sessions(truncate(main_only(u.ins), SPEC.halfway_bar), main_only(u.oos))
    kernel = measured_kernel(lag_profile(visible, key="key"))
    print("measured kernel (lags 0..10):", [round(k, 3) for k in kernel])
    return kernel


def step(state: State, idea: str, change: Callable[[dict], dict]) -> State:
    best, cfg = state
    new = {**cfg, **change(cfg)}
    ev = run_trial(idea, lambda: GeneratorModel(SPEC, **new), new, vs=best)
    keep = compare(ev, load_evaluation(best, OOF), dev_universe().y, SPEC).keep
    return (idea, new) if keep else state


def ladder() -> State:
    state: State = ("r0b_sector", {**SECTOR_REF, "extra_features": ()})
    kernel = measured()
    rungs: list[tuple[str, Callable[[dict], dict]]] = [
        ("r1_kernel_flat10", lambda c: {"kernel": KERNEL_FLAT10}),
        ("r1_kernel_measured", lambda c: {"kernel": kernel}),
        ("r1_type", lambda c: {"key": "type"}),
        ("r1_lam50", lambda c: {"lam": 50.0}),
        ("r1_lam800", lambda c: {"lam": 800.0}),
        ("r1_no_unlabeled", lambda c: {"use_unlabeled": False}),
        ("r1_hl_none", lambda c: {"half_lives": ()}),
        ("r1_hl_short", lambda c: {"half_lives": (1, 3, 10, 1000)}),
    ]
    rungs += [(f"r1_x_{name}", lambda c, n=name: {"extra_features": tuple(c["extra_features"]) + (n,)})
              for name in ("news_count", "live_count", "conflict", "repeat_push", "peer_push")]
    rungs += [("r1_alpha10", lambda c: {"alpha": 10.0}), ("r1_alpha200", lambda c: {"alpha": 200.0})]
    for idea, change in rungs:
        state = step(state, idea, change)
    print("R1 best:", state[0], state[1])
    return state


if __name__ == "__main__":
    print_vocab() if "--vocab" in sys.argv else ladder()
