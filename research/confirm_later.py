"""Fresh-target confirmation gate (README, section Validation): candidate vs base on later windows.

    uv run python research/confirm_later.py volinv primary          # labels from arena.pipeline
    uv run python research/confirm_later.py aug sector --data data/raw

Prints the later-window Δ, its bootstrap s.e. and z, and the Δ per window offset.
"""

from __future__ import annotations

import argparse

from common import SPEC, dev_universe

from arena.confirm import later_window_check
from arena.pipeline import CANDIDATES


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("candidate")
    p.add_argument("base")
    args, _ = p.parse_known_args()
    by_label = {c.label: c for c in CANDIDATES}
    u = dev_universe()
    check = later_window_check(by_label[args.candidate].build(SPEC), by_label[args.base].build(SPEC), u, SPEC)
    print(f"{args.candidate} vs {args.base} | {check} | seeds {[round(d, 4) for d in check.delta_by_seed]}")


if __name__ == "__main__":
    main()
