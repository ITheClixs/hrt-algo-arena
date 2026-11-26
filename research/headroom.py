"""Headroom: how good can a headline-drift model get on this data? (optimistic by design)

Rates are fitted on ALL in-sample sessions, including the ones being scored, so these numbers
are upper bounds, not CV results. 'with future headlines' also uses headlines from bars 50-58,
which nobody can see at bar 49: it is the ceiling of the whole approach.
"""

from __future__ import annotations

from common import SPEC, dev_universe

from arena.events import _per_session, fit_drift_rates, remaining_drift
from arena.features import parkinson_vol
from arena.metrics import score
from arena.universe import main_only


def main() -> None:
    u = dev_universe()
    vol = parkinson_vol(u.ins.high[:, :50], u.ins.low[:, :50])
    for label, s in (("all headlines", u.ins), ("main industry", main_only(u.ins))):
        rates = fit_drift_rates(s, duration=10)
        owed = remaining_drift(s, rates, SPEC.halfway_bar, SPEC.horizon)
        future = _per_session(s, rates, "key", 58, 50, 59, 10)
        for p in (0, 1, 2):
            print(f"{label:14s} vol^{p}: owed by bars<=49 {score(owed / vol**p, u.y):.3f} | "
                  f"with future headlines {score(future / vol**p, u.y):.3f}")


if __name__ == "__main__":
    main()
