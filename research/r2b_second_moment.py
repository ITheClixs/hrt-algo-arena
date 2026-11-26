"""R2 addendum (pre-registered): residual variance in E[r^2], and the m^2 weight."""

from __future__ import annotations

from common import step
from r2_sizing import SECTOR_REF, build


def main() -> None:
    state = ("r0b_sector", {**SECTOR_REF, "extra_features": ()})
    for idea, change in (("r2b_residual_var", lambda c: {"residual_var": True}),
                         ("r2b_m2w0.5", lambda c: {"m2_weight": 0.5}),
                         ("r2b_m2w2", lambda c: {"m2_weight": 2.0})):
        state = step(state, idea, change, build)
    print("R2b best:", state[0])


if __name__ == "__main__":
    main()
