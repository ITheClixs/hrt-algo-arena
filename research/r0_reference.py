"""R0(a): the quickstart model on the development folds (the first reference)."""

from __future__ import annotations

from common import SPEC, run_trial

from arena.references import BaselineDrift


def main() -> None:
    run_trial("r0a_baseline", lambda: BaselineDrift(SPEC), {"model": "BaselineDrift", "vol_power": 1.0})


if __name__ == "__main__":
    main()
