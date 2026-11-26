"""R0(c): the 7 Nov winner's model, re-implemented (drift, counts, peers, bar-based sizing)."""

from __future__ import annotations

from common import SPEC, run_trial

from arena.references import WinnerRef


def main() -> None:
    run_trial("r0c_winner", lambda: WinnerRef(SPEC), {"clip_sd": 2.0, "power": 0.5, "use_unlabeled": True},
              vs="r0a_baseline")


if __name__ == "__main__":
    main()
