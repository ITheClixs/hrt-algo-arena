"""R7 (25 Nov evening): the ideas left after the headroom probes (research/RESULTS.md, R7).

Pre-declared, each one change to the primary (r3_bundle_aug_clip2). Gate (README, section Validation): z >= 2.5 on the
dev folds, then the later-window check.
- var_target: sizing variance from the real target window. News density falls in later bars (0.187 headlines
  per bar in bars 50-59, 0.105 in 90-99), so the augmented windows understate the target window's variance.
- target_w3: ridge weight 3 on the real target window's rows.
- m2w0.75 / m2w1.5: the m^2 weight in the sizing, never re-tuned after augmentation.
- alpha10 / alpha200: calibration ridge strength, never re-tuned after augmentation (11x more rows).
- hl_more: residual-momentum half-lives 50 and 100 added.

    uv run python research/r7_target_window.py
"""

from __future__ import annotations

from common import OOF, SPEC, dev_universe, run_trial

from arena.confirm import later_window_check
from arena.generator import GeneratorModel
from arena.pipeline import PRIMARY
from arena.trials import load_evaluation
from arena.validation import compare

BEST = "r3_bundle_aug_clip2"
GRID: dict[str, dict] = {
    "r7_var_target": {"var_target": True},
    "r7_target_w3": {"target_weight": 3.0},
    "r7_m2w0.75": {"m2_weight": 0.75},
    "r7_m2w1.5": {"m2_weight": 1.5},
    "r7_alpha10": {"alpha": 10.0},
    "r7_alpha200": {"alpha": 200.0},
    "r7_hl_more": {"half_lives": (1, 2, 3, 5, 10, 25, 50, 100, 1000)},
}


def main() -> None:
    u = dev_universe()
    best = load_evaluation(BEST, OOF)
    passed = []
    for idea, change in GRID.items():
        cfg = {**PRIMARY, **change}
        ev = run_trial(idea, lambda c=cfg: GeneratorModel(SPEC, **c), cfg, vs=BEST)
        if compare(ev, best, u.y, SPEC).keep:
            passed.append((idea, cfg))
    for idea, cfg in passed:
        check = later_window_check(GeneratorModel(SPEC, **cfg), GeneratorModel(SPEC, **PRIMARY), u, SPEC)
        print(f"{idea} vs primary | {check}", flush=True)
    if not passed:
        print("R7: nothing passed the dev gate (z >= 2.5)", flush=True)


if __name__ == "__main__":
    main()
