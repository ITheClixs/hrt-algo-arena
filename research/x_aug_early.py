"""Review caveat (25 Nov): augmentation windows at offset >= 30 have less news, so they are less like
the real task. One pre-declared variant: the primary with offsets 0..28 (every 4th) instead of 0..40.

    uv run python research/x_aug_early.py

Gate (README, section Validation): dev-fold z >= 2.5 and a positive later-window delta.
"""

from __future__ import annotations

from common import SPEC, dev_universe, run_trial

from arena.confirm import later_window_check
from arena.generator import SECTOR_REF, GeneratorModel
from arena.pipeline import PRIMARY

EARLY = {**SECTOR_REF, "augment_offsets": tuple(range(0, 29, 4)), "clip_sd": 2.0}


def main() -> None:
    run_trial("x_aug_early28", lambda: GeneratorModel(SPEC, **EARLY), EARLY, vs="r3_bundle_aug_clip2")
    check = later_window_check(GeneratorModel(SPEC, **EARLY), GeneratorModel(SPEC, **PRIMARY), dev_universe(), SPEC)
    print(f"x_aug_early28 vs primary | {check}", flush=True)


if __name__ == "__main__":
    main()
