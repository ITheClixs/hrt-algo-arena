"""R3: time-shift augmentation of the calibration (plan Task 13)."""

from __future__ import annotations

from common import step
from r2_sizing import SECTOR_REF, build


def main() -> None:
    state = ("r0b_sector", {**SECTOR_REF, "extra_features": ()})
    for idea, offsets in (("r3_aug_every4", tuple(range(0, 41, 4))), ("r3_aug_all41", tuple(range(0, 41)))):
        state = step(state, idea, lambda c, o=offsets: {"augment_offsets": o}, build)
    print("R3 best:", state[0])


if __name__ == "__main__":
    main()


def bundle() -> None:
    """Pre-registered bundle of the two theory-backed near-misses."""
    state = ("r0b_sector", {**SECTOR_REF, "extra_features": ()})
    change = {"augment_offsets": tuple(range(0, 41, 4)), "clip_sd": 2.0}
    state = step(state, "r3_bundle_aug_clip2", lambda c: change, build)
    print("bundle best:", state[0])
