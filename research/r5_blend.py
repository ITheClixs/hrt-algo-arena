"""R5: blend of the current best generator model with the best R4 learners (plan Task 15).

Weights over (generator, owed-push TF-IDF, template-age MLP) come from the pre-declared grid.
"""

from __future__ import annotations

import warnings

from common import SPEC, run_trial
from r2_sizing import SECTOR_REF
from sklearn.exceptions import ConvergenceWarning

from arena.blend import BlendModel
from arena.generator import GeneratorModel
from arena.learners import OwedTfidfModel, TemplateAgeMLP

BEST = "r3_bundle_aug_clip2"
EVERY4 = tuple(range(0, 41, 4))
GEN = {**SECTOR_REF, "augment_offsets": EVERY4, "clip_sd": 2.0}
GRID = ((1, 1, 0), (1, 0, 1), (1, 1, 1), (2, 1, 1), (3, 1, 1))


def members():
    return (GeneratorModel(SPEC, **GEN), OwedTfidfModel(SPEC, alpha=3000.0, augment_offsets=EVERY4),
            TemplateAgeMLP(SPEC, hidden=(64,)))


def main() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for weights in GRID:
        name = "r5_blend_" + "_".join(str(w) for w in weights)
        run_trial(name, lambda w=weights: BlendModel(members(), w), {"weights": weights, "generator": GEN}, vs=BEST)


if __name__ == "__main__":
    main()
