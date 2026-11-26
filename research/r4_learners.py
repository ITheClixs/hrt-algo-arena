"""R4: diverse learners as standalone signals (plan Task 14). Their blend value is decided in R5.

    uv run python research/r4_learners.py tfidf
    uv run python research/r4_learners.py mlp
"""

from __future__ import annotations

import sys
import warnings

from common import SPEC, run_trial
from sklearn.exceptions import ConvergenceWarning

from arena.learners import OwedTfidfModel, TemplateAgeMLP

BEST = "r3_bundle_aug_clip2"
EVERY4 = tuple(range(0, 41, 4))


def tfidf() -> None:
    for alpha in (1000.0, 3000.0, 10000.0):
        for name, offsets in (("", ()), ("_aug", EVERY4)):
            params = {"alpha": alpha, "augment_offsets": offsets}
            run_trial(f"r4_tfidf_a{alpha:g}{name}", lambda p=params: OwedTfidfModel(SPEC, **p), params, vs=BEST)


def mlp() -> None:
    warnings.filterwarnings("ignore", category=ConvergenceWarning)
    for hidden in ((32,), (64,)):
        params = {"hidden": hidden}
        run_trial(f"r4_mlp_h{hidden[0]}", lambda p=params: TemplateAgeMLP(SPEC, **p), params, vs=BEST)


if __name__ == "__main__":
    {"tfidf": tfidf, "mlp": mlp}[sys.argv[1] if len(sys.argv) > 1 else "tfidf"]()
