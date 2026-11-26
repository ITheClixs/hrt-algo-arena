"""R2: sizing ladder (plan Task 12) plus the two pre-registered R1 add-ons (outsider push, surprise).

    uv run python research/r2_sizing.py
"""

from __future__ import annotations

from common import SPEC, State, step

from arena.generator import SECTOR_REF, GeneratorModel


def build(cfg: dict) -> GeneratorModel:
    return GeneratorModel(SPEC, **cfg)


def ladder(start: State) -> State:
    state = start
    rungs = [
        ("r1b_outsider", lambda c: {"extra_features": tuple(c["extra_features"]) + ("outsider_push",)}),
        ("r1b_surprise", lambda c: {"extra_features": tuple(c["extra_features"]) + ("surprise",)}),
        ("r2_risk_model", lambda c: {"sizing": "risk_model"}),
    ]
    rungs += [(f"r2_vol_p{p}", lambda c, p=p: {"sizing": "vol", "vol_power": p}) for p in (0.5, 1.0, 1.25, 1.5)]
    rungs += [(f"r2_clip{k:g}", lambda c, k=k: {"clip_sd": k}) for k in (2.0, 3.0)]
    for idea, change in rungs:
        state = step(state, idea, change, build)
    print("R2 best:", state[0], state[1])
    return state


if __name__ == "__main__":
    ladder(("r0b_sector", {**SECTOR_REF, "extra_features": ()}))
