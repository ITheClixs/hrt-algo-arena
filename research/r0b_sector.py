"""R0(b): Edinburgh-style sector model (joint template fit, residual momentum, E[r]/E[r^2])."""

from __future__ import annotations

from common import SPEC, run_trial

from arena.generator import SECTOR_REF, GeneratorModel


def main() -> None:
    run_trial("r0b_sector", lambda: GeneratorModel(SPEC, **SECTOR_REF), SECTOR_REF, vs="r0a_baseline")
    vol_units = {**SECTOR_REF, "units": "vol"}
    run_trial("r0b_sector_volunits", lambda: GeneratorModel(SPEC, **vol_units), vol_units,
              vs="r0b_sector")


if __name__ == "__main__":
    main()
