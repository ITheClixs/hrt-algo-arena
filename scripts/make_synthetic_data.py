"""Write a synthetic dataset in the arcade format to data/synthetic (works offline).

    uv run python scripts/make_synthetic_data.py [n_sessions]
"""

from __future__ import annotations

import sys

from arena.paths import DATA
from arena.synth import make_synthetic, write_arcade_files


def main() -> None:
    n = int(sys.argv[1]) if len(sys.argv) > 1 else 4000
    out = write_arcade_files(make_synthetic(n_sessions=n, seed=7), DATA / "synthetic")
    sys.stdout.write(f"wrote {n} synthetic sessions to {out}\n")


if __name__ == "__main__":
    main()
