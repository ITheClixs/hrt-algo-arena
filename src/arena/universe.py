"""One preprocessing pass shared by every arcade model: companies, templates and industries.

Everything here is label-free and decided from the visible bars (<= halfway) of every part, so
in-sample, out-of-sample and extra sessions are tagged the same way:
  * `entity`  - company name, detected jointly over all headlines (consistent across parts)
  * `key`     - event template: the headline with the company and the numbers masked
  * `type`    - coarser event type: slot phrases (sector, region, partner, ...) masked too
  * `in_main` - the headline's company belongs to the session's dominant industry
All headlines are kept; models choose whether to use only `main_only(s)`.
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from pathlib import Path

import numpy as np
import pandas as pd

from arena.config import ARCADE_ALGO_ARENA, TaskSpec
from arena.data import Sessions, disjoint_from, load_arcade, load_extra_sessions, with_column
from arena.structure import (
    cooccurrence_groups,
    estimate_n_groups,
    keep_headlines,
    label_entities,
    mark_main_group,
)
from arena.text import coarse_types, templates

DEFAULT_N_GROUPS = 5  # the known data: 80 companies in 5 industries of 16
WEAK_GAP = 0.2  # eigengap below this: no clear industry structure (the known data: 0.48)


@dataclass(frozen=True)
class Universe:
    ins: Sessions  # labelled sessions, all bars
    oos: Sessions  # sessions to predict, visible bars only
    y: np.ndarray  # official in-sample returns, aligned to ins.ids
    groups: pd.Series  # company -> industry
    extra: Sessions | None = None  # datathon train sessions (all bars), if loaded
    y_extra: np.ndarray | None = None


def _with_keys(s: Sessions) -> Sessions:
    hl = s.headlines
    keys = templates(hl["headline"], entities=hl["entity"])
    return with_column(with_column(s, "key", keys.to_numpy()), "type", coarse_types(keys).to_numpy())


def build_universe(ins: Sessions, oos: Sessions, y, extra: Sessions | None = None,
                   y_extra=None, n_groups: int | None = 5, halfway: int = 49) -> Universe:
    """Tag every part. Companies and industries come from the HRT sessions alone, so loading
    extra sessions never changes an `_hrtonly` file; extra sessions are mapped onto them.
    `n_groups=None` takes the industry count from the eigengap (`estimate_n_groups`), or
    DEFAULT_N_GROUPS when the gap is too weak to mean anything."""
    if (extra is None) != (y_extra is None):
        raise ValueError("pass extra sessions and their returns together")
    labelled = label_entities(ins, oos)
    if n_groups is None:
        estimate, gap = estimate_n_groups(*labelled, last_bar=halfway)
        n_groups = estimate if gap >= WEAK_GAP else DEFAULT_N_GROUPS
    groups = cooccurrence_groups(*labelled, n_groups=n_groups, last_bar=halfway)
    if extra is not None:
        extra = disjoint_from(extra, ins, oos)
        labelled = (*labelled, label_entities(ins, oos, extra)[2])
    done = tuple(_with_keys(mark_main_group(p, groups, last_bar=halfway)) for p in labelled)
    return Universe(
        ins=done[0], oos=done[1], y=np.asarray(y, dtype=float), groups=groups,
        extra=done[2] if extra is not None else None,
        y_extra=None if y_extra is None else np.asarray(y_extra, dtype=float),
    )


def load_universe(data_dir: str | Path, extra_dir: str | Path | None = None,
                  spec: TaskSpec = ARCADE_ALGO_ARENA, n_groups: int | None = 5) -> Universe:
    """Arcade files (+ optional datathon train sessions) -> tagged `Universe`."""
    d = load_arcade(data_dir)
    extra, y_extra = (None, None) if extra_dir is None else load_extra_sessions(extra_dir, spec)
    return build_universe(d.ins, d.oos, d.y, extra, y_extra, n_groups=n_groups,
                          halfway=spec.halfway_bar)


def without_industry_filter(u: Universe) -> Universe:
    """Every headline counts as main-industry: for draws without (or with doubtful) industries."""
    def all_main(s: Sessions | None) -> Sessions | None:
        return None if s is None else with_column(s, "in_main", np.ones(len(s.headlines), dtype=bool))

    return replace(u, ins=all_main(u.ins), oos=all_main(u.oos), extra=all_main(u.extra))


def main_only(s: Sessions) -> Sessions:
    """Only the headlines about the session's main industry (the ones that move its price)."""
    return keep_headlines(s, "in_main")
