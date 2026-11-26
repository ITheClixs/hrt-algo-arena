"""Hidden structure in the synthetic universe: company groups ("industries").

In the public Algo Arena data the 80 companies split into 5 groups of 16 that show up together
in sessions. Each session is built around one group; headlines about companies outside it
barely move the price. Keeping only main-group headlines lifted CV Sharpe 3.05 -> 3.35.
Everything here is label-free (uses only which companies appear in the visible bars).
"""

from __future__ import annotations

from dataclasses import replace

import numpy as np
import pandas as pd
from sklearn.cluster import SpectralClustering

from arena.data import Sessions, with_column
from arena.text import detect_entities


def label_entities(*sessions: Sessions, col: str = "entity") -> tuple[Sessions, ...]:
    """Detect company names on all headline tables jointly (consistent across ins/oos)."""
    lengths = [len(s.headlines) for s in sessions]
    every = pd.concat([s.headlines["headline"] for s in sessions], ignore_index=True)
    entities = detect_entities(every).to_numpy()
    bounds = np.cumsum([0, *lengths])
    return tuple(
        with_column(s, col, entities[a:b])
        for s, a, b in zip(sessions, bounds[:-1], bounds[1:], strict=True)
    )


def _lift(*sessions: Sessions, last_bar: int, col: str) -> pd.DataFrame:
    """Entity x entity lift: how much more often two companies share a session than by chance."""
    frames = [
        s.headlines.loc[s.headlines["bar_ix"] <= last_bar, ["sidx", col]].assign(src=i)
        for i, s in enumerate(sessions)
    ]
    seen = pd.concat(frames, ignore_index=True)
    presence = pd.crosstab([seen["src"], seen["sidx"]], seen[col]).gt(0)
    m = presence.to_numpy(dtype=float)
    co = m.T @ m
    np.fill_diagonal(co, 0.0)
    p = m.mean(axis=0)
    lift = co / (np.outer(p, p) * len(m) + 1e-12)
    return pd.DataFrame(lift, index=presence.columns, columns=presence.columns)


def cooccurrence_groups(*sessions: Sessions, n_groups: int = 5, last_bar: int = 49,
                        col: str = "entity", seed: int = 0) -> pd.Series:
    """Cluster entities by how often they appear in the same session (lift matrix)."""
    lift = _lift(*sessions, last_bar=last_bar, col=col)
    labels = SpectralClustering(n_groups, affinity="precomputed", random_state=seed).fit_predict(lift.to_numpy())
    return pd.Series(labels, index=lift.index, name="group")


def estimate_n_groups(*sessions: Sessions, last_bar: int = 49, col: str = "entity",
                      max_k: int = 12) -> tuple[int, float]:
    """Number of industries suggested by the eigengap of the normalised lift matrix, and that gap.

    The known data gives 5 with a gap of about 0.48; a gap below ~0.2 means no clear structure."""
    lift = _lift(*sessions, last_bar=last_bar, col=col).to_numpy()
    degree = lift.sum(axis=1)
    norm = lift / np.sqrt(np.outer(degree, degree) + 1e-12)
    ev = np.sort(np.linalg.eigvalsh(norm))[::-1]
    k = min(max_k, len(ev) - 1)
    if k < 2:
        return 1, 0.0
    gaps = ev[1:k] - ev[2 : k + 1]  # gap after the 2nd, 3rd, ... eigenvalue
    best = int(np.argmax(gaps))
    return best + 2, float(gaps[best])


def _dominant(groups: pd.Series) -> float:
    counts = groups.value_counts()
    return counts.idxmax() if len(counts) else np.nan


def mark_main_group(s: Sessions, groups: pd.Series, last_bar: int = 49, col: str = "entity",
                    out: str = "in_main") -> Sessions:
    """Flag headlines whose company belongs to the session's dominant group (decided from
    the visible bars only). Companies without a group never count as main."""
    hl = s.headlines
    group = hl[col].map(groups)
    seen = hl.assign(group=group)[hl["bar_ix"] <= last_bar].drop_duplicates(["sidx", col])
    main = seen.groupby("sidx")["group"].agg(_dominant)
    flag = group.to_numpy() == main.reindex(hl["sidx"]).to_numpy()
    return with_column(s, out, flag)


def keep_headlines(s: Sessions, col: str) -> Sessions:
    """Drop headlines where boolean column `col` is False."""
    return replace(s, headlines=s.headlines[s.headlines[col].astype(bool)].reset_index(drop=True))
