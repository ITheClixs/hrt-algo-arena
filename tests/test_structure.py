import numpy as np
import pandas as pd

from arena.data import from_long
from arena.structure import cooccurrence_groups, keep_headlines, label_entities, mark_main_group

GROUP_A = ("Alvex Systems", "Borin Energy", "Corvel Labs", "Dravik Holdings")
GROUP_B = ("Elnor Devices", "Fennic Foods", "Gorvan Capital", "Hestra Retail Group")


def _two_group_market(n_sessions: int = 120, seed: int = 0):
    rng = np.random.default_rng(seed)
    bars, rows = [], []
    for sid in range(n_sessions):
        bars.append(pd.DataFrame({"session": sid, "bar_ix": np.arange(60), "open": 1.0,
                                  "high": 1.0, "low": 1.0, "close": 1.0}))
        home, away = (GROUP_A, GROUP_B) if sid % 2 else (GROUP_B, GROUP_A)
        for bar in rng.choice(60, size=8, replace=False):
            pool = home if rng.random() < 0.85 else away
            verb = rng.choice(["wins award", "misses estimates", "opens office"])
            rows.append((sid, int(bar), f"{pool[rng.integers(len(pool))]} {verb}"))
    hl = pd.DataFrame(rows, columns=["session", "bar_ix", "headline"])
    return from_long(pd.concat(bars, ignore_index=True), hl)


def test_groups_recover_planted_industries():
    s = _two_group_market()
    (labeled,) = label_entities(s)
    groups = cooccurrence_groups(labeled, n_groups=2)
    assert groups[list(GROUP_A)].nunique() == 1
    assert groups[list(GROUP_B)].nunique() == 1
    assert groups[GROUP_A[0]] != groups[GROUP_B[0]]


def test_main_group_flag_and_filter():
    s = _two_group_market()
    a, b = label_entities(s, s)
    assert (a.headlines["entity"] == b.headlines["entity"]).all()
    groups = cooccurrence_groups(a, b, n_groups=2)
    marked = mark_main_group(a, groups)
    assert 0.7 < marked.headlines["in_main"].mean() < 0.95
    kept = keep_headlines(marked, "in_main")
    assert kept.headlines["in_main"].all()
    assert len(kept.headlines) < len(marked.headlines)


def test_mark_main_group_survives_sessions_with_unknown_companies():
    s = _two_group_market()
    (labeled,) = label_entities(s)
    groups = cooccurrence_groups(labeled, n_groups=2)
    out = mark_main_group(labeled, groups.iloc[:0])  # no company has a known group
    assert not out.headlines["in_main"].any()


def test_estimate_n_groups_finds_the_planted_industry_count():
    from arena.data import from_long
    from arena.structure import estimate_n_groups
    from arena.synth import make_synthetic

    for k in (3, 4, 6):
        m = make_synthetic(n_sessions=600, seed=k, n_industries=k)
        (labeled,) = label_entities(from_long(m.bars, m.headlines))
        n, gap = estimate_n_groups(labeled)
        assert n == k and gap > 0.2
