import numpy as np

from arena.universe import load_universe, main_only


def test_load_universe_tags_templates_entities_and_main_industry(arcade_dir):
    u = load_universe(arcade_dir)
    for s in (u.ins, u.oos):
        assert {"key", "entity", "in_main"} <= set(s.headlines.columns)
    assert u.groups.nunique() == 5 and u.extra is None and len(u.y) == u.ins.n


def test_load_universe_with_extra(arcade_dir, datathon_dir):
    u = load_universe(arcade_dir, extra_dir=datathon_dir)
    assert u.extra.n == 100 and len(u.y_extra) == 100 and "in_main" in u.extra.headlines.columns


def test_main_only_keeps_main_industry_headlines(arcade_dir):
    u = load_universe(arcade_dir)
    m = main_only(u.ins)
    assert m.headlines["in_main"].all() and len(m.headlines) <= len(u.ins.headlines)
    assert np.array_equal(m.ids, u.ins.ids)


def test_extra_sessions_never_change_the_hrt_sessions_tags(arcade_dir, tmp_path):
    """Extra sessions from another draw (other companies, colliding ids) must not move the
    industries, and so must not change any _hrtonly file."""
    import pandas as pd

    from arena.synth import make_synthetic

    other = make_synthetic(n_sessions=150, seed=9, start_id=1000, n_industries=3)
    folder = tmp_path / "other_draw"
    folder.mkdir()
    bars, hl = other.bars, other.headlines[["session", "bar_ix", "headline"]]
    for kind, part in (("seen", bars.bar_ix <= 49), ("unseen", bars.bar_ix > 49)):
        bars[part].to_parquet(folder / f"bars_{kind}_train.parquet", index=False)
    for kind, part in (("seen", hl.bar_ix <= 49), ("unseen", hl.bar_ix > 49)):
        hl[part].to_parquet(folder / f"headlines_{kind}_train.parquet", index=False)
    alone, joint = load_universe(arcade_dir), load_universe(arcade_dir, extra_dir=folder)
    pd.testing.assert_series_equal(alone.groups, joint.groups)
    for a, b in ((alone.ins, joint.ins), (alone.oos, joint.oos)):
        pd.testing.assert_frame_equal(a.headlines, b.headlines)
    taken = set(joint.ins.ids) | set(joint.oos.ids)
    assert joint.extra.n == 150 and not taken & set(joint.extra.ids)
    assert not joint.extra.headlines["in_main"].any()  # none of its companies is in a known industry


def test_universe_takes_the_industry_count(arcade_dir):
    assert load_universe(arcade_dir, n_groups=3).groups.nunique() == 3
