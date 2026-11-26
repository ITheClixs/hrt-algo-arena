import numpy as np
import pandas as pd


def test_synthetic_default_is_unchanged():
    from arena.synth import make_synthetic
    m = make_synthetic(n_sessions=20, seed=7)
    assert len(m.bars) == 2000 and m.headlines["headline"].str.split().str[0].isin(
        ["Alvex", "Borin", "Corvel", "Dravik", "Elnor", "Fennic", "Gorvan", "Hestra", "Ivrel", "Jorvik", "Kelmar",
         "Lunex"]).all()


def test_synthetic_industries_are_recovered(tmp_path):
    from sklearn.metrics import adjusted_rand_score

    from arena.synth import make_synthetic, write_arcade_files
    from arena.universe import load_universe
    m = make_synthetic(n_sessions=600, seed=4, n_industries=5, vol_scaled=True)
    u = load_universe(write_arcade_files(m, tmp_path / "ind"))
    truth = pd.Series(u.groups.index).str.extract(r"^(\w+)")[0]
    assert adjusted_rand_score(truth, u.groups.to_numpy()) > 0.9


def test_generator_facts_find_the_planted_duration(arcade_dir):
    from arena.facts import generator_facts
    from arena.universe import load_universe
    facts = generator_facts(load_universe(arcade_dir))
    assert 9 <= facts["push_duration"] <= 11 and facts["n_groups"] == 5
    assert np.isfinite(list(facts.values())).all()
