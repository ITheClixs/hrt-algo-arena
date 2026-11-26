import json

import numpy as np
import pandas as pd

from arena.cv import signal_fit_predict
from arena.trials import TRIAL_COLUMNS, git_sha, load_evaluation, log_trial, save_evaluation
from arena.validation import compare, evaluate


def test_log_trial_appends_rows_with_fixed_columns(tmp_path, y):
    path = tmp_path / "trials.csv"
    rng = np.random.default_rng(0)
    base = evaluate("base", signal_fit_predict(y + rng.normal(0, y.std(), len(y))), y)
    cand = evaluate("cand", signal_fit_predict(y), y)
    log_trial(base, idea="base", params={"b": 2, "a": 1}, path=path)
    table = log_trial(cand, idea="cand", params={}, comparison=compare(cand, base, y), vs="base", path=path)
    assert tuple(table.columns) == TRIAL_COLUMNS and len(table) == 2
    assert table.loc[0, "params"] == json.dumps({"a": 1, "b": 2})
    assert bool(table.loc[1, "keep"]) and pd.isna(table.loc[0, "delta"])


def test_evaluation_roundtrip(tmp_path, y):
    ev = evaluate("x", signal_fit_predict(y), y)
    save_evaluation(ev, tmp_path)
    back = load_evaluation("x", tmp_path)
    assert back.seed_scores == ev.seed_scores and back.seeds == ev.seeds
    np.testing.assert_array_equal(back.oof, ev.oof)


def test_git_sha_is_hex_or_unknown():
    sha = git_sha()
    assert sha == "unknown" or (len(sha) >= 7 and set(sha) <= set("0123456789abcdef"))
