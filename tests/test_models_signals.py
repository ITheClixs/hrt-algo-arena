import numpy as np
import pandas as pd
import pytest

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.cv import cross_validate, kfold
from arena.data import subset, truncate
from arena.models import DriftModel, TfidfModel, as_fit_predict
from arena.signals import (
    blend_cv,
    blend_predict,
    export_signal,
    leaderboard,
    load_signals,
    save_signal,
    team_folds,
)


def test_drift_model_finds_the_planted_signal(keyed, y):
    res = cross_validate(as_fit_predict(DriftModel(SPEC), keyed, y), y, kfold(keyed.n, 4))
    assert res.mean > 2.0


def test_drift_model_never_peeks_past_halfway(keyed, y):
    fitted = DriftModel(SPEC, vol_power=1.0).fit(subset(keyed, np.arange(300)), y[:300])
    test = subset(keyed, np.arange(300, 400))
    np.testing.assert_allclose(fitted.predict(test), fitted.predict(truncate(test, 49)))


def test_tfidf_model_runs(keyed, y):
    fitted = TfidfModel(SPEC, min_df=1).fit(keyed, y)
    assert fitted.predict(keyed).shape == (keyed.n,)


def test_team_folds_are_deterministic():
    a, b = team_folds(50), team_folds(50)
    assert all((x[1] == z[1]).all() for x, z in zip(a, b, strict=True))


def test_signal_roundtrip_and_blend(tmp_path, keyed, y):
    ids = keyed.ids
    rng = np.random.default_rng(0)
    good, noise = y + rng.normal(0, y.std(), len(y)), rng.normal(size=len(y))
    save_signal(tmp_path, "alice", "good", ids, good, [1, 2], [0.1, -0.2])
    save_signal(tmp_path, "bob", "noise", ids, noise, [1, 2], [0.3, 0.0])
    ins, oos = load_signals(tmp_path)
    assert list(ins.columns) == ["alice/good", "bob/noise"]
    table = leaderboard(ins, pd.Series(y, index=ids))
    assert table.index[0] == "alice/good"
    assert blend_cv(ins, y).mean > 5
    assert blend_predict(ins, oos, y).shape == (2,)


def test_export_signal_uses_team_folds(tmp_path, keyed, y):
    fp = as_fit_predict(DriftModel(SPEC), keyed, y)
    path = export_signal(tmp_path, "carol", "drift", fp, keyed.ids, [5], [0.0])
    assert path.exists()


def test_save_signal_validates(tmp_path):
    with pytest.raises(ValueError, match="lowercase"):
        save_signal(tmp_path, "Alice", "x", [1], [1.0], [2], [1.0])
    with pytest.raises(ValueError, match="NaN"):
        save_signal(tmp_path, "alice", "x", [1], [np.nan], [2], [1.0])
    with pytest.raises(ValueError, match="ids"):
        save_signal(tmp_path, "alice", "x", [1, 2], [1.0], [2], [1.0])
    with pytest.raises(FileNotFoundError):
        load_signals(tmp_path / "empty")


def test_out_of_fold_partial_leaves_uncovered_rows_nan():
    from arena.cv import out_of_fold, walk_forward
    oof = out_of_fold(lambda tr, te: np.ones(len(te)), 10, walk_forward(10, k=2), allow_partial=True)
    assert np.isnan(oof[:5]).all() and (oof[5:] == 1).all()


def test_walk_forward_signals_roundtrip_and_blend(tmp_path, keyed, y):
    from arena.cv import signal_fit_predict, walk_forward
    folds = walk_forward(len(y), k=4)
    export_signal(tmp_path, "dana", "wf", signal_fit_predict(y), keyed.ids, [1], [0.0], folds=folds)
    export_signal(tmp_path, "erin", "wf", signal_fit_predict(-y), keyed.ids, [1], [0.0], folds=folds)
    ins, oos = load_signals(tmp_path)
    assert ins["dana/wf"].isna().sum() == len(y) // 2
    assert leaderboard(ins, pd.Series(y, index=keyed.ids)).index[0] == "dana/wf"
    assert np.isfinite(blend_predict(ins, oos, y)).all()
    assert np.isfinite(blend_cv(ins, y).mean)


def test_blend_cv_maps_walk_forward_folds_onto_complete_rows():
    from arena.cv import walk_forward
    from arena.signals import blend_cv

    rng = np.random.default_rng(0)
    y = rng.normal(size=300)
    ins = pd.DataFrame({"a": y + rng.normal(size=300), "b": rng.normal(size=300)})
    ins.iloc[[5, 250], 0] = np.nan  # one row inside a test block
    res = blend_cv(ins, y, folds=walk_forward(300, k=3))
    assert np.isfinite(res.mean) and res.mean > 0
