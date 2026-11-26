import numpy as np
import pytest

from arena.cv import (
    blocked,
    cross_validate,
    kfold,
    out_of_fold,
    repeated_kfold,
    signal_fit_predict,
    walk_forward,
)


def test_kfold_partitions_everything():
    folds = kfold(103, k=5, seed=3)
    tests = np.concatenate([te for _, te in folds])
    assert sorted(tests.tolist()) == list(range(103))
    for tr, te in folds:
        assert set(tr).isdisjoint(te)
        assert len(tr) + len(te) == 103


def test_blocked_folds_are_contiguous():
    for _, te in blocked(50, 5):
        assert (np.diff(te) == 1).all()


def test_walk_forward_trains_only_on_the_past():
    folds = walk_forward(100, k=4, min_train_frac=0.5)
    assert len(folds) == 4
    for tr, te in folds:
        assert tr.max() < te.min()


def test_oracle_scores_high_and_noise_scores_low():
    y = np.random.default_rng(0).normal(size=1000)
    oracle = cross_validate(lambda tr, te: y[te], y)
    noise = cross_validate(lambda tr, te: np.random.default_rng(1).normal(size=len(te)), y)
    assert oracle.worst > 8  # pnl = y^2 -> Sharpe ~ sqrt(252 / 2) ~ 11.2
    assert abs(noise.mean) < 5


def test_wrong_shape_raises():
    y = np.ones(20)
    with pytest.raises(ValueError, match="shape"):
        cross_validate(lambda tr, te: np.ones(3), y, kfold(20, 2))


def test_repeated_kfold_reports_all_folds():
    y = np.random.default_rng(0).normal(size=200)
    res = repeated_kfold(signal_fit_predict(y), y, k=5, seeds=(0, 1, 2))
    assert len(res.fold_sharpes) == 15
    assert "CV Sharpe" in str(res)
    assert res.std >= 0


def test_out_of_fold_covers_all_and_detects_gaps():
    y = np.arange(10.0)
    np.testing.assert_array_equal(out_of_fold(signal_fit_predict(y), 10, kfold(10, 5)), y)
    with pytest.raises(ValueError, match="cover"):
        out_of_fold(signal_fit_predict(y), 10, kfold(10, 5)[:2])


def test_walk_forward_by_time_never_splits_a_timestamp():
    import pandas as pd

    from arena.cv import walk_forward_by_time
    times = pd.Series(np.repeat(pd.date_range("2015-01-01", periods=100), 3))
    folds = walk_forward_by_time(times, k=5, min_train_frac=0.5)
    assert len(folds) == 5
    for tr, te in folds:
        assert times[tr].max() < times[te].min()


def test_walk_forward_by_time_gap_drops_recent_training_rows():
    import pandas as pd

    from arena.cv import walk_forward_by_time
    times = pd.Series(pd.date_range("2015-01-01", periods=100))
    tr, te = walk_forward_by_time(times, k=2, min_train_frac=0.5, gap=pd.Timedelta(days=5))[0]
    assert times[te].min() - times[tr].max() > pd.Timedelta(days=5)


def test_walk_forward_by_time_handles_unsorted_rows():
    import pandas as pd

    from arena.cv import walk_forward_by_time
    times = pd.Series(pd.date_range("2015-01-01", periods=60)).sample(frac=1.0, random_state=0).reset_index(drop=True)
    for tr, te in walk_forward_by_time(times, k=3):
        assert times[tr].max() < times[te].min()


def test_walk_forward_by_time_ignores_missing_times():
    import pandas as pd

    from arena.cv import walk_forward_by_time

    times = pd.Series(pd.date_range("2020-01-01", periods=200, freq="D"))
    times[17] = pd.NaT
    folds = walk_forward_by_time(times, k=4)
    assert all(len(te) > 0 for _, te in folds)
    assert all(17 not in tr and 17 not in te for tr, te in folds)
    assert folds[-1][1].max() == 199
