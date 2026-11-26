import numpy as np
import pandas as pd
import pytest

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.data import concat_sessions, forward_returns, load_extra_sessions, subset, truncate


def test_concat_sessions_keeps_rows_and_headlines(sessions):
    both = concat_sessions(subset(sessions, np.arange(150)), subset(sessions, np.arange(150, 400)))
    assert both.n == 400
    np.testing.assert_array_equal(both.ids, sessions.ids)
    np.testing.assert_allclose(both.close, sessions.close)
    pd.testing.assert_frame_equal(both.headlines, sessions.headlines)


def test_concat_sessions_rejects_duplicates_and_mismatched_bars(sessions):
    a = subset(sessions, np.arange(10))
    with pytest.raises(ValueError, match="duplicate"):
        concat_sessions(a, a)
    with pytest.raises(ValueError, match="bars"):
        concat_sessions(a, truncate(subset(sessions, np.arange(10, 20)), 49))


def test_load_extra_sessions_returns_full_sessions_and_demeaned_target(datathon_dir):
    extra, y_extra = load_extra_sessions(datathon_dir)
    assert extra.n == 100 and extra.n_bars == 100
    np.testing.assert_allclose(y_extra, forward_returns(extra, SPEC))
    assert abs(y_extra.mean()) < 1e-12


def test_load_extra_sessions_names_missing_files(tmp_path):
    with pytest.raises(FileNotFoundError, match="bars_seen_train.parquet"):
        load_extra_sessions(tmp_path)
