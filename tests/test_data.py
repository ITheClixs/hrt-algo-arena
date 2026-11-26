import numpy as np
import pandas as pd
import pytest

from arena.config import ARCADE_ALGO_ARENA
from arena.data import (
    forward_returns,
    from_long,
    list_data_files,
    load_arcade,
    load_datathon,
    subset,
    truncate,
    with_column,
)
from arena.synth import returns_frame


def test_from_long_shapes(sessions):
    assert sessions.n == 400
    assert sessions.n_bars == 100
    hl = sessions.headlines
    assert hl["sidx"].between(0, sessions.n - 1).all()
    assert (sessions.ids[hl["sidx"]] == hl["session"].to_numpy()).all()
    assert hl[["sidx", "bar_ix"]].apply(tuple, axis=1).is_monotonic_increasing


def test_from_long_without_headlines(market):
    s = from_long(market.bars)
    assert len(s.headlines) == 0
    assert list(s.headlines.columns) == ["session", "sidx", "bar_ix", "headline"]


def test_ragged_bars_raise(market):
    with pytest.raises(ValueError, match="ragged"):
        from_long(market.bars.iloc[1:])


def test_missing_columns_raise(market):
    with pytest.raises(ValueError, match="missing columns"):
        from_long(market.bars.drop(columns=["high"]))


def test_truncate_hides_the_future(sessions):
    seen = truncate(sessions, 49)
    assert seen.n_bars == 50
    assert seen.headlines["bar_ix"].max() <= 49
    np.testing.assert_array_equal(seen.close, sessions.close[:, :50])
    with pytest.raises(ValueError):
        truncate(sessions, 100)


def test_subset_remaps_headlines(sessions):
    sub = subset(sessions, [5, 2])
    assert list(sub.ids) == [sessions.ids[5], sessions.ids[2]]
    assert set(sub.headlines["sidx"]) <= {0, 1}
    assert (sub.ids[sub.headlines["sidx"]] == sub.headlines["session"].to_numpy()).all()
    mask = np.zeros(sessions.n, dtype=bool)
    mask[[1, 3]] = True
    assert subset(sessions, mask).n == 2
    with pytest.raises(ValueError, match="unique"):
        subset(sessions, [1, 1])


def test_with_column_checks_length(sessions):
    assert "x" in with_column(sessions, "x", np.zeros(len(sessions.headlines))).headlines
    with pytest.raises(ValueError):
        with_column(sessions, "x", [1, 2])


def test_forward_returns_match_returns_frame(market, sessions):
    r = forward_returns(sessions, ARCADE_ALGO_ARENA)
    expected = returns_frame(market.bars).set_index("session")["returns"]
    np.testing.assert_allclose(r, expected.reindex(sessions.ids).to_numpy())
    assert abs(r.mean()) < 1e-12
    with pytest.raises(ValueError):
        forward_returns(truncate(sessions, 49), ARCADE_ALGO_ARENA)


def test_load_arcade_roundtrip(arcade_dir):
    data = load_arcade(arcade_dir)
    assert data.ins.n_bars == 100 and data.oos.n_bars == 50
    assert len(data.y) == data.ins.n
    assert data.oos.headlines["bar_ix"].max() <= 49
    assert set(data.ins.ids).isdisjoint(data.oos.ids)


def test_load_arcade_missing_file_is_explained(tmp_path):
    with pytest.raises(FileNotFoundError, match="Download the data"):
        load_arcade(tmp_path)


def test_load_datathon(market, tmp_path):
    bars, hl = market.bars, market.headlines[["session", "bar_ix", "headline"]]
    for split in ("train", "public_test", "private_test"):
        bars[bars.bar_ix <= 49].to_parquet(tmp_path / f"bars_seen_{split}.parquet")
        hl[hl.bar_ix <= 49].to_parquet(tmp_path / f"headlines_seen_{split}.parquet")
    bars[bars.bar_ix > 49].to_parquet(tmp_path / "bars_unseen_train.parquet")
    hl[hl.bar_ix > 49].to_parquet(tmp_path / "headlines_unseen_train.parquet")
    data = load_datathon(tmp_path)
    assert data.train.n_bars == 100
    assert data.public_test.n_bars == 50


def test_list_data_files(arcade_dir):
    pd.DataFrame({"a": [1]}).to_csv(arcade_dir / "extra.csv", index=False)
    inv = list_data_files(arcade_dir)
    assert len(inv) == 6
    assert "headline" in inv.set_index("file").loc["ins_headlines.parquet", "columns"]
