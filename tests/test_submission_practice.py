import numpy as np
import pandas as pd
import pytest

from arena.config import ARCADE_ALGO_ARENA, DATATHON_2026
from arena.practice import MockLeaderboard, holdout_split
from arena.submission import (
    build_submission,
    log_submission,
    validate_submission,
    write_submission,
)


def test_write_submission_arcade_format(tmp_path):
    with pytest.warns(UserWarning, match="1 non-finite prediction"):
        path = write_submission([11000, 11001], [0.5, np.nan], tmp_path / "s" / "p.csv")
    df = pd.read_csv(path)
    assert list(df.columns) == ["headline_id", "pred"]
    assert df["pred"].tolist() == [0.5, 0.0]


def test_build_submission_datathon_columns():
    df = build_submission([1], [2.0], DATATHON_2026)
    assert list(df.columns) == ["session", "target_position"]
    with pytest.raises(ValueError):
        build_submission([1, 2], [2.0])


@pytest.mark.parametrize(
    ("frame", "message"),
    [
        (pd.DataFrame({"headline_id": [1, 1], "pred": [1.0, 1.0]}), "duplicate"),
        (pd.DataFrame({"headline_id": [1], "pred": [1.0]}), "missing"),
        (pd.DataFrame({"headline_id": [1, 2], "pred": [1.0, np.inf]}), "NaN"),
        (pd.DataFrame({"headline_id": [1, 2], "pred": [0.0, 0.0]}), "zero"),
        (pd.DataFrame({"session": [1, 2], "pred": [1.0, 1.0]}), "missing column"),
    ],
)
def test_validate_submission_errors(frame, message):
    with pytest.raises(ValueError, match=message):
        validate_submission(frame, [1, 2], ARCADE_ALGO_ARENA)


def test_ledger_appends(tmp_path):
    ledger = tmp_path / "ledger.csv"
    log_submission(ledger, file="a.csv", label="baseline", cv_sharpe=1.23456, author="alice")
    out = log_submission(ledger, file="b.csv", label="v2", cv_sharpe=2.0, author="bob",
                         leaderboard=1.9, notes="blend")
    assert out["label"].tolist() == ["baseline", "v2"]
    assert out.loc[1, "leaderboard"] == 1.9


def test_holdout_split():
    tr, ho = holdout_split(100, 0.3)
    assert len(ho) == 30 and set(tr).isdisjoint(ho)
    with pytest.raises(ValueError):
        holdout_split(10, 1.5)


def test_mock_leaderboard_budget_and_private_split():
    rng = np.random.default_rng(0)
    ids, ret = np.arange(500), rng.normal(size=500)
    lb = MockLeaderboard(ids, ret, budget=2, private_frac=0.5)
    assert lb.submit(ids, ret, "oracle") > 10
    lb.submit(ids[::-1], ret[::-1])
    assert lb.remaining == 0
    with pytest.raises(RuntimeError):
        lb.submit(ids, ret)
    hist = lb.history(reveal_private=True)
    assert list(hist.columns) == ["label", "public", "private"]
    assert hist["public"].iloc[0] == pytest.approx(hist["public"].iloc[1])


def test_mock_leaderboard_rejects_incomplete_upload():
    lb = MockLeaderboard(np.arange(5), np.ones(5))
    with pytest.raises(ValueError):
        lb.submit(np.arange(4), np.ones(4))
    assert "private" not in lb.history().columns


def test_string_and_zero_padded_ids_are_kept(tmp_path):
    from dataclasses import replace

    from arena.config import ARCADE_ALGO_ARENA

    spec = replace(ARCADE_ALGO_ARENA, id_col="row_id")
    ids = np.array(["a1b2", "0012", "0013"])
    df = build_submission(ids, [0.1, -0.2, 0.3], spec)
    assert df["row_id"].tolist() == ["a1b2", "0012", "0013"]
    path = write_submission(ids, [0.1, -0.2, 0.3], tmp_path / "s.csv", spec)
    assert path.read_text().splitlines()[1:3] == ["a1b2,0.1", "0012,-0.2"]
    assert build_submission(np.array([11000.0, 11001.0]), [1.0, 2.0])["headline_id"].dtype == np.int64


def test_retire_stale_uploads_moves_only_runner_files_from_another_run(tmp_path):
    from arena.submission import retire_stale_uploads

    out = tmp_path / "subs"
    assert retire_stale_uploads(out, "A") == []
    for name in ("01_primary_hrtonly.csv", "01_primary_withextra.csv", "ledger.csv", "my_blend.csv"):
        (out / name).write_text("x\n")
    assert retire_stale_uploads(out, "A") == []  # same run identity: keep
    moved = retire_stale_uploads(out, "B")
    assert sorted(p.name for p in moved) == ["01_primary_hrtonly.csv", "01_primary_withextra.csv"]
    assert all(p.exists() and p.parent.parent == out / "old" for p in moved)
    assert sorted(p.name for p in out.glob("*.csv")) == ["ledger.csv", "my_blend.csv"]
