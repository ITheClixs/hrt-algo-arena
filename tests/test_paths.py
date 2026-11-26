from pathlib import Path

import pandas as pd
import pytest

from arena import paths


def test_env_var_wins(monkeypatch, tmp_path):
    monkeypatch.setenv("ARENA_DATA", str(tmp_path))
    assert paths.default_data_dir() == tmp_path.resolve()


def test_first_candidate_with_parquet(monkeypatch, tmp_path):
    monkeypatch.delenv("ARENA_DATA", raising=False)
    empty, full = tmp_path / "empty", tmp_path / "full"
    empty.mkdir()
    full.mkdir()
    pd.DataFrame({"a": [1]}).to_parquet(full / "x.parquet")
    monkeypatch.setattr(paths, "CANDIDATES", (empty, full))
    with pytest.warns(UserWarning):
        assert paths.default_data_dir() == full


def test_no_data_is_explained(monkeypatch, tmp_path):
    monkeypatch.delenv("ARENA_DATA", raising=False)
    monkeypatch.setattr(paths, "CANDIDATES", (tmp_path / "nope",))
    with pytest.raises(FileNotFoundError, match="fetch_gdrive"):
        paths.default_data_dir()


def test_repo_root_contains_pyproject():
    assert (paths.REPO / "pyproject.toml").exists()
    assert isinstance(paths.SIGNALS, Path)


def test_falling_back_past_data_raw_warns(monkeypatch, tmp_path):
    monkeypatch.delenv("ARENA_DATA", raising=False)
    raw, practice = tmp_path / "raw", tmp_path / "practice"
    raw.mkdir()
    practice.mkdir()
    pd.DataFrame({"a": [1]}).to_parquet(practice / "x.parquet")
    monkeypatch.setattr(paths, "CANDIDATES", (raw, practice))
    with pytest.warns(UserWarning, match="NOT the competition data"):
        assert paths.default_data_dir() == practice


def test_find_dir_with(tmp_path):
    names = ("a.parquet", "b.parquet")
    sub = tmp_path / "drive" / "folder"
    sub.mkdir(parents=True)
    assert paths.find_dir_with(tmp_path, names) is None
    for n in names:
        (sub / n).write_bytes(b"x")
    assert paths.find_dir_with(tmp_path, names) == sub
    assert paths.find_dir_with(sub, names) == sub
    other = tmp_path / "other"
    other.mkdir()
    for n in names:
        (other / n).write_bytes(b"x")
    with pytest.raises(ValueError, match="several folders"):
        paths.find_dir_with(tmp_path, names)
    assert paths.find_dir_with(tmp_path / "nope", names) is None
