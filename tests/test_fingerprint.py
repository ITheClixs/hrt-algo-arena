from arena.fingerprint import (
    ARCADE_FILES,
    KNOWN_ARCADE_SHA256,
    file_sha256,
    fingerprint,
    same_as_known,
)


def test_known_hashes_cover_the_five_arcade_files():
    assert set(KNOWN_ARCADE_SHA256) == set(ARCADE_FILES) and len(ARCADE_FILES) == 5
    assert KNOWN_ARCADE_SHA256["ins_bars.parquet"] == (
        "32dc426a6078c77d60fcf4b403677c3ba4030247e4573784f7345e80f8e25960")


def test_fingerprint_hashes_present_files_and_marks_missing(arcade_dir):
    assert fingerprint(arcade_dir)["ins_bars.parquet"] == file_sha256(arcade_dir / "ins_bars.parquet")
    (arcade_dir / "oos_headlines.parquet").unlink()
    assert fingerprint(arcade_dir)["oos_headlines.parquet"] is None


def test_same_as_known(arcade_dir):
    assert not same_as_known(arcade_dir)
    assert same_as_known(arcade_dir, known=fingerprint(arcade_dir))


def test_content_hash_survives_a_resave_but_not_a_changed_value(arcade_dir, tmp_path):
    import pandas as pd

    from arena.fingerprint import content_sha256

    path = arcade_dir / "ins_bars.parquet"
    df = pd.read_parquet(path)
    resaved = tmp_path / "resaved.parquet"
    df.sample(frac=1.0, random_state=0).astype({"bar_ix": "int32"}).to_parquet(
        resaved, compression="gzip")
    assert file_sha256(resaved) != file_sha256(path)
    assert content_sha256(resaved) == content_sha256(path)
    changed = tmp_path / "changed.parquet"
    df.assign(close=df["close"].where(df.index != 7, df["close"] + 1e-9)).to_parquet(changed)
    assert content_sha256(changed) != content_sha256(path)


def test_match_known_reports_how_the_files_match(arcade_dir):
    import pandas as pd

    from arena.fingerprint import content_fingerprint, match_known

    known_bytes, known_content = fingerprint(arcade_dir), content_fingerprint(arcade_dir)
    assert match_known(arcade_dir, known_bytes, known_content) == "bytes"
    df = pd.read_parquet(arcade_dir / "oos_headlines.parquet")
    df.iloc[::-1].to_parquet(arcade_dir / "oos_headlines.parquet", index=False)
    assert match_known(arcade_dir, known_bytes, known_content) == "content"
    assert match_known(arcade_dir, known_bytes, {}) is None


def test_known_content_hashes_and_scorer_hash_are_pinned():
    from arena.fingerprint import KNOWN_ARCADE_CONTENT, KNOWN_SESSIONS_PY_SHA256

    assert set(KNOWN_ARCADE_CONTENT) == set(ARCADE_FILES)
    assert len(KNOWN_SESSIONS_PY_SHA256) == 64


def test_scorer_status(tmp_path):
    from arena.fingerprint import file_sha256, scorer_status

    assert scorer_status(tmp_path) == "missing"
    (tmp_path / "sessions.py").write_text("HALFWAY_BAR = 49\n")
    assert scorer_status(tmp_path) == "changed"
    assert scorer_status(tmp_path, known=file_sha256(tmp_path / "sessions.py")) == "same"
