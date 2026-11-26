from dataclasses import replace

import pandas as pd

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.fingerprint import fingerprint
from arena.pipeline import CANDIDATES
from arena.runner import main
from arena.submission import validate_submission
from arena.synth import write_arcade_files


def _oos_ids(d):
    return pd.read_parquet(d / "oos_bars.parquet")["session"].unique()


def test_runner_writes_valid_uploads_in_two_versions(arcade_dir, datathon_dir, tmp_path, capsys):
    out = tmp_path / "subs"
    args = ["--data", str(arcade_dir), "--extra", str(datathon_dir), "--out", str(out), "--cv", "none"]
    assert main(args) == 0
    assert "NEW DATA" in capsys.readouterr().out
    names = sorted(p.name for p in out.glob("*.csv"))
    assert len(names) == 2 * len(CANDIDATES) and {"01_primary_hrtonly.csv", "01_primary_withextra.csv"} <= set(names)
    for f in out.glob("*.csv"):
        validate_submission(pd.read_csv(f), _oos_ids(arcade_dir), SPEC)


def test_runner_without_extra_writes_hrtonly_only(arcade_dir, tmp_path, capsys):
    out = tmp_path / "subs"
    assert main(["--data", str(arcade_dir), "--extra", "none", "--out", str(out), "--cv", "none"]) == 0
    names = sorted(p.name for p in out.glob("*.csv"))
    assert len(names) == len(CANDIDATES) and all(n.endswith("_hrtonly.csv") for n in names)
    assert "_hrtonly files only" in capsys.readouterr().out


def test_runner_reports_missing_files(tmp_path, capsys):
    assert main(["--data", str(tmp_path), "--extra", "none", "--out", str(tmp_path / "o")]) == 2
    assert "missing ins_bars.parquet" in capsys.readouterr().out


def test_runner_recognises_the_known_dataset(arcade_dir, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("arena.runner.KNOWN_ARCADE_SHA256", fingerprint(arcade_dir))
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--cv", "none"])
    assert "SAME DATA" in capsys.readouterr().out


def test_runner_fast_cv_prints_a_score(arcade_dir, tmp_path, capsys):
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--cv", "fast"])
    assert "CV " in capsys.readouterr().out


def test_runner_accepts_a_different_horizon(market, tmp_path):
    spec69 = replace(SPEC, exit_bar=69)
    d = write_arcade_files(market, tmp_path / "h69", spec=spec69)
    out = tmp_path / "o"
    args = ["--data", str(d), "--extra", "none", "--out", str(out), "--cv", "none", "--exit", "69"]
    assert main(args) == 0
    validate_submission(pd.read_csv(out / "01_primary_hrtonly.csv"), _oos_ids(d), spec69)


def test_runner_prints_facts_on_new_data(arcade_dir, tmp_path, capsys):
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--cv", "none"])
    assert "push_duration" in capsys.readouterr().out


def test_runner_only_writes_the_requested_candidate(arcade_dir, tmp_path):
    out = tmp_path / "o"
    assert main(["--data", str(arcade_dir), "--extra", "none", "--out", str(out), "--cv", "none",
                 "--only", "sector"]) == 0
    assert [p.name for p in out.glob("*.csv")] == ["04_sector_hrtonly.csv"]


def test_runner_fast_cv_covers_every_requested_candidate(arcade_dir, tmp_path, capsys):
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--cv", "fast",
          "--only", "sector"])
    assert "quick 5-fold" in capsys.readouterr().out


# --- day-of safety (25 Nov review) -------------------------------------------------------------

def _no_env(monkeypatch):
    monkeypatch.delenv("ARENA_DATA", raising=False)


def test_runner_prints_the_folder_it_reads(arcade_dir, tmp_path, capsys):
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--cv", "none"])
    assert f"data: {arcade_dir}" in capsys.readouterr().out


def test_runner_refuses_the_practice_fallback_unless_asked(arcade_dir, tmp_path, capsys, monkeypatch):
    _no_env(monkeypatch)
    monkeypatch.setattr("arena.runner.RAW", tmp_path / "raw_empty")
    monkeypatch.setattr("arena.runner.PRACTICE_ARCADE", arcade_dir)
    out = tmp_path / "o"
    assert main(["--extra", "none", "--out", str(out), "--cv", "none"]) == 2
    assert "--practice" in capsys.readouterr().out and not out.exists()
    assert main(["--practice", "--extra", "none", "--out", str(out), "--cv", "none"]) == 0
    assert "PRACTICE DATA" in capsys.readouterr().out


def test_runner_finds_the_files_in_a_subfolder_of_raw(arcade_dir, tmp_path, capsys, monkeypatch):
    import shutil

    _no_env(monkeypatch)
    raw = tmp_path / "raw"
    shutil.copytree(arcade_dir, raw / "AlgoArena Data")
    monkeypatch.setattr("arena.runner.RAW", raw)
    assert main(["--extra", "none", "--out", str(tmp_path / "o"), "--cv", "none", "--only", "primary"]) == 0
    assert f"data: {raw / 'AlgoArena Data'}" in capsys.readouterr().out


def test_runner_rejects_an_unknown_label(arcade_dir, tmp_path, capsys):
    out = tmp_path / "o"
    assert main(["--data", str(arcade_dir), "--out", str(out), "--only", "primay"]) == 2
    text = capsys.readouterr().out
    assert "primay" in text and "primary" in text and not out.exists()


def test_new_data_leaves_the_extra_sessions_out_by_default(arcade_dir, datathon_dir, tmp_path, capsys,
                                                            monkeypatch):
    monkeypatch.setattr("arena.runner.DEFAULT_EXTRA", datathon_dir)
    out = tmp_path / "o"
    assert main(["--data", str(arcade_dir), "--out", str(out), "--cv", "none", "--only", "primary"]) == 0
    assert [p.name for p in out.glob("*.csv")] == ["01_primary_hrtonly.csv"]
    assert "extra sessions off" in capsys.readouterr().out


def test_same_data_uses_the_extra_sessions_by_default(arcade_dir, datathon_dir, tmp_path, monkeypatch):
    monkeypatch.setattr("arena.runner.DEFAULT_EXTRA", datathon_dir)
    monkeypatch.setattr("arena.runner.KNOWN_ARCADE_SHA256", fingerprint(arcade_dir))
    out = tmp_path / "o"
    assert main(["--data", str(arcade_dir), "--out", str(out), "--cv", "none", "--only", "primary"]) == 0
    assert sorted(p.name for p in out.glob("*.csv")) == ["01_primary_hrtonly.csv", "01_primary_withextra.csv"]


def test_runner_reads_a_changed_horizon_off_the_files(market, tmp_path, capsys):
    d = write_arcade_files(market, tmp_path / "h69", spec=replace(SPEC, exit_bar=69))
    out = tmp_path / "o"
    assert main(["--data", str(d), "--extra", "none", "--out", str(out), "--cv", "none", "--only", "primary"]) == 0
    assert "exit at close of bar 69" in capsys.readouterr().out


def test_runner_aborts_when_a_flag_contradicts_the_files(arcade_dir, tmp_path, capsys):
    out = tmp_path / "o"
    assert main(["--data", str(arcade_dir), "--extra", "none", "--out", str(out), "--exit", "69"]) == 2
    assert "contradicts" in capsys.readouterr().out and not out.exists()


def test_runner_writes_every_file_before_the_cv_and_survives_a_cv_crash(arcade_dir, tmp_path, capsys,
                                                                       monkeypatch):
    def boom(*args, **kwargs):
        raise RuntimeError("boom")

    monkeypatch.setattr("arena.runner.evaluate", boom)
    out = tmp_path / "o"
    assert main(["--data", str(arcade_dir), "--extra", "none", "--out", str(out), "--cv", "fast"]) == 0
    assert len(list(out.glob("*.csv"))) == len(CANDIDATES)
    assert "CV n/a (RuntimeError: boom)" in capsys.readouterr().out


def test_quick_cv_works_when_oos_ids_overlap_the_in_sample_ids(arcade_dir, tmp_path, capsys):
    ins_ids = pd.read_parquet(arcade_dir / "ins_bars.parquet")["session"].unique()
    for name in ("oos_bars.parquet", "oos_headlines.parquet"):
        df = pd.read_parquet(arcade_dir / name)
        df["session"] -= df["session"].min() - ins_ids.min()  # oos ids now start where ins ids start
        df.to_parquet(arcade_dir / name, index=False)
    out = tmp_path / "o"
    assert main(["--data", str(arcade_dir), "--extra", "none", "--out", str(out), "--cv", "fast",
                 "--only", "primary"]) == 0
    text = capsys.readouterr().out
    assert "quick 5-fold" in text and "CV n/a" not in text
    validate_submission(pd.read_csv(out / "01_primary_hrtonly.csv"), _oos_ids(arcade_dir), SPEC)


def test_weak_industry_structure_turns_the_filter_off_on_new_data(arcade_dir, tmp_path, capsys):
    # the `market` fixture has no industries: the eigengap is weak, so the files must not use 5 arbitrary groups
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--only", "primary"])
    text = capsys.readouterr().out
    assert "industries:" in text and "industry filter OFF" in text
    assert "with the 5-group industry filter: CV" in text


def test_clear_industries_keep_the_filter_and_show_the_cv_without_it(tmp_path, capsys):
    from arena.synth import make_synthetic

    d = write_arcade_files(make_synthetic(n_sessions=600, seed=5, n_industries=5), tmp_path / "ind5")
    main(["--data", str(d), "--extra", "none", "--out", str(tmp_path / "o"), "--only", "primary"])
    text = capsys.readouterr().out
    assert "eigengap suggests 5" in text and "industry filter OFF" not in text
    assert "without industry filter: CV" in text


def test_a_broken_extra_folder_never_costs_the_hrtonly_files(arcade_dir, datathon_dir, tmp_path, capsys):
    (datathon_dir / "bars_unseen_train.parquet").write_bytes(b"not a parquet file")
    out = tmp_path / "o"
    assert main(["--data", str(arcade_dir), "--extra", str(datathon_dir), "--out", str(out), "--cv", "none",
                 "--only", "primary"]) == 0
    assert [p.name for p in out.glob("*.csv")] == ["01_primary_hrtonly.csv"]
    assert "cannot read the extra sessions" in capsys.readouterr().out


def test_two_copies_under_raw_stop_cleanly(arcade_dir, tmp_path, capsys, monkeypatch):
    import shutil

    _no_env(monkeypatch)
    raw = tmp_path / "raw"
    for sub in ("copy1", "copy2"):
        shutil.copytree(arcade_dir, raw / sub)
    monkeypatch.setattr("arena.runner.RAW", raw)
    assert main(["--extra", "none", "--out", str(tmp_path / "o")]) == 2
    assert "several folders" in capsys.readouterr().out


def test_unreadable_files_stop_cleanly(arcade_dir, tmp_path, capsys):
    rets = pd.read_parquet(arcade_dir / "ins_returns.parquet").rename(columns={"returns": "ret"})
    rets.to_parquet(arcade_dir / "ins_returns.parquet", index=False)
    assert main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o")]) == 2
    assert "cannot read the arcade files" in capsys.readouterr().out


def test_uploads_from_other_data_are_moved_out_of_the_way(arcade_dir, datathon_dir, market, tmp_path, capsys):
    out = tmp_path / "o"
    main(["--data", str(arcade_dir), "--extra", str(datathon_dir), "--out", str(out), "--cv", "none",
          "--only", "primary"])
    other = write_arcade_files(market, tmp_path / "other", oos_frac=0.4)
    capsys.readouterr()
    main(["--data", str(other), "--extra", "none", "--out", str(out), "--cv", "none", "--only", "primary"])
    assert [p.name for p in out.glob("*.csv")] == ["01_primary_hrtonly.csv"]
    moved = list((out / "old").glob("*/*.csv"))
    assert sorted(p.name for p in moved) == ["01_primary_hrtonly.csv", "01_primary_withextra.csv"]
    assert "moved 2 upload file(s)" in capsys.readouterr().out


def test_uploads_from_the_same_data_stay(arcade_dir, tmp_path):
    out = tmp_path / "o"
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(out), "--cv", "none", "--only", "sector"])
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(out), "--cv", "none", "--only", "primary"])
    assert sorted(p.name for p in out.glob("*.csv")) == ["01_primary_hrtonly.csv", "04_sector_hrtonly.csv"]


def test_stored_cv_is_not_claimed_for_a_changed_model(arcade_dir, tmp_path, capsys, monkeypatch):
    monkeypatch.setattr("arena.runner.KNOWN_ARCADE_SHA256", fingerprint(arcade_dir))
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--only", "primary",
          "--no-industry"])
    text = capsys.readouterr().out
    assert "(stored" not in text and "quick 5-fold" in text


def test_horizon_line_says_when_it_came_from_the_flags(arcade_dir, tmp_path, capsys):
    import numpy as np

    rets = pd.read_parquet(arcade_dir / "ins_returns.parquet")
    rets["returns"] = np.random.default_rng(0).normal(size=len(rets))
    rets.to_parquet(arcade_dir / "ins_returns.parquet", index=False)
    assert main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--cv", "none",
                 "--only", "primary", "--halfway", "49", "--exit", "59"]) == 0
    assert "from --halfway/--exit" in capsys.readouterr().out


def test_withextra_files_say_cv_na(arcade_dir, datathon_dir, tmp_path, capsys):
    main(["--data", str(arcade_dir), "--extra", str(datathon_dir), "--out", str(tmp_path / "o"), "--cv", "fast",
          "--only", "sector"])
    assert "_withextra: CV n/a" in capsys.readouterr().out


def test_a_failing_candidate_does_not_stop_the_others(arcade_dir, tmp_path, capsys, monkeypatch):
    from arena.pipeline import Candidate

    def broken(spec):
        raise RuntimeError("cannot build")

    monkeypatch.setattr("arena.runner.CANDIDATES", (Candidate("broken", broken, "x"), CANDIDATES[0]))
    out = tmp_path / "o"
    assert main(["--data", str(arcade_dir), "--extra", "none", "--out", str(out), "--cv", "none"]) == 1
    assert [p.name for p in out.glob("*.csv")] == ["02_primary_hrtonly.csv"]
    assert "FAILED 01_broken" in capsys.readouterr().out


def test_no_industry_flag_keeps_every_headline(arcade_dir, tmp_path, capsys):
    assert main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--cv", "none",
                 "--only", "primary", "--no-industry"]) == 0
    assert "industry filter OFF" in capsys.readouterr().out


def test_runner_reports_the_scorer_file(arcade_dir, tmp_path, capsys):
    main(["--data", str(arcade_dir), "--extra", "none", "--out", str(tmp_path / "o"), "--cv", "none",
          "--only", "primary"])
    assert "sessions.py: not shipped" in capsys.readouterr().out
