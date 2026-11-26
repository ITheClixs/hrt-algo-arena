from dataclasses import replace

import numpy as np
import pytest

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.data import load_arcade
from arena.horizon import detect_horizon, resolve_spec, scorer_constants
from arena.synth import write_arcade_files


def test_detects_the_standard_horizon(arcade_dir):
    h = detect_horizon(load_arcade(arcade_dir))
    assert (h.halfway, h.exit, h.demeaned) == (49, 59, True) and h.ok and h.corr > 0.9999


@pytest.mark.parametrize(("halfway", "exit_bar"), [(49, 69), (59, 69)])
def test_detects_a_different_horizon(market, tmp_path, halfway, exit_bar):
    spec = replace(SPEC, halfway_bar=halfway, exit_bar=exit_bar)
    h = detect_horizon(load_arcade(write_arcade_files(market, tmp_path / "d", spec=spec)))
    assert (h.halfway, h.exit) == (halfway, exit_bar)


def test_detects_raw_returns(market, tmp_path):
    spec = replace(SPEC, demean_returns=False)
    h = detect_horizon(load_arcade(write_arcade_files(market, tmp_path / "raw", spec=spec)))
    assert h.exit == 59 and not h.demeaned


def test_no_match_when_returns_are_not_close_to_close(arcade_dir):
    d = load_arcade(arcade_dir)
    noise = np.random.default_rng(0).normal(size=d.y.shape)
    h = detect_horizon(replace(d, y=noise))
    assert not h.ok and h.exit is None and "no exit bar" in h.describe()


def test_resolve_spec_uses_the_files_and_rejects_contradicting_flags(arcade_dir):
    d = load_arcade(arcade_dir)
    assert resolve_spec(SPEC, d).exit_bar == 59
    assert resolve_spec(SPEC, d, halfway=49, exit_bar=59).halfway_bar == 49
    with pytest.raises(ValueError, match="exit"):
        resolve_spec(SPEC, d, exit_bar=69)
    with pytest.raises(ValueError, match="halfway"):
        resolve_spec(SPEC, d, halfway=59)


def test_resolve_spec_needs_both_flags_when_the_files_do_not_say(arcade_dir):
    d = load_arcade(arcade_dir)
    blind = replace(d, y=np.random.default_rng(1).normal(size=d.y.shape))
    with pytest.raises(ValueError, match="--halfway and --exit"):
        resolve_spec(SPEC, blind)
    spec = resolve_spec(SPEC, blind, halfway=49, exit_bar=59)
    assert (spec.halfway_bar, spec.exit_bar) == (49, 59)


def test_scorer_constants(tmp_path):
    (tmp_path / "sessions.py").write_text("HALFWAY_BAR = 49\n\"\"\"doc\"\"\"\n\nHORIZON_BARS = 10\n")
    assert scorer_constants(tmp_path / "sessions.py") == {"HALFWAY_BAR": 49, "HORIZON_BARS": 10}
    assert scorer_constants(tmp_path / "missing.py") == {}
