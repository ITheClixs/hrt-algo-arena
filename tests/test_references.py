import numpy as np

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.cv import cross_validate, kfold
from arena.data import subset, truncate
from arena.modelapi import as_fit_predict_with_unlabeled
from arena.references import WinnerRef
from arena.synth import make_synthetic, write_arcade_files
from arena.universe import load_universe


def test_winner_ref_never_peeks_past_halfway(arcade_dir):
    u = load_universe(arcade_dir)
    fitted = WinnerRef(SPEC).fit(subset(u.ins, np.arange(150)), u.y[:150], unlabeled=u.oos)
    test = subset(u.ins, np.arange(150, u.ins.n))
    np.testing.assert_allclose(fitted.predict(test), fitted.predict(truncate(test, 49)))


def test_winner_ref_finds_the_planted_signal(tmp_path):
    # Unregularised least squares on 10 features (like the original) needs more than the
    # 150 training sessions per fold of the shared fixture; 1,000 in-sample sessions here.
    u = load_universe(write_arcade_files(make_synthetic(n_sessions=2000, seed=1), tmp_path / "big"))
    fp = as_fit_predict_with_unlabeled(WinnerRef(SPEC), u.ins, u.y, unlabeled=u.oos)
    assert cross_validate(fp, u.y, kfold(u.ins.n, 4)).mean > 1.0
