import numpy as np
import pytest

from arena.augment import augmented, shift_window
from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.data import forward_returns, subset, truncate
from arena.generator import GeneratorModel
from arena.universe import load_universe


def test_shift_window_zero_is_the_original_problem(keyed):
    w = shift_window(keyed, 0)
    assert w.n_bars == 60
    np.testing.assert_allclose(w.close, keyed.close[:, :60])


def test_shift_window_moves_headlines(keyed):
    w = shift_window(keyed, 10)
    src = keyed.headlines[(keyed.headlines.bar_ix >= 10) & (keyed.headlines.bar_ix < 70)]
    assert len(w.headlines) == len(src) and w.headlines.bar_ix.min() >= 0
    assert set(w.ids) == set(keyed.ids * 100 + 10)


def test_shift_window_rejects_windows_past_the_end(keyed):
    with pytest.raises(ValueError):
        shift_window(keyed, 41)


def test_augmented_targets_and_groups(keyed):
    s, y, groups = augmented(keyed, (0, 40))
    assert s.n == 2 * keyed.n and np.array_equal(groups[: keyed.n], np.arange(keyed.n))
    np.testing.assert_allclose(y[: keyed.n], forward_returns(keyed, SPEC))


@pytest.mark.parametrize("offsets", [(0, 20), (0, 40)])
def test_generator_with_augmentation_never_peeks_past_halfway(arcade_dir, offsets):
    u = load_universe(arcade_dir)
    fitted = GeneratorModel(SPEC, augment_offsets=offsets).fit(subset(u.ins, np.arange(150)), u.y[:150],
                                                               unlabeled=u.oos)
    test = subset(u.ins, np.arange(150, u.ins.n))
    np.testing.assert_allclose(fitted.predict(test), fitted.predict(truncate(test, 49)))


def test_augmented_skips_windows_that_do_not_fit_a_longer_horizon(keyed):
    from dataclasses import replace
    spec69 = replace(SPEC, exit_bar=69)  # 50 visible + 20 held: offsets above 30 do not fit 100 bars
    s, y, groups = augmented(keyed, (0, 20, 40), spec69)
    assert s.n == 2 * keyed.n and np.isfinite(y).all()
    with pytest.raises(ValueError, match="fit"):
        augmented(keyed, (40,), spec69)


def test_fitting_offsets_and_target_window_options(arcade_dir):
    import numpy as np

    from arena.augment import fitting_offsets
    from arena.config import ARCADE_ALGO_ARENA as SPEC
    from arena.generator import GeneratorModel, gk_vol
    from arena.universe import load_universe

    assert fitting_offsets(100, (0, 20, 40, 45), SPEC) == (0, 20, 40)
    u = load_universe(arcade_dir)
    base = GeneratorModel(SPEC, augment_offsets=(0, 20, 40))
    plain = base.fit(u.ins, u.y)
    target = GeneratorModel(SPEC, augment_offsets=(0, 20, 40), var_target=True).fit(u.ins, u.y)
    expected = float((u.y / gk_vol(base.prepare(u.ins), 49)).var())
    assert np.isclose(target.var_y, expected) and not np.isclose(plain.var_y, expected)
    weighted = GeneratorModel(SPEC, augment_offsets=(0, 20, 40), target_weight=3.0).fit(u.ins, u.y)
    assert not np.allclose(weighted.ridge.coef_, plain.ridge.coef_)
    assert np.isfinite(weighted.predict(u.oos)).all()
