import numpy as np
import pytest

from arena.config import ARCADE_ALGO_ARENA as SPEC
from arena.cv import cross_validate, kfold
from arena.data import subset, truncate
from arena.learners import OwedTfidfModel, TemplateAgeMLP
from arena.modelapi import as_fit_predict_with_unlabeled
from arena.universe import load_universe

MODELS = [OwedTfidfModel(SPEC, min_df=1), TemplateAgeMLP(SPEC, max_iter=30, augment_offsets=(0, 20))]


@pytest.mark.parametrize("model", MODELS)
def test_learner_never_peeks_past_halfway(arcade_dir, model):
    u = load_universe(arcade_dir)
    fitted = model.fit(subset(u.ins, np.arange(150)), u.y[:150], unlabeled=u.oos)
    test = subset(u.ins, np.arange(150, u.ins.n))
    np.testing.assert_allclose(fitted.predict(test), fitted.predict(truncate(test, 49)))


@pytest.mark.parametrize("model", MODELS[:1])
def test_learner_finds_the_planted_signal(arcade_dir, model):
    u = load_universe(arcade_dir)
    fp = as_fit_predict_with_unlabeled(model, u.ins, u.y, unlabeled=u.oos)
    assert cross_validate(fp, u.y, kfold(u.ins.n, 4)).mean > 1.0
