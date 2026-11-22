"""Statistical sanity checks: can the pipeline fool itself?

If the modelling procedure leaked information (e.g. scaled with held-out rows, or
picked hyperparameters on them), it could look good even on MEANINGLESS labels.
So we shuffle the labels and check the honest error falls to no-skill level.
"""

import warnings

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor
from sklearn.exceptions import ConvergenceWarning
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import KFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler

from src.evaluate import loocv_predict, score
from src.features import FEATURE_SETS

warnings.filterwarnings("ignore", category=ConvergenceWarning)


@pytest.fixture(scope="module")
def train(split, features):
    return features.loc[split["train"]]


def no_skill_mape(tr):
    return score(tr.cycle_life, loocv_predict(DummyRegressor(), tr[["q2"]].values, tr.y.values))["mape"]


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_shuffled_labels_give_no_skill_linear(train, seed):
    y = np.random.default_rng(seed).permutation(train.y.values)
    p = loocv_predict(make_pipeline(StandardScaler(), LinearRegression()),
                      train[FEATURE_SETS["full"]].values, y)
    shuffled = score(10 ** y, p)["mape"]
    assert shuffled > 0.9 * no_skill_mape(train), shuffled


def test_shuffled_labels_give_no_skill_elastic_net(train):
    # Whole procedure (scaling + alpha/l1 search) inside an outer 4-fold CV.
    from src.models import make_elastic_net
    y = np.random.default_rng(0).permutation(train.y.values)
    p = cross_val_predict(make_elastic_net(n_repeats=3), train[FEATURE_SETS["full"]].values, y,
                          cv=KFold(4, shuffle=True, random_state=0))
    shuffled = score(10 ** y, p)["mape"]
    assert shuffled > 0.9 * no_skill_mape(train), shuffled


def test_real_labels_beat_no_skill_by_far(train):
    # The flip side: with the true labels the same procedure must find real signal.
    p = loocv_predict(make_pipeline(StandardScaler(), LinearRegression()),
                      train[["dq_log_var"]].values, train.y.values)
    assert score(train.cycle_life, p)["mape"] < 0.6 * no_skill_mape(train)


def test_test_features_resemble_train_features(split, features):
    # Primary test should be the same population as train (interpolation): no
    # feature mean differs by more than 1 training standard deviation.
    tr, te = features.loc[split["train"]], features.loc[split["test1"]]
    cols = FEATURE_SETS["full"]
    z = ((te[cols].mean() - tr[cols].mean()) / tr[cols].std()).abs()
    assert (z < 1).all(), z.round(2).to_dict()


def test_the_shuffle_check_would_catch_a_real_leak(train):
    # Test of the test. A classic leak: choose features using ALL labels (including
    # each held-out cell's), then cross-validate. On pure-noise features and shuffled
    # labels, an honest procedure scores no-skill; this leaky one looks skilful,
    # so the shuffle check above would flag it.
    rng = np.random.default_rng(0)
    y = rng.permutation(train.y.values)
    noise = rng.normal(size=(len(y), 500))
    r = np.abs([np.corrcoef(noise[:, j], y)[0, 1] for j in range(noise.shape[1])])
    leaky_X = noise[:, np.argsort(r)[-5:]]                 # picked with the held-out labels
    p = loocv_predict(LinearRegression(), leaky_X, y)
    assert score(10 ** y, p)["mape"] < 0.9 * no_skill_mape(train)
