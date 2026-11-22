"""Models: behaviour on synthetic data, the frozen models, and end-to-end numbers."""

import warnings

import numpy as np
import pytest
from sklearn.exceptions import ConvergenceWarning

from src.evaluate import score
from src.features import FEATURE_SETS
from src.models import describe, make_classifier, make_elastic_net
from src.split import OUTLIER, paper_split

warnings.filterwarnings("ignore", category=ConvergenceWarning)


# --- synthetic: does the procedure do what we think? -----------------------------
def test_elastic_net_finds_true_signal_and_drops_noise():
    rng = np.random.default_rng(0)
    X = rng.normal(size=(60, 6))
    y = 3.0 - 0.5 * X[:, 0] + 0.3 * X[:, 1] + rng.normal(scale=0.05, size=60)
    d = describe(make_elastic_net().fit(X, y), list("abcdef"))
    assert d["coefs"]["a"] < -0.3 and d["coefs"]["b"] > 0.15       # right signs and sizes
    assert all(abs(d["coefs"][k]) < 0.05 for k in "cdef")           # noise near zero


def test_scaling_does_not_change_predictions():
    # The scaler is inside the pipeline, so measuring a feature in mAh instead of Ah
    # (x1000) must give the same predictions. Without it, the penalty would differ.
    rng = np.random.default_rng(1)
    X = rng.normal(size=(50, 3))
    y = X @ [0.4, -0.2, 0.0] + rng.normal(scale=0.1, size=50)
    X2 = X.copy()
    X2[:, 0] *= 1000
    p1 = make_elastic_net().fit(X, y).predict(X)
    p2 = make_elastic_net().fit(X2, y).predict(X2)
    np.testing.assert_allclose(p1, p2, rtol=1e-4, atol=1e-6)


def test_same_seed_same_model():
    rng = np.random.default_rng(2)
    X, y = rng.normal(size=(40, 4)), rng.normal(size=40)
    a, b = make_elastic_net(seed=3).fit(X, y)[-1], make_elastic_net(seed=3).fit(X, y)[-1]
    assert a.alpha_ == b.alpha_ and np.array_equal(a.coef_, b.coef_)


def test_classifier_on_separable_data():
    rng = np.random.default_rng(3)
    X = rng.normal(size=(60, 2))
    y = (X[:, 0] > 0).astype(int)
    clf = make_classifier().fit(X, y)
    p = clf.predict_proba(X)[:, 1]
    assert ((p > 0.5) == y).mean() > 0.95 and p.min() >= 0 and p.max() <= 1


# --- the frozen models -------------------------------------------------------------
def test_frozen_models_were_trained_on_train_cells_only(cells, frozen, features):
    split = paper_split(cells)
    assert frozen["train_ids"] == split["train"]
    assert not set(frozen["train_ids"]) & (set(split["test1"]) | set(split["test2"]) | {OUTLIER})


def test_refitting_on_train_reproduces_frozen_model(frozen, features):
    # Strongest evidence of no leakage: train cells alone are enough to recreate
    # exactly the frozen coefficients. (Full model; the other two are faster versions.)
    m = frozen["models"]["full"]
    tr = features.loc[frozen["train_ids"]]
    refit = make_elastic_net(seed=0).fit(tr[m["features"]].values, tr.y.values)
    np.testing.assert_allclose(refit[-1].coef_, m["pipeline"][-1].coef_, atol=1e-10)
    assert refit[-1].alpha_ == pytest.approx(m["pipeline"][-1].alpha_)


def test_frozen_feature_lists_match_code(frozen):
    for name, m in frozen["models"].items():
        assert m["features"] == FEATURE_SETS[name]


# Phase 7 numbers, as published in README: MAPE %, test sets.
EXPECTED_MAPE = {  # (train, primary without b2c1, secondary)
    "variance": (14.2, 13.2, 11.4),
    "discharge": (9.4, 9.6, 9.1),
    "full": (7.3, 9.0, 12.8),
}


@pytest.mark.parametrize("name", list(EXPECTED_MAPE))
def test_end_to_end_numbers_match_the_report(name, cells, frozen, features):
    split = paper_split(cells)
    m = frozen["models"][name]
    for ids, expected in zip((split["train"], split["test1"], split["test2"]), EXPECTED_MAPE[name]):
        df = features.loc[ids]
        got = score(df.cycle_life, m["pipeline"].predict(df[m["features"]].values))["mape"]
        assert got == pytest.approx(expected, abs=0.05), (name, got, expected)


def test_one_cell_prediction_does_not_depend_on_batch(frozen, features):
    # Predicting a cell alone or among 123 others must give the same number.
    m = frozen["models"]["full"]
    X = features[m["features"]].values
    together = m["pipeline"].predict(X)
    alone = np.array([m["pipeline"].predict(X[i:i + 1])[0] for i in range(len(X))])
    np.testing.assert_allclose(together, alone)


def test_physics_direction(frozen, features):
    # More change in the discharge curve by cycle 100 -> shorter predicted life.
    for name in ("variance", "full"):
        m = frozen["models"][name]
        x = features[m["features"]].median().values.reshape(1, -1)
        worse = x.copy()
        worse[0, m["features"].index("dq_log_var")] += 0.5
        assert m["pipeline"].predict(worse)[0] < m["pipeline"].predict(x)[0], name


def test_tiny_measurement_noise_barely_moves_predictions(frozen, features):
    rng = np.random.default_rng(4)
    m = frozen["models"]["full"]
    X = features[m["features"]].values
    noisy = X * (1 + rng.normal(scale=1e-4, size=X.shape))      # 0.01% noise
    life, life_noisy = 10 ** m["pipeline"].predict(X), 10 ** m["pipeline"].predict(noisy)
    assert np.max(np.abs(life_noisy / life - 1)) < 0.01             # < 1% change


def test_missing_feature_is_an_error_not_a_guess(frozen, features):
    m = frozen["models"]["full"]
    X = features[m["features"]].values[:1].copy()
    X[0, 0] = np.nan
    with pytest.raises(ValueError):
        m["pipeline"].predict(X)


def test_predictions_are_always_positive_and_finite(frozen, features):
    # Even far outside the training range (extrapolation), 10**y stays a valid life.
    m = frozen["models"]["full"]
    X = features[m["features"]].values
    extreme = np.vstack([X.min(0) - 3 * X.std(0), X.max(0) + 3 * X.std(0)])
    life = 10 ** m["pipeline"].predict(extreme)
    assert np.all(np.isfinite(life)) and np.all(life > 0)
