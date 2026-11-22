"""Metrics: checked against numbers worked out by hand."""

import numpy as np
import pytest
from sklearn.dummy import DummyRegressor

from src.evaluate import loocv_predict, mape, rmse, score


def test_rmse_by_hand():
    # errors +10 and -20 -> sqrt((100 + 400) / 2)
    assert rmse([100, 200], [110, 180]) == pytest.approx(np.sqrt(250))


def test_mape_by_hand():
    # 10% and 10% off
    assert mape([100, 200], [110, 180]) == pytest.approx(10.0)


def test_perfect_prediction_scores_zero():
    life = np.array([300.0, 800.0, 1500.0])
    s = score(life, np.log10(life))
    assert s["rmse"] == pytest.approx(0, abs=1e-9) and s["mape"] == pytest.approx(0, abs=1e-9)


def test_score_converts_from_log10():
    # predicting log10(1100) for a 1000-cycle cell = 100 cycles / 10% off, not 0.04 log units
    s = score([1000.0], [np.log10(1100.0)])
    assert s["rmse"] == pytest.approx(100.0) and s["mape"] == pytest.approx(10.0)


def test_mape_is_relative_rmse_is_absolute():
    # Same 100-cycle miss: large in % for a short cell, small for a long one.
    assert mape([200], [300]) == pytest.approx(50.0)
    assert mape([2000], [2100]) == pytest.approx(5.0)
    assert rmse([200], [300]) == rmse([2000], [2100]) == pytest.approx(100.0)


def test_loocv_really_leaves_one_out():
    # A mean model's LOO prediction for cell i is the mean of the OTHER cells.
    # If cell i leaked into its own fit, we'd get the overall mean instead.
    y = np.array([1.0, 2.0, 3.0, 10.0])
    p = loocv_predict(DummyRegressor(), np.zeros((4, 1)), y)
    expected = (y.sum() - y) / (len(y) - 1)
    np.testing.assert_allclose(p, expected)
