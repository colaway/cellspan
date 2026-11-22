"""Error metrics, reported in CYCLES (not log units) so they mean something physically.

Models predict y = log10(cycle life); we convert back with 10**y before scoring.
Same definitions as the paper:
    RMSE = sqrt(mean((pred - true)^2))       in cycles
    MAPE = mean(|pred - true| / true) * 100  in percent
"""

import numpy as np
from sklearn.model_selection import LeaveOneOut, cross_val_predict

from src.split import from_target


def rmse(true_cycles, pred_cycles):
    true_cycles, pred_cycles = np.asarray(true_cycles), np.asarray(pred_cycles)
    return float(np.sqrt(np.mean((pred_cycles - true_cycles) ** 2)))


def mape(true_cycles, pred_cycles):
    true_cycles, pred_cycles = np.asarray(true_cycles), np.asarray(pred_cycles)
    return float(np.mean(np.abs(pred_cycles - true_cycles) / true_cycles) * 100)


def score(true_cycles, pred_y):
    """RMSE and MAPE from log10 predictions."""
    pred = from_target(pred_y)
    return {"rmse": rmse(true_cycles, pred), "mape": mape(true_cycles, pred)}


def loocv_predict(model, X, y):
    """Leave-one-out predictions: each cell predicted by a model fit on all the others.
    Uses only the data passed in (the training set), so it never touches the tests."""
    return cross_val_predict(model, X, y, cv=LeaveOneOut())
