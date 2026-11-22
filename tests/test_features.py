"""Tests for the future-data guard in src/features.py.

Each test tries a different way of letting information from after cycle 100
reach the feature code, and checks that it is refused. Run: pytest -q
"""

import numpy as np
import pytest

from src.features import FutureDataError, compute_features, delta_q, early_window
from tests.conftest import make_synthetic_cell


@pytest.fixture(params=["synthetic", "real"])
def cell(request):
    """Every guard test runs twice: on a synthetic 120-cycle cell (needs no data, so it
    runs in CI) and on real cell b1c5 (skipped if the dataset isn't downloaded)."""
    if request.param == "synthetic":
        return make_synthetic_cell(dq=np.tile([-1.0, -1.0, -1.0, -5.0], 250) * 1e-3)
    return request.getfixturevalue("cells")["b1c5"]


def test_full_cell_is_rejected(cell):
    # The most likely mistake: forgetting to cut the cell down at all.
    with pytest.raises(FutureDataError):
        compute_features(cell)


def test_label_is_rejected(cell):
    w = early_window(cell)
    w["cycle_life"] = cell["cycle_life"]
    with pytest.raises(FutureDataError, match="label"):
        compute_features(w)


def test_cycle_101_summary_is_rejected(cell):
    w = early_window(cell)
    w["summary"] = {k: np.array(v[:101]) for k, v in cell["summary"].items()}
    with pytest.raises(FutureDataError):
        compute_features(w)


def test_cycle_101_curve_is_rejected(cell):
    w = early_window(cell)
    w["cycles"]["100"] = cell["cycles"]["100"]  # key '100' = cycle 101
    with pytest.raises(FutureDataError):
        compute_features(w)


def test_window_is_accepted_and_finite(cell):
    feats = compute_features(early_window(cell))
    assert all(np.isfinite(v) for v in feats.values())


def test_features_ignore_data_after_cycle_100(cell):
    # Changing everything after cycle 100 must not change any feature.
    tampered = {**cell,
                "summary": {k: v.copy() for k, v in cell["summary"].items()},
                "cycles": dict(cell["cycles"])}
    for v in tampered["summary"].values():
        v[100:] = 999.0
    tampered["cycles"]["150"] = {"Qdlin": np.full(1000, 999.0)}
    assert compute_features(early_window(tampered)) == compute_features(early_window(cell))


def test_delta_q_uses_cycles_100_and_10(cell):
    # Guards the off-by-one: cycle N lives at key str(N-1).
    dq = delta_q(early_window(cell))
    expected = cell["cycles"]["99"]["Qdlin"] - cell["cycles"]["9"]["Qdlin"]
    np.testing.assert_array_equal(dq, expected)


def test_short_window_rejects_later_cycles(cell):
    # Phase 8 uses windows as short as 5 cycles; the guard must follow last_cycle.
    with pytest.raises(FutureDataError):
        compute_features(early_window(cell, 6), first_cycle=4, last_cycle=5)
    cell = {**cell, "cycles": dict(cell["cycles"])}
    # Make sure cycles 4 and 5 differ (the synthetic cell's curves are all identical,
    # which would give Delta-Q = 0 and log10(0) = -inf).
    cell["cycles"]["4"] = {"Qdlin": cell["cycles"]["4"]["Qdlin"] - np.linspace(0, 1e-3, 1000)}
    feats = compute_features(early_window(cell, 5), first_cycle=4, last_cycle=5)
    assert np.isfinite(feats["dq_log_var"])
