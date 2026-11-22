"""Feature VALUES (the guard tests are in test_features.py).

A synthetic cell with hand-computable answers checks each formula and indexing
convention; the real data checks that features.csv is reproducible from raw.
"""

import numpy as np
import pandas as pd
import pytest

from src.features import (FEATURE_SETS, build_feature_table, clean_capacity, clean_ir,
                          compute_features, early_window)
from tests.conftest import make_synthetic_cell

# Delta-Q pattern [-1,-1,-1,-5] mAh repeated 250x. Its moments by hand:
#   mean -2; deviations 1,1,1,-3 -> m2 = 12/4 = 3, m3 = -24/4 = -6, m4 = 84/4 = 21
#   skewness = m3 / m2^1.5 = -1.1547; kurtosis (MATLAB, not minus 3) = m4 / m2^2 = 2.3333
PATTERN = np.tile([-1.0, -1.0, -1.0, -5.0], 250) * 1e-3


@pytest.fixture(scope="module")
def synth():
    return compute_features(early_window(make_synthetic_cell(dq=PATTERN)))


def test_delta_q_variance_uses_n_minus_1(synth):
    expected = 3e-6 * 1000 / 999          # MATLAB var divides by N-1
    assert synth["dq_log_var"] == pytest.approx(np.log10(expected))


def test_delta_q_min(synth):
    assert synth["dq_log_min"] == pytest.approx(np.log10(5e-3))


def test_delta_q_skew_and_kurtosis_conventions(synth):
    assert synth["dq_log_skew"] == pytest.approx(np.log10(6 / 3 ** 1.5))
    assert synth["dq_log_kurt"] == pytest.approx(np.log10(21 / 9))   # 2.33, not -0.67


def test_capacity_features(synth):
    assert synth["q2"] == pytest.approx(1.08)               # cycle 2, not the cycle-1 placeholder
    assert synth["fade_slope"] == pytest.approx(-1e-4)
    assert synth["fade_intercept"] == pytest.approx(1.08 + 2e-4)  # line extrapolated to cycle 0
    assert synth["qmax_minus_q2"] == pytest.approx(0.0)     # capacity only falls


def test_other_summary_features(synth):
    assert synth["chargetime_mean"] == pytest.approx(11.0)  # cycles 2-6
    assert synth["tmax"] == pytest.approx(35.0)             # cycle-1 value of 99 ignored
    assert synth["ir_min"] == pytest.approx(0.016)          # the zero at cycle 12 ignored
    assert synth["ir_diff"] == pytest.approx(0.001)         # cycle 100 minus cycle 2


def test_clean_capacity_removes_only_glitches():
    qd = 1.08 - 1e-4 * np.arange(50)
    spiky = qd.copy()
    spiky[20] = 2.88                                         # like b1c18 at cycle 40
    cleaned = clean_capacity(spiky)
    assert abs(cleaned[20] - qd[20]) < 1e-3
    np.testing.assert_allclose(np.delete(cleaned, 20), np.delete(qd, 20))
    np.testing.assert_allclose(clean_capacity(qd), qd)      # clean data untouched


def test_clean_ir_zero_becomes_missing():
    out = clean_ir([0.016, 0.0, 0.017])
    assert np.isnan(out[1]) and out[0] == 0.016


def test_short_cell_is_refused():
    cell = make_synthetic_cell()
    cell["summary"] = {k: v[:99] for k, v in cell["summary"].items()}  # died at cycle 99
    with pytest.raises(ValueError):
        early_window(cell)


# --- real data -------------------------------------------------------------------
def test_features_csv_is_reproducible(cells, features):
    fresh = build_feature_table(cells)
    pd.testing.assert_frame_equal(fresh, features.loc[fresh.index], check_dtype=False,
                                  check_exact=False, rtol=1e-9)


def test_features_are_finite_and_complete(features):
    cols = sorted({f for fs in FEATURE_SETS.values() for f in fs})
    assert features[cols].notna().all().all()
    assert np.isfinite(features[cols].to_numpy()).all()
    assert len(features) == 124


def test_a_cell_features_do_not_depend_on_other_cells(cells, features):
    # Features are per-cell: computing one cell alone must give the same row.
    one = build_feature_table(cells, ids=["b3c10"])
    pd.testing.assert_frame_equal(one, features.loc[["b3c10"]], check_dtype=False)


def test_physically_plausible_ranges(features):
    assert features.q2.between(1.0, 1.15).all()             # nominal 1.1 Ah
    assert features.ir_min.between(0.01, 0.03).all()        # ohms
    assert features.tmax.between(30, 45).all()              # chamber at 30 C
    assert (features.dq_log_var < 0).all()                  # var of a few mAh << 1


def test_key_feature_still_predicts_life(cells, split, features):
    # The paper's central finding, as a regression guard: r ~ -0.9 on train.
    tr = features.loc[split["train"]]
    assert np.corrcoef(tr.dq_log_var, tr.y)[0, 1] < -0.85
