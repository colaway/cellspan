"""Early-cycle features computed ONLY from cycles 1..100.

The core rule of this project: a feature may use nothing a real deployment would
not have at prediction time. So the API is split in two steps:

    window = early_window(cell)        # cut a cell down to cycles 1..100, drop the label
    feats  = compute_features(window)  # refuses anything that reaches past cycle 100

`compute_features` re-checks its input instead of trusting the caller, so passing a
full cell (or anything containing cycle 101+, or the cycle-life label) raises
FutureDataError rather than silently producing optimistic features.

Indexing reminder: cycle N is summary index N-1 and cycles key str(N-1).
"""

import numpy as np
import pandas as pd
from scipy import stats
from scipy.ndimage import median_filter

from src.split import target

LAST_CYCLE = 100  # use cycles up to and including this one
FIRST_CYCLE = 10  # Delta-Q reference cycle: Q_100(V) - Q_10(V)

# Paper's three models (Phase 6). Names refer to columns produced below.
FEATURE_SETS = {
    "variance": ["dq_log_var"],
    "discharge": ["dq_log_var", "dq_log_min", "dq_log_skew", "dq_log_kurt",
                  "q2", "qmax_minus_q2"],
    "full": ["dq_log_var", "dq_log_min", "fade_slope", "fade_intercept", "q2",
             "chargetime_mean", "tmax", "ir_min", "ir_diff"],
}

# A QD point further than this from the median of its 5-cycle neighbourhood is
# treated as a measurement glitch. Typical early cycle-to-cycle change is ~0.1 mAh;
# the glitches we found are 0.5-1.8 Ah, so 10 mAh separates them cleanly.
QD_GLITCH_AH = 0.010


class FutureDataError(ValueError):
    """Raised when feature extraction is handed data from beyond the allowed window."""


# ----------------------------------------------------------------------------
# Step 1: the window
# ----------------------------------------------------------------------------
def early_window(cell, last_cycle=LAST_CYCLE):
    """Return a copy of `cell` containing only cycles 1..last_cycle, and no label."""
    if cell["summary"]["cycle"].size < last_cycle:
        raise ValueError(f"cell has fewer than {last_cycle} cycles")
    return {
        "last_cycle": last_cycle,
        "Vdlin": cell["Vdlin"],
        "summary": {k: np.array(v[:last_cycle]) for k, v in cell["summary"].items()},
        "cycles": {k: v for k, v in cell["cycles"].items() if int(k) < last_cycle},
    }


def _check_window(window, last_cycle):
    """Fail loudly if the window contains anything a model may not see."""
    for forbidden in ("cycle_life", "n_cycles_recorded"):
        if forbidden in window:
            raise FutureDataError(f"window contains '{forbidden}': that leaks the label")
    s = window["summary"]
    if any(len(v) > last_cycle for v in s.values()):
        raise FutureDataError(f"summary arrays extend past cycle {last_cycle}")
    if np.nanmax(s["cycle"]) > last_cycle:
        raise FutureDataError(f"summary contains cycle {np.nanmax(s['cycle']):.0f} > {last_cycle}")
    later = [k for k in window["cycles"] if int(k) + 1 > last_cycle]
    if later:
        raise FutureDataError(f"per-cycle data for cycle(s) {int(later[0]) + 1}... > {last_cycle}")


# ----------------------------------------------------------------------------
# Cleaning helpers (each fixes a specific problem found in Phase 2/4)
# ----------------------------------------------------------------------------
def clean_capacity(qd):
    """Replace single-cycle glitches (e.g. b1c18: 2.88 Ah at cycle 40) by the local median."""
    qd = np.asarray(qd, dtype=float)
    local = median_filter(qd, size=5, mode="nearest")
    return np.where(np.abs(qd - local) > QD_GLITCH_AH, local, qd)


def clean_ir(ir):
    """IR = 0 is physically impossible: it is a placeholder for a missed measurement."""
    ir = np.asarray(ir, dtype=float)
    return np.where(ir > 0, ir, np.nan)


# ----------------------------------------------------------------------------
# Step 2: the features
# ----------------------------------------------------------------------------
def delta_q(window, first_cycle=FIRST_CYCLE, last_cycle=LAST_CYCLE):
    """Q_last(V) - Q_first(V) on the shared 1000-point voltage grid."""
    _check_window(window, last_cycle)
    if not 1 < first_cycle < last_cycle:
        raise ValueError("need 1 < first_cycle < last_cycle (cycle 1 is missing in batch 1)")
    c = window["cycles"]
    return c[str(last_cycle - 1)]["Qdlin"] - c[str(first_cycle - 1)]["Qdlin"]


def compute_features(window, first_cycle=FIRST_CYCLE, last_cycle=LAST_CYCLE):
    """All candidate features for one cell, as a dict."""
    _check_window(window, last_cycle)
    s = window["summary"]
    dq = delta_q(window, first_cycle, last_cycle)

    # Summary series over cycles 2..last_cycle (cycle 1 is a placeholder in batch 1).
    cyc = s["cycle"][1:last_cycle]
    qd = clean_capacity(s["QD"][1:last_cycle])
    ir = clean_ir(s["IR"][1:last_cycle])
    slope, intercept = np.polyfit(cyc, qd, 1)

    return {
        # --- Delta-Q(V) features. log10 because they span orders of magnitude.
        # ddof=1 / bias=True / fisher=False reproduce MATLAB's var/skewness/kurtosis.
        "dq_log_var": np.log10(np.var(dq, ddof=1)),
        "dq_log_min": np.log10(np.abs(dq.min())),
        "dq_log_skew": np.log10(np.abs(stats.skew(dq, bias=True))),
        "dq_log_kurt": np.log10(np.abs(stats.kurtosis(dq, fisher=False, bias=True))),
        # --- Capacity-fade features
        "q2": qd[0],
        "qmax_minus_q2": qd.max() - qd[0],
        "fade_slope": slope,
        "fade_intercept": intercept,
        # --- Other summary features
        # "First 5 cycles" = cycles 2-6 for every cell, since batch 1 has no cycle 1.
        "chargetime_mean": np.mean(s["chargetime"][1:6]),
        "tmax": np.max(s["Tmax"][1:last_cycle]),
        "ir_min": np.nanmin(ir),
        "ir_diff": ir[-1] - ir[0],
    }


def build_feature_table(cells, ids=None, first_cycle=FIRST_CYCLE, last_cycle=LAST_CYCLE):
    """DataFrame of features (one row per cell) plus the label columns.

    The label is attached *after* feature extraction, from the original cell,
    so it can never influence the features."""
    ids = list(cells) if ids is None else ids
    rows = {}
    for k in ids:
        feats = compute_features(early_window(cells[k], last_cycle), first_cycle, last_cycle)
        feats["cycle_life"] = cells[k]["cycle_life"]
        feats["y"] = target(cells[k]["cycle_life"])
        feats["batch"] = cells[k]["batch"]
        rows[k] = feats
    return pd.DataFrame.from_dict(rows, orient="index")
