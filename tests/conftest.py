"""Shared fixtures. Loading the 124 cells takes ~3 s, so it happens once per test run.

Tests that need the real data are skipped (not failed) if data/processed is empty,
so the pure-logic tests still run on a fresh clone without the 8 GB download.
"""

import pickle
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

ROOT = Path(__file__).resolve().parents[1]
PROCESSED = ROOT / "data" / "processed"


def _need(path):
    if not path.exists():
        pytest.skip(f"needs {path.relative_to(ROOT)} (run the pipeline first)")


@pytest.fixture(scope="session")
def cells():
    _need(PROCESSED / "batch1.pkl")
    from src.data import load_all
    return load_all()


@pytest.fixture(scope="session")
def split(cells):
    from src.split import paper_split
    return paper_split(cells)


@pytest.fixture(scope="session")
def features():
    _need(PROCESSED / "features.csv")
    return pd.read_csv(PROCESSED / "features.csv", index_col=0)


@pytest.fixture(scope="session")
def frozen():
    _need(PROCESSED / "models.pkl")
    with open(PROCESSED / "models.pkl", "rb") as fp:
        return pickle.load(fp)


def make_synthetic_cell(n_cycles=120, dq=None):
    """A fake cell whose every feature can be computed by hand.

    - QD falls exactly linearly: 1.08 - 1e-4 * (cycle - 2), so slope/intercept are known.
    - Cycle 1 has placeholder values (as in batch 1) that features must ignore.
    - Qdlin: a flat base curve; cycle 100 = base + dq, so Delta-Q = dq exactly.
    """
    cyc = np.arange(1, n_cycles + 1, dtype=float)
    qd = 1.08 - 1e-4 * (cyc - 2)
    tmax = np.full(n_cycles, 35.0)
    ir = np.full(n_cycles, 0.016)
    ir[99] = 0.017            # cycle 100
    ir[11] = 0.0              # cycle 12: a "missed measurement" zero
    chargetime = np.full(n_cycles, 10.0)
    chargetime[1:6] = [9, 10, 11, 12, 13]   # cycles 2-6 -> mean 11
    qd[0], tmax[0], chargetime[0] = 0.0, 99.0, 0.0   # cycle 1 placeholder
    base = np.linspace(1.1, 0.0, 1000)
    dq = np.zeros(1000) if dq is None else dq
    cycles = {str(j): {"Qdlin": base.copy()} for j in range(n_cycles)}
    cycles["99"] = {"Qdlin": base + dq}
    return {
        "cycle_life": 500.0, "batch": 1, "Vdlin": np.linspace(3.6, 2.0, 1000),
        "summary": {"cycle": cyc, "QD": qd, "Tmax": tmax, "IR": ir, "chargetime": chargetime},
        "cycles": cycles,
    }
