"""Convert the raw .mat batch files into one pickle per batch.

The .mat files are MATLAB v7.3, which is HDF5 underneath. Layout, as found by opening
them with h5py:

    batch/                  one row per cell, each entry is a reference to the real data
        cycle_life          number (NaN if the cell never reached end of life)
        policy_readable     charging policy text, stored as UTF-16 code units
        Vdlin               1000-point voltage grid used for Qdlin / Tdlin
        summary             group: one value per cycle (IR, QDischarge, Tmax, ...)
        cycles              group: one reference per cycle to time series (V, I, Qd, ...)
        barcode, channel_id MATLAB "string" objects; h5py can't decode them, so skipped

Per-cycle time series are only kept for the first KEEP_CYCLES cycles. Features only
use cycles 1-100, and keeping every cycle of every cell makes files too big to load
comfortably in 16 GB of RAM. Summary data is kept for the whole life of the cell.

Usage:
    python -m src.convert          # all three batches
    python -m src.convert 2        # just batch 2
"""

import pickle
import sys
import time
from pathlib import Path

import h5py
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
RAW = ROOT / "data" / "raw"
PROCESSED = ROOT / "data" / "processed"

RAW_FILES = {
    1: "2017-05-12_batchdata_updated_struct_errorcorrect.mat",
    2: "2017-06-30_batchdata_updated_struct_errorcorrect.mat",
    3: "2018-04-12_batchdata_updated_struct_errorcorrect.mat",
}
KEEP_CYCLES = 300

# Names in the file -> shorter names used in this project.
SUMMARY_NAMES = {"cycle": "cycle", "QDischarge": "QD", "QCharge": "QC", "IR": "IR",
                 "Tmax": "Tmax", "Tavg": "Tavg", "Tmin": "Tmin", "chargetime": "chargetime"}
CYCLE_NAMES = {"t": "t", "V": "V", "I": "I", "T": "T", "Qc": "Qc", "Qd": "Qd",
               "Qdlin": "Qdlin", "Tdlin": "Tdlin", "discharge_dQdV": "dQdV"}


class BatchFile:
    """Read-only view of one batch file. Every field is reached through an HDF5
    object reference, so `get` follows the reference and flattens the result."""

    def __init__(self, path):
        self.h5 = h5py.File(path, "r")
        self.table = self.h5["batch"]

    def __len__(self):
        return self.table["cycle_life"].shape[0]

    def get(self, ref):
        return np.asarray(self.h5[ref][()]).ravel()

    def field(self, name, i):
        return self.get(self.table[name][i, 0])

    def text(self, name, i):
        return "".join(map(chr, self.field(name, i)))

    def cell(self, i):
        summary_group = self.h5[self.table["summary"][i, 0]]
        summary = {short: np.asarray(summary_group[name][()]).ravel()
                   for name, short in SUMMARY_NAMES.items()}

        cycle_group = self.h5[self.table["cycles"][i, 0]]
        n_cycles = cycle_group["V"].shape[0]
        cycles = {str(j): {short: self.get(cycle_group[name][j, 0])
                           for name, short in CYCLE_NAMES.items()}
                  for j in range(min(n_cycles, KEEP_CYCLES))}

        return {
            "cycle_life": float(self.field("cycle_life", i)[0]),
            "charge_policy": self.text("policy_readable", i),
            "Vdlin": self.field("Vdlin", i),
            "summary": summary,
            "cycles": cycles,               # keys are str(cycle number - 1)
            "n_cycles_recorded": n_cycles,  # before cutting to KEEP_CYCLES
        }


def convert(batch):
    data = BatchFile(RAW / RAW_FILES[batch])
    cells = {}
    for i in range(len(data)):
        cells[f"b{batch}c{i}"] = data.cell(i)
        print(f"\r  batch {batch}: cell {i + 1}/{len(data)}", end="", flush=True)
    print()
    return cells


if __name__ == "__main__":
    PROCESSED.mkdir(parents=True, exist_ok=True)
    for b in [int(a) for a in sys.argv[1:]] or [1, 2, 3]:
        start = time.time()
        out = PROCESSED / f"batch{b}.pkl"
        with open(out, "wb") as fp:
            pickle.dump(convert(b), fp, protocol=pickle.HIGHEST_PROTOCOL)
        print(f"  -> {out.name}, {out.stat().st_size / 1e6:.0f} MB, {time.time() - start:.0f}s")
