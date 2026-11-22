"""Load the three converted batches and clean them into the 124 cells used in the paper.

What cleaning means here, and how each step is decided:

1. Batch-1 cells that hadn't reached end of life when batch 1 stopped. Most tests were
   stopped right as capacity crossed 0.88 Ah, so a normal record ends just above 0.88.
   Ten batch-1 records end well above it (0.91 - 1.04 Ah). Found from the data.
2. Five of those ten were moved and kept cycling in batch 2. Their batch-2 records have a
   charging-policy text that is a cut-off copy of the batch-1 text (e.g. "80%)-3.6C") and
   start part-worn (~1.0 Ah instead of ~1.065 Ah for a new cell). Each is paired with the
   batch-1 cell whose last capacity best matches its first capacity, and the two records
   are joined into one cell. Found from the data.
3. The other five unfinished batch-1 cells have no continuation: their life is unknown, so
   they are dropped.
4. Six batch-3 cells are excluded, as in the paper's 124-cell set. Two of them (b3c23,
   b3c32) have no recorded cycle life; the dataset authors flagged the other four as noisy
   channels. This list can't be derived from the data, so it is written out.

Cycle life is then recomputed from each cell's capacity curve (cycle_life_from_summary).

Order matters: the paper's split (src/split.py) is defined by position, so cells stay in
file order, batch 1 then 2 then 3.
"""

import pickle
from pathlib import Path

import numpy as np
from scipy.optimize import linear_sum_assignment

from src.convert import KEEP_CYCLES

PROCESSED = Path(__file__).resolve().parents[1] / "data" / "processed"

NOMINAL_CAPACITY = 1.1                   # Ah
EOL_CAPACITY = 0.8 * NOMINAL_CAPACITY    # 0.88 Ah = end of life
STOP_TOLERANCE = 0.005                   # a record ending within 5 mAh of 0.88 counts as finished

EXCLUDED_BATCH3 = ["b3c2", "b3c23", "b3c32", "b3c37", "b3c42", "b3c43"]


def load_batch(batch):
    with open(PROCESSED / f"batch{batch}.pkl", "rb") as fp:
        return pickle.load(fp)


def valid_capacity(cell):
    """Discharge capacity per cycle, without the 0 placeholders for missing cycles."""
    qd = cell["summary"]["QD"]
    return qd[qd > 0]


def cycle_life_from_summary(cell):
    """Cycles until capacity drops below 0.88 Ah, from the summary data.

    Two quirks: missing cycles are stored as QD = 0 (ignored), and most tests were
    stopped as soon as capacity crossed 0.88 without saving that cycle, so a record
    ending just above 0.88 means life = last recorded cycle + 1.
    Returns NaN if the cell clearly never reached end of life.
    """
    qd, cyc = cell["summary"]["QD"], cell["summary"]["cycle"]
    keep = qd > 0
    qd, cyc = qd[keep], cyc[keep]
    below = np.nonzero(qd < EOL_CAPACITY)[0]
    if below.size:
        return float(cyc[below[0]])
    if qd[-1] < EOL_CAPACITY + STOP_TOLERANCE:
        return float(cyc[-1] + 1)
    return np.nan


def capacity_at_end(cell, n=30):
    return float(np.median(valid_capacity(cell)[-n:]))


def capacity_at_start(cell, n=30):
    return float(np.median(valid_capacity(cell)[1:n + 1]))   # skip the first, often odd, cycle


def unfinished(cells):
    """IDs of records that end clearly above end-of-life capacity."""
    return [k for k, c in cells.items() if valid_capacity(c)[-1] > EOL_CAPACITY + STOP_TOLERANCE]


def find_continuations(batch1, batch2):
    """{batch-1 id: batch-2 id} for cells that kept cycling in batch 2 (see step 2 above)."""
    open_ends = unfinished(batch1)
    pairs = {}
    # Group candidates by policy: a batch-2 record can only continue a batch-1 cell
    # whose policy text ends with the batch-2 record's (cut-off) text.
    for policy in sorted({c["charge_policy"] for c in batch2.values()}):
        b2_ids = [k for k, c in batch2.items() if c["charge_policy"] == policy]
        b1_ids = [k for k in open_ends
                  if batch1[k]["charge_policy"].endswith(policy) and batch1[k]["charge_policy"] != policy]
        if not b1_ids:
            continue
        # Pair by capacity continuity: minimise the total SQUARED gap between where each
        # batch-1 cell stopped and where a batch-2 record starts (Hungarian algorithm).
        # - Medians over 30 cycles, because single-cycle capacity is noisy.
        # - Squared, not absolute: every cell drops a few mAh over the pause, and with
        #   absolute gaps different pairings can tie exactly (we hit this: the 3.6C trio
        #   got mispaired). Squaring penalises one big jump more than two small ones,
        #   which favours the pairing that keeps the cells in order.
        cost = np.array([[(capacity_at_end(batch1[a]) - capacity_at_start(batch2[b])) ** 2
                          for b in b2_ids] for a in b1_ids])
        rows, cols = linear_sum_assignment(cost)
        pairs.update({b1_ids[r]: b2_ids[c] for r, c in zip(rows, cols)})
    return pairs


def join(first, second, second_id):
    """One cell from a record and its continuation. Batch 2 restarts the cycle counter,
    so its cycle numbers are shifted to follow on from the first record."""
    offset = len(first["summary"]["cycle"])
    summary = {k: np.concatenate([v, second["summary"][k] + (offset if k == "cycle" else 0)])
               for k, v in first["summary"].items()}
    cycles = dict(first["cycles"])
    if len(cycles) == first["n_cycles_recorded"]:     # first record wasn't cut short
        for j, cyc in enumerate(second["cycles"].values()):
            if len(cycles) >= KEEP_CYCLES:
                break
            cycles[str(first["n_cycles_recorded"] + j)] = cyc
    return {**first, "summary": summary, "cycles": cycles,
            "n_cycles_recorded": first["n_cycles_recorded"] + second["n_cycles_recorded"],
            "merged_from": second_id}


def load_all():
    """{cell id: cell} for the paper's 124 cells, in file order."""
    b1, b2, b3 = load_batch(1), load_batch(2), load_batch(3)

    pairs = find_continuations(b1, b2)
    for k1, k2 in pairs.items():
        b1[k1] = join(b1[k1], b2.pop(k2), k2)
    for k in unfinished(b1):                 # what's still unfinished has no continuation
        del b1[k]
    for k in EXCLUDED_BATCH3:
        del b3[k]

    cells = {}
    for batch, group in ((1, b1), (2, b2), (3, b3)):
        for k, c in group.items():
            c["batch"] = batch
            c["cycle_life"] = cycle_life_from_summary(c)
            cells[k] = c
    return cells
