"""Target definition and the paper's train / primary-test / secondary-test split.

The paper's split is defined by *position* in the cleaned cell list (batch 1, then
2, then 3; 84 cells in batches 1+2):

    primary test   = even positions 0, 2, ..., 82 of batches 1+2, plus position 83
    train          = odd positions 1, 3, ..., 81
    secondary test = all of batch 3

Positions are fragile (any change in cell order silently changes the split), so
we resolve them to cell IDs once and check the result with assertions.
"""

import numpy as np

# The 148-cycle cell. Its capacity collapses within the first 100 cycles
# (-104 mAh by cycle 100, vs a median of +2 mAh), so it behaves unlike any
# other cell; the paper reports primary-test results with it excluded.
OUTLIER = "b2c1"


def target(cycle_life):
    """What we predict: log10(cycle life). Use `from_target` to go back."""
    return np.log10(cycle_life)


def from_target(y):
    return 10.0 ** np.asarray(y)


def paper_split(cells, drop_outlier=True):
    """Return {'train': [...], 'test1': [...], 'test2': [...]} lists of cell IDs."""
    ids = list(cells)
    n12 = sum(cells[k]["batch"] in (1, 2) for k in ids)  # 84 cells in batches 1+2
    assert n12 == 84 and len(ids) == 124, "cell list differs from the paper's 124 cells"
    # The split relies on batches being contiguous and in order 1, 2, 3.
    assert [cells[k]["batch"] for k in ids] == sorted(cells[k]["batch"] for k in ids)

    test1_idx = list(range(0, n12, 2)) + [83]
    train_idx = list(range(1, n12 - 1, 2))
    split = {
        "train": [ids[i] for i in train_idx],
        "test1": [ids[i] for i in test1_idx],
        "test2": ids[n12:],
    }
    sizes = {name: len(v) for name, v in split.items()}
    assert sizes == {"train": 41, "test1": 43, "test2": 40}, sizes
    # Leakage guard: no cell may appear in more than one set.
    all_ids = sum(split.values(), [])
    assert len(all_ids) == len(set(all_ids)) == 124

    if drop_outlier:
        for name in split:
            split[name] = [k for k in split[name] if k != OUTLIER]
    return split
