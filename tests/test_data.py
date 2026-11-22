"""Data integrity after loading and cleaning (Phase 1 facts, now enforced)."""

import numpy as np

from src.data import (EXCLUDED_BATCH3, cycle_life_from_summary, find_continuations, load_batch,
                      unfinished)

# Facts about the paper's 124-cell set, used to check what the code derives from the data.
CONTINUED = {"b1c0": "b2c7", "b1c1": "b2c8", "b1c2": "b2c9", "b1c3": "b2c15", "b1c4": "b2c16"}
DROPPED_BATCH1 = ["b1c8", "b1c10", "b1c12", "b1c13", "b1c22"]


def test_124_cells_in_batch_order(cells):
    batches = [c["batch"] for c in cells.values()]
    assert len(cells) == 124
    assert [batches.count(b) for b in (1, 2, 3)] == [41, 43, 40]
    assert batches == sorted(batches)


def test_removed_cells_are_gone(cells):
    for k in DROPPED_BATCH1 + list(CONTINUED.values()) + EXCLUDED_BATCH3:
        assert k not in cells


def test_life_range(cells):
    life = np.array([c["cycle_life"] for c in cells.values()])
    assert life.min() == 148 and life.max() == 2237
    assert cells["b2c1"]["cycle_life"] == 148


def test_stored_life_matches_capacity_curve(cells):
    # Independent recomputation from the summary data, for every cell.
    bad = {k: (c["cycle_life"], cycle_life_from_summary(c)) for k, c in cells.items()
           if cycle_life_from_summary(c) != c["cycle_life"]}
    assert not bad, bad


def test_every_cell_has_the_first_100_cycles(cells):
    for k, c in cells.items():
        assert len(c["summary"]["cycle"]) >= 100, k
        assert {"9", "99"} <= set(c["cycles"]), k        # cycles 10 and 100
        assert c["cycles"]["99"]["Qdlin"].shape == (1000,), k


def test_cycle_numbers_increase(cells):
    # Includes the 5 merged batch-1/batch-2 cells, whose counter must not restart.
    for k, c in cells.items():
        assert np.all(np.diff(c["summary"]["cycle"]) > 0), k


def test_carryover_cells_were_merged(cells):
    for k in ["b1c0", "b1c1", "b1c2", "b1c3", "b1c4"]:
        assert "merged_from" in cells[k]


def test_unfinished_cells_found_from_data(cells):
    assert sorted(unfinished(load_batch(1))) == sorted(list(CONTINUED) + DROPPED_BATCH1)


def test_continuations_found_from_data(cells):
    # Derived from policy text + capacity continuity, never from a hard-coded list.
    assert find_continuations(load_batch(1), load_batch(2)) == CONTINUED


def test_joined_cell_lifetimes(cells):
    # Batch-1 record length + 1 + batch-2 record length (e.g. b1c0: 1189 + 1 + 662).
    b1, b2 = load_batch(1), load_batch(2)
    for k1, k2 in CONTINUED.items():
        expected = len(b1[k1]["summary"]["cycle"]) + 1 + len(b2[k2]["summary"]["cycle"])
        assert cells[k1]["cycle_life"] == expected, k1
