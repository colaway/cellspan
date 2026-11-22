"""The paper's split: sizes, rules, no overlap, determinism."""

import numpy as np
import pytest

from src.split import OUTLIER, from_target, paper_split, target


def test_target_round_trip():
    life = np.array([148.0, 550.0, 2237.0])
    np.testing.assert_allclose(from_target(target(life)), life)


def test_sizes_with_and_without_outlier(cells):
    full = paper_split(cells, drop_outlier=False)
    assert {k: len(v) for k, v in full.items()} == {"train": 41, "test1": 43, "test2": 40}
    dropped = paper_split(cells)
    assert len(dropped["test1"]) == 42 and OUTLIER not in dropped["test1"]
    assert OUTLIER in full["test1"]


def test_sets_are_disjoint_and_cover_all_cells(cells):
    s = paper_split(cells, drop_outlier=False)
    all_ids = s["train"] + s["test1"] + s["test2"]
    assert len(all_ids) == len(set(all_ids)) == 124
    assert set(all_ids) == set(cells)


def test_position_rules(cells):
    ids = list(cells)
    s = paper_split(cells, drop_outlier=False)
    assert s["train"] == [ids[i] for i in range(1, 83, 2)]          # odd positions, not 83
    assert s["test1"] == [ids[i] for i in range(0, 84, 2)] + [ids[83]]
    assert all(cells[k]["batch"] == 3 for k in s["test2"])
    assert all(cells[k]["batch"] in (1, 2) for k in s["train"] + s["test1"])


def test_split_is_deterministic(cells):
    assert paper_split(cells) == paper_split(cells)


def test_wrong_cell_list_is_refused(cells):
    smaller = dict(list(cells.items())[:-1])
    with pytest.raises(AssertionError):
        paper_split(smaller)
    shuffled = dict(reversed(list(cells.items())))   # batch 3 first breaks the position rule
    with pytest.raises(AssertionError):
        paper_split(shuffled)
