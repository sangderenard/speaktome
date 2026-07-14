"""Tests for speaktome.core.noodle_explorer.

NoodleExplorer.grow is a documented stub (see the STUB block in
noodle_explorer.py) -- its DFS/BFS budget and termination policy aren't
designed yet. These tests cover the parts that ARE implemented (the
Noodle data shape) and confirm the stub fails loudly rather than silently,
per AGENTS/CODING_STANDARDS.md.
"""

import pytest

from speaktome.core.noodle_explorer import Direction, Noodle, NoodleExplorer


def test_noodle_starts_incomplete():
    n = Noodle(direction=Direction.FORWARD, tokens=[1, 2, 3])
    assert n.complete is False
    assert n.total_score is None


def test_noodle_finalize_marks_complete_and_records_score():
    n = Noodle(direction=Direction.BACKWARD, tokens=[4, 5])
    n.finalize(total_score=-1.23)
    assert n.complete is True
    assert n.total_score == -1.23


def test_noodle_tokens_are_reading_order_regardless_of_direction():
    # A backward noodle's tokens are still stored left-to-right; growth
    # order and reading order are not the same thing.
    n = Noodle(direction=Direction.BACKWARD, tokens=[7, 8, 9])
    assert n.tokens == [7, 8, 9]


@pytest.mark.stub
def test_grow_is_a_documented_stub_not_a_silent_noop():
    explorer = NoodleExplorer(model_wrapper=None, backward_scorer=None, choice_policy=None)
    with pytest.raises(NotImplementedError):
        explorer.grow(anchor_tokens=None, num_noodles=4)
