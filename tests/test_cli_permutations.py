"""Unit tests for :mod:`cli_permutations` utilities."""

import logging

from speaktome.util.cli_permutations import CLIArgumentMatrix
# --- END HEADER ---

logger = logging.getLogger(__name__)


def test_permutation_generation() -> None:
    """Enumerate permutations and verify expected combinations."""
    logger.info('test_permutation_generation start')
    matrix = CLIArgumentMatrix()
    matrix.add_option('--flag-a', [None])
    matrix.add_option('--num', [1, 2])
    combos = matrix.generate()
    assert combos == [['--flag-a', '--num', '1'], ['--flag-a', '--num', '2']]
    logger.info('test_permutation_generation end')
