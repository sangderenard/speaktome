"""CLI entry-point tests.

These quick checks confirm that the command-line interface responds to ``-h``
and that a few simple argument permutations run without error when optional
dependencies are available.
"""

import subprocess
import sys
import pytest

from speaktome.util.cli_permutations import CLIArgumentMatrix
# --- END HEADER ---


def test_help_message() -> None:
    """Invoke ``--help`` and ensure the usage banner appears."""
    result = subprocess.run([
        sys.executable,
        '-m', 'speaktome.speaktome',
        '-h'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'usage:' in result.stdout.lower()


def test_basic_combinations() -> None:
    """Run a minimal CLI cycle with a few argument permutations."""
    pytest.importorskip('torch', reason='CLI requires torch for full run')
    pytest.importorskip('transformers', reason='CLI requires transformers for full run with torch')
    matrix = CLIArgumentMatrix()
    matrix.add_option('--max_steps', [1])
    matrix.add_option('--safe_mode', [None])
    combos = matrix.generate()
    for combo in combos:
        result = subprocess.run([
            sys.executable,
            '-m', 'speaktome.speaktome',
            *combo,
            'hi'
        ], capture_output=True, text=True)
        assert result.returncode == 0
