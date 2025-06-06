import subprocess
import sys

from speaktome.cli_permutations import CLIArgumentMatrix


def test_help_message():
    result = subprocess.run([
        sys.executable,
        '-m', 'speaktome.speaktome',
        '-h'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'usage:' in result.stdout.lower()


def test_basic_combinations():
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
