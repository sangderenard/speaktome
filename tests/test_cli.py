import subprocess
import sys


def test_help_message():
    result = subprocess.run([
        sys.executable,
        '-m', 'speaktome.speaktome',
        '-h'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'usage:' in result.stdout.lower()
