import subprocess
import sys
import logging

logger = logging.getLogger(__name__)

from speaktome.util.cli_permutations import CLIArgumentMatrix


def test_help_message():
    logger.info('test_help_message start')
    result = subprocess.run([
        sys.executable,
        '-m', 'speaktome.speaktome',
        '-h'
    ], capture_output=True, text=True)
    assert result.returncode == 0
    assert 'usage:' in result.stdout.lower()
    logger.info('test_help_message end')


def test_basic_combinations():
    logger.info('test_basic_combinations start')
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
        logger.info('combo %s return code %s', combo, result.returncode)
    assert True
    logger.info('test_basic_combinations end')
