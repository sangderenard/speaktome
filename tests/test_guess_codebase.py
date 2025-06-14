import pathlib
from AGENTS.tools.dynamic_header_recognition import guess_codebase


def test_guess_codebase_speaktome():
    path = pathlib.Path('speaktome/__init__.py').resolve()
    assert guess_codebase(path) == 'speaktome'


def test_guess_codebase_no_map(tmp_path):
    path = pathlib.Path('speaktome/__init__.py').resolve()
    missing_map = tmp_path / 'missing.json'
    assert guess_codebase(path, map_file=missing_map) == 'speaktome'

