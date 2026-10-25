"""Run selected tests with HEAD module text, without replacing working files."""
import argparse
import importlib
from pathlib import Path
import subprocess
import sys

parser = argparse.ArgumentParser(description=__doc__)
parser.add_argument('--repo', required=True, type=Path)
parser.add_argument('--module', action='append', required=True)
parser.add_argument('tests', nargs='+')
args = parser.parse_args()
sys.path.insert(0, str(args.repo.resolve()))
for name in args.module:
    module = importlib.import_module(name)
    path = name.replace('.', '/') + '.py'
    source = subprocess.check_output(['git', 'show', 'HEAD:' + path], cwd=args.repo).decode('utf-8')
    exec(compile(source, str(args.repo.resolve() / path), 'exec'), module.__dict__)
import pytest
raise SystemExit(pytest.main(['-q', *args.tests]))
