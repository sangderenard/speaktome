"""Run pytest with one module's HEAD source in memory, preserving the checkout."""
import argparse
import importlib
from pathlib import Path
import subprocess
import sys


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--repo", type=Path, required=True)
    parser.add_argument("--module", required=True)
    parser.add_argument("--failures-from", type=Path)
    args, pytest_args = parser.parse_known_args()
    sys.path.insert(0, str(args.repo.resolve()))
    module = importlib.import_module(args.module)
    relative = Path(module.__file__).resolve().relative_to(args.repo.resolve()).as_posix()
    source = subprocess.run(["git", "show", "HEAD:" + relative], cwd=args.repo,
                            capture_output=True, text=True, check=True).stdout
    old_functions = {name: value for name, value in vars(module).items()
                     if callable(value) and getattr(value, "__module__", None) == args.module}
    exec(compile(source, "HEAD:" + relative, "exec"), vars(module))
    # Imports may already have captured function aliases while loading the
    # target module. Redirect only exact aliases of its own previous objects.
    for loaded in tuple(sys.modules.values()):
        if loaded is None or loaded is module:
            continue
        for name, previous in old_functions.items():
            if vars(loaded).get(name) is previous:
                vars(loaded)[name] = vars(module)[name]
    if args.failures_from:
        pytest_args += [line.split()[1] for line in args.failures_from.read_text(
            encoding="utf-8", errors="replace").splitlines() if line.startswith("FAILED tests/")]
    import pytest
    return pytest.main(pytest_args)


if __name__ == "__main__":
    raise SystemExit(main())
