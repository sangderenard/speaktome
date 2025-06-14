#!/usr/bin/env python3
"""Run the test suite and collect stubbed tests."""
from __future__ import annotations

try:
    import argparse
    import pytest
    from pathlib import Path
except Exception:
    import os
    import sys
    from pathlib import Path

    def _find_repo_root(start: Path) -> Path:
        current = start.resolve()
        required = {
            "speaktome",
            "laplace",
            "tensorprinting",
            "timesync",
            "AGENTS",
            "fontmapper",
            "tensors",
            "testenv",
        }
        for parent in [current, *current.parents]:
            if all((parent / name).exists() for name in required):
                return parent
        return current

    if "ENV_SETUP_BOX" not in os.environ:
        root = _find_repo_root(Path(__file__))
        box = root / "ENV_SETUP_BOX.md"
        try:
            os.environ["ENV_SETUP_BOX"] = f"\n{box.read_text()}\n"
        except Exception:
            os.environ["ENV_SETUP_BOX"] = "environment not initialized"
        print(os.environ["ENV_SETUP_BOX"])
        sys.exit(1)
    import subprocess
    try:
        root = _find_repo_root(Path(__file__))
        subprocess.run(
            [sys.executable, "-m", "AGENTS.tools.auto_env_setup", str(root)],
            check=False,
        )
    except Exception:
        pass
    try:
        ENV_SETUP_BOX = os.environ["ENV_SETUP_BOX"]
    except KeyError as exc:
        raise RuntimeError("environment not initialized") from exc
    print(f"[HEADER] import failure in {__file__}")
    print(ENV_SETUP_BOX)
    sys.exit(1)
# --- END HEADER ---


class StubCollector:
    def __init__(self) -> None:
        self.stub_tests: list[str] = []

    def pytest_collection_modifyitems(self, session, config, items):  # type: ignore[override]
        for item in items:
            if 'stub' in item.keywords:
                self.stub_tests.append(item.nodeid)


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description="Run pytest and collect stub tests")
    parser.add_argument(
        "--skip-stubs",
        action="store_true",
        help="Skip tests marked with @pytest.mark.stub",
    )
    args = parser.parse_args(argv)

    collector = StubCollector()
    pytest_args = ['-v', 'tests']
    if args.skip_stubs:
        pytest_args.append('--skip-stubs')

    ret = pytest.main(pytest_args, plugins=[collector])
    todo = Path('testing/stub_todo.txt')
    with todo.open('w') as f:
        if collector.stub_tests:
            f.write('Stub tests remaining:\n')
            for nodeid in collector.stub_tests:
                f.write(f'- {nodeid}\n')
        else:
            f.write('No stub tests remain.\n')
    print(f'Stub list written to {todo}')
    return ret


if __name__ == '__main__':
    raise SystemExit(main())
