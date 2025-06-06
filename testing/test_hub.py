"""Run the test suite and collect stubbed tests."""
from __future__ import annotations

import pytest
from pathlib import Path


class StubCollector:
    def __init__(self) -> None:
        self.stub_tests: list[str] = []

    def pytest_collection_modifyitems(self, session, config, items):  # type: ignore[override]
        for item in items:
            if 'stub' in item.keywords:
                self.stub_tests.append(item.nodeid)


def main() -> int:
    collector = StubCollector()
    ret = pytest.main(['-v', 'tests'], plugins=[collector])
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
