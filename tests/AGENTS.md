# Testing Guidelines

This directory holds the automated test suite. Use `./.venv/bin/pytest -v`
(or `.venv\Scripts\pytest.exe -v` on Windows) to run all tests.

## Stub Tests
Some tests are marked with the `stub` marker when a realistic implementation
is difficult or requires optional dependencies. These placeholders ensure that
every class has a corresponding test entry.

Run `python testing/test_hub.py` to execute the suite and generate
`testing/stub_todo.txt` which lists all stubbed tests pending real coverage.

When expanding a stub into a full test, remove the `@pytest.mark.stub` decorator
and update any relevant documentation.

## Environment Setup Caution

Tests should never attempt to import packages that the setup scripts are
responsible for installing. If a dependency is missing, the header guard will
print the contents of `ENV_SETUP_BOX.md` and exit. Do not try to import
`AGENTS.tools.header_utils` or any other helper inside the setup routines as a
way to bootstrap installation&mdash;that defeats the purpose of the environment
check and often fails on clean systems.
