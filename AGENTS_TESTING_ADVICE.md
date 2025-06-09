# Testing Advice for Agents

Two directories relate to tests:

- `tests/` contains the official `pytest` suite. Run the full suite with `./.venv/bin/pytest -v` (Windows: `.venv\Scripts\pytest.exe -v`) or by executing `python testing/test_hub.py`.
- `testing/` hosts helper scripts for manual exploration. `test_hub.py` is a small wrapper around `pytest` that also writes `testing/stub_todo.txt` listing any tests marked with the `stub` marker. Pass `--skip-stubs` to the script if you want to temporarily ignore those placeholders.

Log files for each test run are stored under `testing/logs/pytest_<TIMESTAMP>.log`. The last ten logs are kept automatically.

Use the scripts in `testing/` when experimenting, but rely on the suite in `tests/` for formal verification.
