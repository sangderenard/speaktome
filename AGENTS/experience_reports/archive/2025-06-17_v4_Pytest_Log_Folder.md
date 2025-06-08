# Pytest Log Folder

**Date/Version:** 2025-06-17 v4
**Title:** Pytest Log Folder

## Overview
Implemented automatic logging for test runs so each pytest session writes to a unique file under `testing/logs`.

## Prompt History
```
Can you prepare a log folder and make sure  pytest testing auto-logs to unique filenames per session
```

## Steps Taken
1. Created `testing/logs/` directory.
2. Updated `tests/conftest.py` to add a `pytest_configure` hook that sets up a timestamped log file.
3. Documented the log location in `README.md`.
4. Ran `pytest -q` to verify log generation.

## Observed Behaviour
- A new log file appears in `testing/logs` after running tests.
- All existing tests continue to pass.

## Lessons Learned
Integrating logging via `pytest_configure` keeps test output organized without modifying individual test scripts.

## Next Steps
- Periodically clean old logs to manage disk usage.
