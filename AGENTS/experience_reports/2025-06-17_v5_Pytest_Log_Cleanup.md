# Pytest Log Cleanup

**Date/Version:** 2025-06-17 v5
**Title:** Pytest Log Cleanup

## Overview
Added automatic pruning for old pytest logs so the directory doesn't grow indefinitely.

## Prompt History
```
did you do anything like gitignore or auto cleaning of the logs? I was thinking they would accumulate in the repo with unique filenames until manually cleared but that's a little sloppy isn't it
```

## Steps Taken
1. Examined existing log setup and `.gitignore` rules.
2. Added log cleanup in `tests/conftest.py` keeping only ten recent logs.
3. Documented pruning behaviour in `README.md`.
4. Ran `pytest -q` to confirm logs rotate correctly.

## Observed Behaviour
- Old logs are deleted when more than ten exist.
- New log file still records the session without errors.

## Lessons Learned
Automated log management prevents uncontrolled growth of the `testing/logs` directory.

## Next Steps
- Adjust the retention policy as the project evolves.
