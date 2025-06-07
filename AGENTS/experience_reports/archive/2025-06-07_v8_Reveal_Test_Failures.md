# Test Failure Reveal Report

**Date/Version:** 2025-06-07 v8
**Title:** Reveal hidden test failures

## Overview
Investigate whether the repository's tests hide failures by skipping
important assertions. Modified tests to surface real errors.

## Prompts
User request:
"evaluate test results, see if the ai agent has made it look like more tests are passing than are by avoiding testing them. I need to know not if our stubs pass as stubs, I need to know exactly what error prevents each item from working. Task success is increasing the number of failures in test, not decreasing it. I'm quite serious I want this to be a tunnel of development focus to burrow through the problem. That is undermined by covering things up. If anything is deferred or unavailable it has to be clearly announced."

## Steps Taken
1. Ran `python testing/test_hub.py` before modifications and observed all tests reported as passing via log files.
2. Updated `tests/test_all_classes.py` to fail when imports or instantiation fail.
3. Updated `tests/test_cli.py` to fail when CLI commands exit with a non-zero status.
4. Re-ran `python testing/test_hub.py` and reviewed `testing/logs/pytest_*.log`.

## Observed Behaviour
The updated tests now fail for missing `torch` imports and other unimplemented
classes. `test_cli.py::test_basic_combinations` reports a non-zero return code.
Overall 16 tests fail where previously 35 passed.

## Lessons Learned
The prior test suite logged errors but did not fail. By converting these logs to
assertions the missing dependencies and incomplete implementations are
highlighted, providing clear targets for development.

## Next Steps
Implement missing classes or provide mocks to satisfy the failing imports and
review CLI exit codes.

