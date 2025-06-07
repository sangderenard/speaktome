# Pytest Tee and Stub Skip Option

**Date/Version:** 2025-06-17 v10
**Title:** Pytest log tee and optional stub skipping

## Overview
Implemented tee logging so stdout still prints to the console while being captured in the log file. Added a `--skip-stubs` option to skip tests marked with `@pytest.mark.stub`. The flag is off by default so unfinished tests remain visible unless explicitly suppressed.

## Prompts
"We capture pytest stdout and it's confusing agents, can we set it up to tee the output to log not steal it?  also, we have this test skipping thing? I want it present but disabled. I want it to have to be a special argument to enable skipping listed stubs, we're trying to create an on-rails development area with pytest being a procedural pressure test. We can't go around not knowing right exactly where it isn't finished. The press is to get 100% of present code working in the next 3 days, preferrably yesterday."

## Steps Taken
1. Modified `tests/conftest.py` to replace stdout with a tee stream writing to both the terminal and the log file.
2. Added `pytest_addoption` hook for `--skip-stubs` and implemented `pytest_collection_modifyitems` to honor it.
3. Updated `testing/test_hub.py` to accept the new argument and pass it through to pytest.
4. Ran `python AGENTS/validate_guestbook.py` to ensure report naming and archive rules.
5. Executed `pytest -v` to verify the logging and option behaviour.

## Observed Behaviour
- Test output now appears in the console and still writes to `testing/logs/pytest_<timestamp>.log`.
- When invoked with `--skip-stubs`, tests marked `stub` are skipped; otherwise they run normally.

## Lessons Learned
Making the logging less intrusive helps both human and automated agents follow failures directly in the console while still archiving the session logs. Optional skipping keeps the development pressure intact by default.

## Next Steps
Continue reviewing failing tests and incrementally replace stubs with real implementations.
