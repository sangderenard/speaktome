# Pytest Auto Setup Attempt

## Overview
Implemented an automatic environment setup helper for pytest. When tests start,
`conftest.py` now tries to run `setup_env.sh` or `setup_env.ps1` through a new
utility `AGENTS.tools.auto_env_setup.run_setup_script`. If that fails, the
helper offers a timed y/N prompt via `ask_no_venv` to install packages without
a virtual environment. Only after these attempts do we skip the session.

## Prompts
- "In previous turns, the user had the following interaction..." (see prompt)
- "Alright, can you make sure everything is nice and smooth and is it possible for use to have pytest wrap every test in an attempt running the setup bash or powershell..."

## Steps Taken
1. Created `AGENTS/tools/auto_env_setup.py` with `run_setup_script` and
   `ask_no_venv` helpers.
2. Updated `tests/conftest.py` to call these helpers and only skip when setup
   remains impossible.
3. Added a new `_auto_setup_and_check` function used during `pytest_configure`.
4. Modified `pytest_collection_modifyitems` to respect the global skip flag.
5. Ran the guestbook validator and test hub.

## Observed Behaviour
- The validator succeeded with no errors.
- The test hub attempted to run but skipped the suite because the environment
  could not be configured automatically.

## Lessons Learned
Automating environment setup from within pytest cannot activate the virtual
environment for the current process, but it can provide a graceful fallback.
All tests will now be skipped rather than abruptly exiting when setup is
missing.

## Next Steps
Investigate ways to propagate environment changes back to pytest, possibly by
re-executing under the new interpreter when setup succeeds.
