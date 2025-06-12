# Environment Setup Testing

**Date/Version:** 1749761998 v1
**Title:** Env_Setup_Testing

## Overview
Followed repository instructions to set up the virtual environment before running tests.

## Prompts
- "Why didn't you follow the repo instructions for prepping the environment to run anything"
- Instructions in `AGENTS.md` and `ENV_SETUP_OPTIONS.md` about using `setup_env_dev.sh` and validating guestbook entries.

## Steps Taken
1. Executed `bash setup_env_dev.sh --headless --codebases=speaktome,tools --notorch`.
2. Activated the virtual environment and installed dev requirements.
3. Set `PYTHONPATH` to the repository root and ran `python testing/test_hub.py` using the venv interpreter.
4. Created `.venv/pytest_enabled` and verified `SPEAKTOME_ACTIVE_FILE` contents.

## Observed Behaviour
- Environment setup attempted to install packages but some downloads were blocked.
- Test run failed with an internal error due to missing optional dependencies.

## Lessons Learned
Setting up the environment with the provided scripts is necessary for tests to run. Missing optional packages can still cause failures if tests require them.

## Next Steps
Investigate how to mock or skip tests requiring unavailable packages, and explore additional codebases.
