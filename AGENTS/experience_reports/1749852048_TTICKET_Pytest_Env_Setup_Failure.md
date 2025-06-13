# Trouble Ticket Report

**Date:** 1749852048
**Title:** Pytest environment setup failure

## Environment
- **OS:** Linux 6.12.13
- **Python:** Python 3.12.10
- **Setup Command:** `python testing/test_hub.py`

## Steps to Reproduce
1. Executed the test wrapper:
   ```bash
   python testing/test_hub.py 2>&1 | tee testing/logs/pytest_1749852022.log
   ```
2. Attempted to run `bash setup_env.sh --from-dev -codebases speaktome` when tests failed.

## Logs and Output
Key excerpts from `testing/logs/pytest_1749852022.log`:
```
[INFO] poetry-core missing; installing to enable editable builds
Warning: command 'env POETRY_VIRTUALENVS_IN_PROJECT=true poetry install --sync --no-interaction --without cpu-torch --without gpu-torch' failed with status 1
...
Skipped: Environment not initialized. See ENV_SETUP_OPTIONS.md
Automated setup failed. Skipping all tests.
```
The setup script also reported a failed attempt to install dependencies due to a missing build backend (`poetry.core.masonry.api`).

## Attempted Fixes
- Ran the provided `setup_env.sh` script with `--from-dev` to install the `speaktome` codebase, but the install step failed.
- No further action taken.

## Current Status
Tests remain skipped because the environment could not be initialized. Network access may be required for dependency installation.

## Prompt History
- "run pytest, pipe to file, wrap with trouble ticket template structure and place in the experience reports with the correct filename"
- "always check the files in the repo ecosystem for your benefit. the project has a particular ethos and theory..."
