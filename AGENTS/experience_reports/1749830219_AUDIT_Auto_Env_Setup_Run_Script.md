# Auto Env Setup Audit

**Date:** 1749830219
**Title:** Auto environment setup when executing tests

## Scope
Evaluate the automatic environment initialization triggered by the test suite when no virtual environment exists.

## Methodology
- Attempted to run a single pytest via `pytest -k test_check_run_setup_missing -vv` without any prior environment setup.
- Observed the output for automatic installation steps and resulting behaviour.

## Detailed Observations
- `tests/conftest.py` uses `run_setup_script` to invoke `setup_env.sh` if `.venv` is missing.
- Running the command initiated Poetry dependency installation but failed while building the `tensors` package, producing a `ModuleOrPackageNotFoundError`.
- The script then called `dev_group_menu.py` with an incorrect `-groups 0` argument and exited with an error message.
- After the failed attempt, pytest displayed `Skipped: Environment not initialized. See ENV_SETUP_OPTIONS.md` and aborted execution.

## Analysis
The automatic setup procedure successfully launched but could not complete due to build failures in the `tensors` package and argument issues when launching the development menu. Because the environment did not finish initializing, the tests skipped entirely.

## Recommendations
- Investigate the editable install failure for `tensors` during Poetry setup.
- Review argument handling in `setup_env.sh` when no groups are specified to avoid passing an invalid `-groups 0` option.
- Ensure environment errors surface clearly to help users troubleshoot.

## Prompt History
- "audit the experience of trying to setup the environment automatically by trying to run any project script and depending on the new auto environment setup"
