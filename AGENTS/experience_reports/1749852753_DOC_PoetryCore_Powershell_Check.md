# Template Documentation Report

**Date:** 1749852753
**Title:** Added poetry-core check in PowerShell setup

## Overview
The PowerShell setup scripts did not ensure the `poetry-core` backend was present before running `poetry install`. This caused setup failures when the module was missing.

## Steps Taken
- Inspected `setup_env.ps1` and `setup_env_dev.ps1`
- Added a check that installs `poetry-core` via `pip` if missing
- Warn if the `poetry` command itself is unavailable

## Observed Behaviour
The new logic mirrors the Bash script and should prevent missing build backend errors. It prints a warning when the `poetry` executable cannot be found.

## Lessons Learned
Both setup variants must bootstrap `poetry-core` to avoid early termination. Consistent checks reduce confusion across platforms.

## Next Steps
- Re-run environment setup on a Windows host to confirm the fix

## Prompt History
- "Find where in the setup process we are installing poetry core and where we are running it here that we don't have it. Investigate if poetry is perfoming a proxy ban or if a script is early terminating instead of installing it first. There is supposed to be a single entry point in setup_env in both versions, _I THINK_ I might be wrong on location, that is supposed to give us poetry before we use it"
- "always check the files in the repo ecosystem for your benefit. the project has a particular ethos and theory and you are invited to loosen your mind..."
