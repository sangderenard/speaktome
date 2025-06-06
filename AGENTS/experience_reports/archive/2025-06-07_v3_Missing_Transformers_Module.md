# User Experience Report

**Date/Version:** 2025-06-07 v3
**Title:** Missing Transformers Module After Running `run` Without Setup

## Overview
Attempted to launch the demo straight from the command prompt using the `run` wrapper. The script failed with `ModuleNotFoundError: No module named 'transformers.GPT2LMHeadModel'` before any output appeared.

## Steps Taken
1. Navigated to the repository directory on Windows.
2. Executed `run` from `cmd.exe`.
3. Observed the stack trace complaining that `transformers` was missing.
4. Ran `todo/validate_guestbook.py` in WSL to check the guest book rules.

## Observed Behaviour
- The application aborted immediately during import because dependencies were not installed.
- Validation script listed all reports without renaming any files.

## Lessons Learned
- The README explains that you must create the virtual environment first with `setup_env.ps1` (or `setup_env.sh` on Linux) which installs packages like `torch` and `transformers`.
- Forgetting this step causes the import failure shown above.

## Next Steps
- Run `setup_env.ps1` followed by `fetch_models.ps1` to prepare the environment.
- Retry the demo with `run.cmd -s "Hello" -m 10 -c -a 5 --final_viz` once dependencies and models are in place.
