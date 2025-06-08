# User Experience Report

**Date/Version:** 2025-06-07 v6
**Title:** Venv Sanity Check for Windows Run Scripts

## Overview
Implemented a check in `run.ps1` and `run.cmd` to verify they invoke the Python interpreter inside `.venv`. This helps catch cases where the script is launched from the wrong directory or the environment is missing.

## Steps Taken
1. Updated both scripts to resolve the virtual environment path relative to the script location.
2. Each script now calls Python to print `sys.executable` and compares it to the expected `.venv` path, emitting a warning if they differ.
3. Ran `todo/validate_guestbook.py` to confirm naming.

## Observed Behaviour
Running either script now prints a warning when the interpreter path doesn't match `.venv\Scripts\python.exe`.

## Lessons Learned
Simple sanity checks prevent confusion when multiple Python installations exist.

## Next Steps
None.
