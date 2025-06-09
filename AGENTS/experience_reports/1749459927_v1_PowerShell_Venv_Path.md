# PowerShell Venv Path Fix

**Date/Version:** 1749459927 v1

## Overview
Running `setup_env_dev.ps1 -extras` on Linux with PowerShell failed because `.venv\Scripts\python.exe` did not exist. The script assumed a Windows-style virtual environment layout.

## Prompt History
- "diagnose why the setup_env_dev extended installations are failing saying .venv\etc etc is not found, correct please."

## Steps Taken
1. Investigated both setup scripts.
2. Noticed PowerShell scripts used Windows-only paths.
3. Added cross-platform logic to detect `$IsWindows` and set the correct venv paths.

## Observed Behaviour
Scripts now locate `.venv/bin/python` on Unix-like systems and `.venv\Scripts\python.exe` on Windows.

## Lessons Learned
Cross-platform PowerShell scripts should use `$IsWindows` to handle differing virtual environment layouts.

## Next Steps
Review other scripts for similar assumptions.
