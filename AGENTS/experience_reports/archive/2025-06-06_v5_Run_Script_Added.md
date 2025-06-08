# Unified Run Script Added

**Date/Version:** 2025-06-06 v5
**Title:** Added cross-platform run wrappers

## Overview
This brief report documents adding `run.sh` and `run.ps1` to ensure the project
is executed with the virtual environment's Python on all platforms.

## Steps Taken
1. Created `run.sh` and `run.ps1` that call `.venv`'s Python directly.
2. Updated the README with usage examples referencing these scripts.

## Observed Behaviour
Both scripts simply forward arguments to `python -m speaktome.speaktome` using the
virtual environment, avoiding the previous package import error.

## Lessons Learned
Providing wrapper scripts simplifies instructions and guarantees we run the code
with the intended interpreter.

## Next Steps
- Encourage new users to always invoke the program through these scripts.
