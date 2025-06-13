# Update Pyproject Imports Script

**Date/Version:** 1749793921 v1
**Title:** Added tool to auto-insert imports into pyproject

## Overview
Implemented `add_imports_to_pyproject.py` which parses a Python file for imported modules and ensures they are present in the project's `pyproject.toml`. New dependencies default to an `unsorted` group while Torch-related packages are placed in the optional `cpu-torch` group.

## Prompts
- "create a script capable of processing a python script for all imports, automatically putting them in a group in the pyproject.toml if they are not already there..."
- Repository guidelines in `AGENTS.md` and `AGENTS/CODING_STANDARDS.md`.

## Steps Taken
1. Wrote new tool under `AGENTS/tools` following header format.
2. Updated `validate_headers.py` to import `sys` correctly.
3. Ran unit tests.

## Observed Behaviour
Script successfully updates dependency groups when tested locally. All tests pass.

## Lessons Learned
The pre-existing header validation tool lacked a `sys` import which caused execution errors. Always verify tool scripts with provided validators.

## Next Steps
Consider enhancing detection of standard library modules and expanding Torch package heuristics.
