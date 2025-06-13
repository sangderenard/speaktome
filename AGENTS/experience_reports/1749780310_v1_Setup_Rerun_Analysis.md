# Experience Report: Setup Rerun Analysis

**Date/Version:** 1749780310 v1
**Title:** Investigating numpy version conflict

## Overview
Re-ran `setup_env_dev.sh` with torch installation to capture full error output and verify Python requirements.

## Prompts
- Root `AGENTS.md` instructions to run setup scripts and record findings.
- User query: "we put in the python requirements didnt we? do we actually need those requirements or can we adjust versions to get the same functionality and performance without having this conflict?"

## Steps Taken
1. Executed `bash setup_env_dev.sh -torch` and saved output to `setup.log`.
2. Ran `python testing/test_hub.py` to check whether tests run without the environment.
3. Inspected `pyproject.toml` files for Python and numpy version pins.

## Observed Behaviour
- Poetry failed to resolve dependencies because `speaktome` requires `numpy >=1.26`, which itself demands Python >=3.9. The root project only specifies `python >=3.8`.
- `AGENTS/tools/dev_group_menu.py` raised `ModuleNotFoundError` for `AGENTS` and `tomli`.
- The test hub script failed with `ModuleNotFoundError: No module named 'AGENTS'`.

## Lessons Learned
The numpy requirement originates from `speaktome/pyproject.toml`. Installing torch triggers this dependency and exposes the Python version conflict. Without adjusting versions or using Python >=3.9, `poetry install` cannot proceed.

## Next Steps
- Consider lowering the numpy pin to a version compatible with Python 3.8 if older interpreters must be supported.
- Alternatively, raise the project-wide Python requirement to >=3.9 in `pyproject.toml` so `numpy >=1.26` resolves cleanly.
- Revisit `dev_group_menu.py` import paths after the environment installs successfully.
