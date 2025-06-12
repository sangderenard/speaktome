# Investigating Tomli Usage and Dev Group

**Date/Version:** 1749765746 v1
**Title:** Tomli Usage and Dev Group Review

## Overview
Explored the repository to determine where `tomli` is used and confirm if the `dev` optional dependency group exists in the codebase map.

## Prompt History
- "investigate who is using tomli and if tomli is already mentioned in the agents/tools codebase toml. I have noticed that the tools installation seems to be failing, though in your case it looks like you didn't even ask for tools or specify you are a dev - can you confirm the dev group exists in the codebase map and what project it is a part of"

## Steps Taken
1. Searched the repository for `tomli` and `tomllib` usage.
2. Inspected `AGENTS/tools/pyproject.toml` to verify dependencies.
3. Checked `AGENTS/tools` scripts for `tomli` imports.
4. Reviewed `AGENTS/codebase_map.json` to find any `dev` groups.
5. Ran `PYTHONPATH=$(pwd) python3 AGENTS/validate_guestbook.py` to validate the report.

## Observed Behaviour
- `tomli` is used in `AGENTS/tools/update_codebase_map.py`, `dev_group_menu.py`, and `ensure_pyproject_deps.py` as a fallback for Python versions older than 3.11.
- `AGENTS/tools/pyproject.toml` already lists `tomli; python_version<'3.11'` in the dependencies.
- `AGENTS/codebase_map.json` includes a `dev` group only for the `tensors` codebase.
- The validation script confirmed the guestbook filenames conform to the required pattern.

## Lessons Learned
- The failing tools installation seen in earlier reports was due to missing dependencies (`AGENTS` module and `tomli`) when `dev_group_menu.py` was run outside the repository's Python path. Setting `PYTHONPATH` and ensuring the virtual environment has the `tomli` package resolves this.
- The `dev` optional dependency group is defined for the `tensors` project, but not for others in `codebase_map.json`.

## Next Steps
- Document the need to activate the virtual environment and set `PYTHONPATH` when running `AGENTS/tools` utilities.
