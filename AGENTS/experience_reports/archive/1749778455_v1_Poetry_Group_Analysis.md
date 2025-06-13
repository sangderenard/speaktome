# Poetry Group Analysis

**Date/Version:** 1749778455 v1
**Title:** Poetry_Group_Analysis

## Prompt History
- "Find out why poetry tried to install torch when you said without, what does the without argument do, are the names right for the configuration? is someone trying to install torch under circumstances where it hasn't been explicitly requested? why didn't poetry install agents/tools? why is dev group looking for AGENTS not AGENTS/tools? why isn't the no-venv flag working? does poetry fail entirely if torch fails to download? if so can you isolate it? we honestly don't give a shit about torch if we can't get it."

## Overview
Investigated why `poetry install` attempted to fetch PyTorch wheels despite using the `--without` option. Examined configuration files and logs to see how dependency groups are handled during setup.

## Steps Taken
1. Ran `poetry install --dry-run --without cpu-torch --without gpu-torch -vv` and observed network requests to `download.pytorch.org`.
2. Inspected `pyproject.toml` to verify the dependency group names and the absence of a top-level torch dependency.
3. Parsed the file using `tomllib` to confirm group metadata.
4. Noted that `poetry install` still resolved `torch` because the groups lack the `optional = true` flag, making them mandatory.
5. Checked `AGENTS/codebase_map.json` and `dev_group_menu.py` to understand how `AGENTS/tools` is installed after the base environment.
6. Verified that misspelling the `--no-venv` flag still results in `.venv` creation.
7. Observed that poetry aborts entirely when torch wheels cannot be downloaded, so other packages (like `AGENTS/tools`) never get installed.

## Observed Behaviour
- The solver reported `speaktome-hub` depends on `torch (2.3.1+cpu)` even with `--without` arguments because the group definition isn't optional.
- `poetry install` fails once it cannot reach the PyTorch index, halting further installation of path dependencies.
- `dev_group_menu.py` searches for the `AGENTS` package because the `AGENTS/tools` project exposes modules in that namespace.
- Misspelling the `--no-venv` flag does nothing; the correct form is `--no-venv`.

## Lessons Learned
- `--without` only skips dependency groups marked `optional = true` in `pyproject.toml`.
- Torch packages are treated as mandatory due to missing `optional = true`, so network errors stop the install.
- `AGENTS/tools` isn't installed when `poetry install` fails early.

## Next Steps
- Update the torch groups in `pyproject.toml` to include `optional = true` so they are skipped by default.
- Ensure documentation emphasizes the correct flag `-no-venv`.
- Consider removing the PyTorch source when offline to prevent failures.
