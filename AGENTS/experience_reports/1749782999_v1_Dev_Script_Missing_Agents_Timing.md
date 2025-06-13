# Dev Script Missing AGENTS Timing

**Date/Version:** 1749782999 v1
**Title:** Dev Script Missing AGENTS Timing

## Overview
Investigated exactly where the `ModuleNotFoundError: No module named 'AGENTS'` occurs during environment setup.

## Prompt History
- "maybe it's in the dev script, but I need you to time precisely the moment - the very instruction - which stimulates the error of a missing module called agents"

## Steps Taken
1. Ran `bash -x setup_env.sh` to trace script execution.
2. Observed the command executing `AGENTS/tools/dev_group_menu.py` via the venv Python.

## Observed Behaviour
- Error triggered when the script executed:
  `./.venv/bin/python AGENTS/tools/dev_group_menu.py --install --record /tmp/speaktome_active.json`
- The stack trace shows the import statement `from AGENTS.tools.header_utils import ENV_SETUP_BOX` failing, producing `ModuleNotFoundError: No module named 'AGENTS'`.

## Lessons Learned
- `dev_group_menu.py` requires the `AGENTS` package and `tomli` before the environment finishes installing dependencies.
- Running the setup script with `bash -x` reveals the failing command and line numbers, confirming the error originates from the import statement at line 26 of `dev_group_menu.py`.

## Next Steps
- Ensure `AGENTS` and `tomli` are installed before invoking `dev_group_menu.py`, or adjust the script to handle missing dependencies gracefully.
