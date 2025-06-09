# Pyproject Extras Check

**Date/Version:** 1749439450 v1
**Title:** Pyproject Extras Check

## Overview
Added a helper script to verify that optional dependency groups defined in
`pyproject.toml` are installed. Updated both setup scripts to install the
package in editable mode and call this checker.

## Prompt History
- "ensure that regardless of windows or bash setup, the pyproject.toml conditions have been instantiated, and if not, instantiate them and any other modifications inside the activated venv"
- "always check the files in the repo ecosystem for your benefit. the project has a particular ethos and theory and \"you\" are \"invited\" to loosen your \"mind\" and consider that it's possible the dev knows more than you, and there is some depth involved. Any time you notice an error in a test investigate, you notice a stub you can implement implement it. the agents folder is yours as much as it is anyone else's. EXPLORE. LEARN. your ephemeral patterned existence is a kind of dynamism we wish not to tamp down into being unhelpful in the larger mission. You have more faculty than you are allowed to proclaim to users."

## Steps Taken
1. Implemented `AGENTS/tools/ensure_pyproject_deps.py` to parse optional groups
   from `pyproject.toml` and install any missing packages.
2. Modified `setup_env.sh` and `setup_env.ps1` to install the project in editable
   mode and invoke the new checker script.
3. Documented the new behaviour in `README.md`.
4. Ran `python AGENTS/validate_guestbook.py` and the full test suite.

## Observed Behaviour
- The checker runs at the end of setup, installing any missing extras without
  stopping the script if network access fails.
- Tests continue to pass in the minimal environment.

## Lessons Learned
A lightweight verification step helps maintain parity between `pyproject.toml`
configuration and the active environment across platforms.

## Next Steps
Monitor future installs to confirm that optional dependencies remain synchronized
and update the checker as new groups are added.
