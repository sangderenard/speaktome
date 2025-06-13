# Setup Environment Failures

**Date/Version:** 1749765262 v1
**Title:** Setup_Environment_Failures

## Overview
Attempted to follow repository instructions to set up the Python environment using provided scripts. Collected errors encountered during the process.

## Prompts
- "perform an experience report on trying to set up the environment. for every failure, every error, log it, why you think it happened, and whether you were following instructions or acting independently"
- Root `AGENTS.md` guidance to document environment setup attempts and run `AGENTS/validate_guestbook.py`.

## Steps Taken
1. Ran `bash setup_env_dev.sh --headless --codebases=speaktome,tools --notorch`.
2. Ran `bash setup_env_dev.sh --headless --codebases=speaktome,tools --notorch` again with output captured.
3. Ran `bash setup_env.sh -headless -codebases=speaktome,tools -notorch`.
4. Tried running `python testing/test_hub.py` with the virtual environment activated and `PYTHONPATH` set.

## Observed Behaviour
- `setup_env_dev.sh` installed pip and wheel but failed to import `AGENTS` during its dev menu phase, producing `ModuleNotFoundError: No module named 'AGENTS'` and `ModuleNotFoundError: No module named 'tomli'`.
- The script printed a developer menu and awaited input despite the `--headless` flag. It repeated "Select option" until interrupted.
- `setup_env.sh` launched an interactive selection tool and crashed with `NameError: name 'getch_timeout' is not defined` before completing.
- `testing/test_hub.py` exited early with a message that the environment was not configured.

## Lessons Learned
Even with headless flags, the setup scripts attempted interactive steps and failed due to missing packages or functions. The environment remained partially configured and tests refused to run.

## Next Steps
Investigate why `AGENTS` and `tomli` were missing during setup. Determine whether the editable install step is running correctly and why the `getch_timeout` function was undefined. Consider updating the setup scripts or documentation to clarify headless usage.
