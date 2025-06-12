# Environment Setup Failures

**Date/Version:** 1749765345 v1
**Title:** Environment Setup Failures

## Overview
Attempted to follow repository instructions for environment setup using the provided scripts. Documenting each error encountered while following instructions or trying independent fixes.

## Prompts
- "perform an experience report on trying to set up the environment. for every failure, every error, log it, why you think it happened, and whether you were following instructions or acting independently"
- "always check the files in the repo ecosystem for your benefit... EXPLORE. LEARN."

## Steps Taken
1. Ran `bash setup_env.sh` in repo root.
2. Ran `bash setup_env.sh -headless -codebases=speaktome` to try non‑interactive mode.
3. Ran `bash setup_env_dev.sh -codebases=speaktome` as advised by docs.
4. Attempted manual run of `AGENTS/tools/dev_group_menu.py` with `PYTHONPATH` set.
5. Tried `pip install -e speaktome` to debug package install.

## Observed Behaviour
- Initial script runs produced `ModuleNotFoundError: No module named 'AGENTS'` and `ModuleNotFoundError: No module named 'tomli'` before completing. Excerpt:
  ```
  from AGENTS.tools.header_utils import ENV_SETUP_BOX
  ModuleNotFoundError: No module named 'AGENTS'
  ...
  ModuleNotFoundError: No module named 'tomli'
  ```
- Environment finished with message to activate `.venv` but no codebases installed. Snippet:
  ```
  [OK] Environment ready. Activate with 'source .venv/bin/activate'.
  * Torch = missing
  Selections recorded to /tmp/speaktome_active.json
  ```
  (File was not actually created.)
- Developer setup script showed the same import errors and then timed out at the menu.
- Manually invoking the dev menu with `PYTHONPATH=$PWD` avoided the import error but `pip` failed during editable install with `BackendUnavailable: Cannot import 'setuptools.build_meta'`.
- Direct `pip install -e speaktome` failed with missing dependency `tensors>=0.1.0`.

## Lessons Learned
- The setup scripts appear to run the dev menu without setting `PYTHONPATH`, causing import failures. Running the script as a module or exporting `PYTHONPATH` resolves that issue.
- `speaktome` depends on the local `tensors` package which must also be installed. The automated tool likely handles this but failed due to earlier import errors.
- Manual package installation results in backend errors with pip 25.1.1 even after updating setuptools.

## Next Steps
- Adjust scripts or environment to ensure `PYTHONPATH` includes the repo root before launching `dev_group_menu.py`.
- Investigate the `setuptools.build_meta` error when installing packages in editable mode.
- Confirm how the active selection file should be created so pytest can enable.
