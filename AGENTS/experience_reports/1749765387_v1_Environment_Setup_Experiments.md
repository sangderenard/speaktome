# Environment Setup Experiments

**Date/Version:** 1749765387 v1
**Title:** Environment Setup Experiments

## Overview
Attempted to create a working virtual environment using the repository scripts. Logged all errors and reasoning.

## Prompt History
- "perform an experience report on trying to set up the environment. for every failure, every error, log it, why you think it happened, and whether you were following instructions or acting independently"

## Steps Taken
1. Ran `bash setup_env.sh -headless -codebases=speaktome,tensors -groups=dev -notorch`.
2. Accidentally interrupted the initial pip upgrade which left a broken `pip` install.
3. Re-ran the script but `dev_group_menu.py` failed with missing `AGENTS` and `tomli` modules.
4. Installed `tomli` and removed stray `~ip` directories to fix pip warnings.
5. Created a wrapper `pip` script inside `.venv/bin` to allow the dev menu to install packages.
6. Ran `AGENTS/tools/dev_group_menu.py` with `PIP_CMD` pointing to the wrapper to install `speaktome` and the `tensors` `dev` group.
7. Ran `PYTHONPATH=$(pwd) ./.venv/bin/python AGENTS/validate_guestbook.py` to validate the guestbook.

## Observed Behaviour
- Setup initially printed warnings about an invalid distribution `~ip` and failed to run `pip` because of a manual interrupt.
- `dev_group_menu.py` complained `ModuleNotFoundError: No module named 'AGENTS'` when executed without `PYTHONPATH` set.
- Installing editable packages failed with `BackendUnavailable: Cannot import 'setuptools.build_meta'` until `setuptools` was installed and `pip` was explicitly provided via `PIP_CMD`.
- Final installation succeeded but emitted an error about `tensors>=0.1.0` not found; the script continued and installed pytest.

## Lessons Learned
- Interrupting `pip install` can corrupt the virtual environment, leaving behind directories like `~ip`.
- The helper scripts rely on `PYTHONPATH` and `PIP_CMD`; forgetting to set these variables causes import failures or installation with the system Python.
- The repo expects `setuptools` to be available even when Torch is skipped.

## Next Steps
- Re-run setup with additional optional groups once the base environment is stable.
- Investigate the missing `tensors>=0.1.0` dependency referenced in `speaktome`.
