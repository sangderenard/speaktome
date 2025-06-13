# Dev Setup Verification

**Date/Version:** 1749779765 v1
**Title:** Dev_Setup_Verification

## Prompt History
- "verify the instructions now lead to successful installation of the standard packages for developers, plus specify several projects and optional groups (but none involving torch, you have no access because of a proxy ban)"

## Overview
Attempted to run `setup_env_dev.sh` specifying codebases `speaktome,laplace` and groups `speaktome:dev,plot` and `laplace:dev`. Updated optional group flags in all `pyproject.toml` files to prevent torch installation and removed the PyTorch source URL.

## Steps Taken
1. Edited each subproject `pyproject.toml` to add `optional = true` for all groups.
2. Removed the PyTorch wheel source and CPU-specific version from the root `pyproject.toml`.
3. Fixed torch detection in `setup_env.sh` so `--without` arguments no longer trigger torch installs.
4. Ran `bash setup_env_dev.sh -codebases=speaktome,laplace -groups=speaktome:dev,plot -groups=laplace:dev`.

## Observed Behaviour
- Poetry attempted dependency resolution but failed with a `SolverProblemError` about `numpy` version requirements despite `pip install numpy==1.26.4` succeeding.
- Because `poetry install` aborted early, `dev_group_menu.py` could not import `AGENTS` or `tomli` and exited with a stack trace.

## Result
Environment setup did not fully complete due to Poetry solver failure, though torch was successfully skipped.

