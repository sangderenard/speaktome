# Bug Hunting dev_group_menu Fix

## Overview
Attempted bug hunting job by running the job dispenser. Setup environment and ran tests. Identified failure when installing codebase `tools` via `dev_group_menu` due to incorrect path handling. Added fix to use registry paths and avoid build isolation.

## Prompts
- Root instructions from `AGENTS.md` about using the job dispenser and recording experience reports.
- Job description `bug_hunting_job.md` instructing to run tests and fix warnings.
- User message: "log in read instructions obtain a job and fulfill it".

## Steps Taken
1. Ran `python -m AGENTS.tools.dispense_job` which returned `bug_hunting_job.md`.
2. Executed `setup_env_dev.sh` to create the environment. Torch install failed due to network restrictions.
3. Activated `.venv` and ran `python -m AGENTS.tools.dev_group_menu --install --codebases tools,speaktome --groups speaktome:dev`.
4. Encountered errors because the script resolved the `tools` path incorrectly. Modified `dev_group_menu.py` to map names to paths and pass `--no-build-isolation` during installation.
5. Attempted to run `python testing/test_hub.py`, but imports failed due to missing dependencies.

## Observed Behaviour
- `dev_group_menu` attempted to install from `/workspace/speaktome/tools` instead of `AGENTS/tools`.
- After patching path handling the install still failed because building `speaktome-agent-tools` requires network packages which were blocked.
- Test hub script reported missing dependencies even after environment setup.

## Lessons Learned
The codebase registry names discard directory prefixes, causing mismatched install paths. Explicitly tracking full paths resolves the issue. Test suite requires packages that may not be available without network access.

## Next Steps
- Investigate a lighter-weight install path for `speaktome-agent-tools` that avoids network dependencies.
- Ensure tests can run headlessly without external downloads.
