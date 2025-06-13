# Environment Setup Experience

**Date/Version:** EPOCH v1
**Title:** Environment Setup Experience

## Overview
Attempting to set up the repository environment using provided setup scripts.

## Prompt History
- "perform an experience report on trying to set up the environment. for every failure, every error, log it, why you think it happened, and whether you were following instructions or acting independently"

## Steps Taken
1. Ran `bash setup_env.sh -headless -notorch -codebases=speaktome`.

## Observed Behaviour
- Setup created `.venv` and upgraded pip and wheel.
- [DEBUG] output showed codebases auto-populated with all options despite specifying only `speaktome`.
- During execution of `dev_group_menu.py` two exceptions occurred:
  - `ModuleNotFoundError: No module named 'AGENTS'` (script couldn't import repository modules).
  - `ModuleNotFoundError: No module named 'tomli'`.
- Final message printed `[OK] Environment ready`, but additional error occurred when checking torch:
  - `AttributeError: module 'importlib' has no attribute 'util'`.
- The environment installed only `pip` and `wheel` (verified with `pip list`).
- Attempting `python AGENTS/validate_guestbook.py` failed with the same `ModuleNotFoundError` for `AGENTS`.



## Lessons Learned

- Setup scripts depend on local modules and additional Python packages that weren't installed by default.
- Headless mode auto-selected all codebases despite the `-codebases` argument, possibly due to argument parsing order.
- Failing to install packages meant the validation script could not run.

## Next Steps
- Re-run setup with full interactive mode or ensure necessary dependencies are installed.
- Investigate argument parsing in `setup_env.sh` so that headless selections are honored.
- Install `AGENTS` tools to allow running validation scripts.

