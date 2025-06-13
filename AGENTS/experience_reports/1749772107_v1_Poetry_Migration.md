# Poetry Migration

**Date/Version:** 1749772107 v1
**Title:** Poetry_Migration

## Prompt History
- "continue the process of reducing and eliminating the need for setuptools in favor of poetry for all configurations and installations"

## Overview
Removed the remaining `setup.py` from `AGENTS/tools` and updated documentation to use Poetry commands. Added high-visibility stubs in legacy training files indicating that packaging has moved to `pyproject.toml`.

## Steps Taken
1. Deleted `AGENTS/tools/setup.py`.
2. Replaced pip instructions with Poetry commands in `speaktome/AGENTS.md` and `speaktome/README.md`.
3. Converted `training/notebook/1731933842_NoClass*.py` into stubs referencing Poetry packaging.
4. Ran `AGENTS/validate_guestbook.py` to ensure guestbook consistency.

## Observed Behaviour
- `poetry install` fails due to blocked network access to download.pytorch.org, preventing full environment setup.

## Lessons Learned
- Defining external package sources in `pyproject.toml` can trigger network requests even when groups are skipped.

## Next Steps
- Consider removing or conditionally enabling PyTorch sources to allow offline installations.
