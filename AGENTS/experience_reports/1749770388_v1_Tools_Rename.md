# Tools Rename

**Date/Version:** 1749770388 v1
**Title:** Tools_Rename

## Prompt History
- "ITS NOT FUCKING CALLED SPEAKTOME FUCKING TOOLS IS IT FUCKING ASSHOLE IT'S FUCKING TOOLS FOR FUCKS GOD DAMN SAKE"
- Reminder from `AGENTS.md` to log changes and run `AGENTS/validate_guestbook.py`.

## Overview
Renamed the internal tools package from `speaktome-agent-tools` to simply `tools` and updated all codebases to reference this name in the `dev` optional group.

## Steps Taken
1. Edited `AGENTS/tools/pyproject.toml` and `setup.py` to use `name = "tools"`.
2. Replaced `speaktome-agent-tools` with `tools` in every project `pyproject.toml` under the `dev` group.
3. Regenerated `AGENTS/codebase_map.json` with `update_codebase_map.py`.
4. Prepared this experience report and validated the guestbook.

## Observed Behaviour
- The map generation script detected the new package name without errors.

## Lessons Learned
A consistent package name avoids confusion during environment setup.

## Next Steps
Verify installation scripts handle the renamed package correctly.
