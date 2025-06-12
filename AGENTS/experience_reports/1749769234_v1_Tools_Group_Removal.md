# Tools Group Removal

**Date/Version:** 1749769234 v1
**Title:** Tools_Group_Removal

## Prompt History
- "there is no speaktome-agent-tools unless you rename the tools entry"
- Root `AGENTS.md` guidance to document actions in a guestbook entry and run `AGENTS/validate_guestbook.py`.

## Observations
- Each project listed `speaktome-agent-tools` in its `dev` optional dependencies, but this package isn't published externally.
- The environment setup scripts install `AGENTS/tools` directly, so the extra dependency caused confusion.

## Steps Taken
1. Reverted `dev` groups across all `pyproject.toml` files to remove `speaktome-agent-tools`.
2. Ran `PYTHONPATH=$(pwd) python AGENTS/tools/update_codebase_map.py > AGENTS/codebase_map.json` to rebuild the map.

## Lessons Learned
- Optional dependency groups should only reference installable packages. Including internal paths can mislead users and break setup workflows.
