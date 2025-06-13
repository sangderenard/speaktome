# Tools Group Addition

**Date/Version:** 1749769533 v1
**Title:** Tools_Group_Addition

## Prompt History
- "I am asking you to put tools inside a dev group on all entries are you having a hard time parsing english right now"
- Root `AGENTS.md` reminder to record experience reports and run `validate_guestbook.py`.

## Overview
Restore `speaktome-agent-tools` as a dependency in the `dev` optional group for every project so the setup scripts can include helper tools.

## Steps Taken
1. Edited each `pyproject.toml` to include `speaktome-agent-tools` under the `dev` optional group.
2. Regenerated `AGENTS/codebase_map.json` using `update_codebase_map.py`.
3. Created this experience report and ran the guestbook validator.

## Observed Behaviour
- The script correctly detected the new `dev` groups across all projects, confirming the map generation works from TOML files.

## Lessons Learned
Adding the tools package as a dev extra ensures common utilities remain accessible during development without affecting runtime dependencies.

## Next Steps
- Revisit the environment setup scripts to ensure they leverage these groups when installing editable packages.
