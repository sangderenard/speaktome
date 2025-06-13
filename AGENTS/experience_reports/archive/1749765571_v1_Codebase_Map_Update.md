# Codebase Map Update

**Date/Version:** 1749765571 v1
**Title:** Codebase_Map_Update

## Prompt History
- "AGENTS shouldn't be the codebase, it needs to be AGENTS/tools, can you look into that with the codebase map and the tool that builds it"
- Root `AGENTS.md` guidance to add a new experience report and run `AGENTS/validate_guestbook.py`.

## Observations
- Checked `AGENTS/codebase_map.json` which already listed a `tools` entry pointing to `AGENTS/tools`.
- Ran `python AGENTS/tools/update_codebase_map.py` with `PYTHONPATH` set to repository root to rebuild the map.
- Updated the map to include dependency groups discovered across codebases and verified the `tools` path uses `AGENTS/tools`.

## Next Steps
- Commit the refreshed `codebase_map.json` so setup scripts use the latest information.
