# Experience Report: Torch Group Fix

**Date/Version:** 1749780906 v1
**Title:** Move torch deps to torch group

## Overview
Adjust optional dependency groups so Torch packages are only installed when
`-torch` or `-gpu` is specified.

## Prompts
- "poetry is able to run, so we do not have a problem obtaining it. what we have a problem obtaining is torch, and your task which you must fix is to make the torch exclusion work correctly. All instances of torch in any requirements must be put in a torch specific group which MUST NOT be included unless specified with -torch or -gpu"

## Steps Taken
1. Renamed the `ml` group in `fontmapper/pyproject.toml` to `torch`.
2. Updated `AGENTS/codebase_map.json` accordingly.
3. Regenerated the codebase map via `python AGENTS/tools/update_codebase_map.py`.

## Observed Behaviour
The new group name mirrors other projects and makes it clear Torch is optional.
`setup_env_dev.sh` will not install these packages unless the torch flags are used.

## Lessons Learned
Explicitly naming Torch groups keeps dependency selection obvious. Renaming
avoids confusion with generic `ml` groups.

## Next Steps
- Test environment setup with and without the torch flag to verify exclusion.
