# Submodule Update Utility Extraction

**Date:** 1749972101
**Title:** Submodule Update Utility Extraction

## Overview
Pulled the recursive submodule update logic from archived git hooks and
implemented it as a standalone tool under `AGENTS/tools`.

## Steps Taken
- Created `update_submodules.py` replicating the hook commands
- Attempted to run the script to verify behaviour

## Observed Behaviour
- Script attempted to clone submodules but failed with network errors
  (`ssh: connect to host github.com port 22: Network is unreachable`)
  leaving the repository unchanged.

## Lessons Learned
- Submodule updates rely on network access; the wheelhouse submodule cannot be
  cloned in this environment.

## Next Steps
- Investigate using HTTPS or an internal mirror to populate the submodule.

## Prompt History
- "try to pull the submodule recursive update code out and test if it gives us
  fresh clean copies of the submodule"
