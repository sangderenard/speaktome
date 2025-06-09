# Dev Group Menu Dynamic Discovery

**Date/Version:** 1749496649 v1
**Title:** Implement dynamic group discovery and rename outbox drafts

## Overview
Implemented automatic codebase and group detection for `AGENTS/tools/dev_group_menu.py` and renamed unlabeled outbox messages.

## Prompts
```
check the agents/messages/outbox and rename the untitled markdowns something apropriate, and on the message that is about:

------

# Proposal: Dynamic Codebase/Group Discovery System

## Context
Currently `AGENTS/tools/dev_group_menu.py` uses a hardcoded dictionary of codebases and their groups. This creates maintenance overhead and risks getting out of sync with actual codebase configurations.
------

I want you to work on building out the dev_group_menu to spec
```

## Steps Taken
1. Renamed `Untitled-1.md` and `Untitled-2.md` with epoch-based filenames.
2. Implemented functions to read `CODEBASE_REGISTRY.md` and extract optional dependency groups from `pyproject.toml` files.
3. Updated `dev_group_menu.py` to build its menu from discovered data.
4. Installed `cffi` and `setuptools` to satisfy test dependencies.
5. Ran `python testing/test_hub.py --skip-stubs` to verify all tests pass.

## Observed Behaviour
- Tests initially failed due to missing `cffi` and `setuptools`.
- After installing the packages, all tests passed (31 passed, 30 skipped).

## Lessons Learned
Dynamic discovery reduces manual maintenance and keeps setup options consistent with project configuration.

## Next Steps
- Consider caching group discovery results for speed if needed.
