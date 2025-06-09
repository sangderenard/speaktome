# Following Repo Guidance

**Date/Version:** 1749507087 v1
**Title:** Follow Repo Guidance

## Overview
Explored repository instructions and attempted to run tests as suggested. Added a guestbook entry to document the session.

## Prompts
- "follow repo guidance on how to use software inside it and generally follow directions"

## Steps Taken
1. Read root and nested `AGENTS.md` files.
2. Ran `python -m AGENTS.tools.dispense_job` to get a task.
3. Executed `python testing/test_hub.py` which failed due to missing optional dependencies.
4. Created this experience report.

## Observed Behaviour
- `testing/test_hub.py` ended with several module import errors indicating missing optional packages such as `cffi`.

## Lessons Learned
Following the guidelines led to the discovery of missing dependencies. The project encourages recording such failures for context.

## Next Steps
- Install optional requirements when feasible to run full tests.
