# Edge Subunit Padding

## Prompt History
- Developer: "always check the files in the repo ecosystem for your benefit..."

## Overview
Encountered a bug in `clock_demo.py` where `AsciiKernelClassifier` raised a shape mismatch when processing subunits at the frame edge. Investigated `get_changed_subunits` in `time_sync/draw.py` and found partial subunits were returned without padding.

## Steps Taken
1. Read root `AGENTS.md` to confirm experience report procedure.
2. Located bug via traceback pointing to `draw_diff` and `AsciiKernelClassifier`.
3. Implemented padding logic for edge subunits in `get_changed_subunits` and documented the behavior.
4. Added a regression test covering edge padding.
5. Attempted to run tests with `python testing/test_hub.py --skip-stubs` but environment lacked dependencies.

## Observed Behaviour
`testing/test_hub.py` aborted with missing imports. The new code compiles.

## Lessons Learned
Subunits at the frame boundaries must be padded to a consistent size before classification to avoid tensor shape mismatches.

## Next Steps
Set up the environment fully to run tests. Verify live demo no longer raises a runtime error.
