# User Experience Report

**Date/Version:** 2025-06-07 v1
**Title:** Project Reorganization

## Overview
Restructured the repository to place all Python modules under a dedicated `speaktome/` package. This keeps the root clean and makes it easier to locate the code. Added a `testing/` folder for lightweight evaluation scripts.

## Steps Taken
1. Moved source files into `speaktome/` and updated imports that relied on relative model paths.
2. Created `testing/lookahead_demo.py` as a simple script exercising the `LookaheadController`.
3. Updated `README` to describe the new layout.
4. Ran `todo/validate_guestbook.py` to ensure the guest book rules still pass.

## Observed Behaviour
- The validation script reported no filename issues.
- Running the demo script prints the tokens and scores after a short lookahead run.

## Lessons Learned
Keeping source under a package and moving test scripts out of sight makes the repository easier to navigate without affecting functionality.

## Next Steps
Consider expanding the testing utilities to cover more components as they stabilize.
