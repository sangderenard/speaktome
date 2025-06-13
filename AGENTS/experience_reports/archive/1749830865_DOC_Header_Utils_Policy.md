# Header Utils Policy Update

**Date:** 1749830865
**Title:** Document header_utils deprecation and update runner

## Overview
Removed the stray import of `header_utils` from `run_header_checks.py` and
implemented the script so it sequentially runs the other header utilities.
The README now explicitly states that `header_utils.py` should never be
imported.

## Steps Taken
- Edited `run_header_checks.py` to fetch `ENV_SETUP_BOX` via `os.environ`
  and run `validate_headers.py` then `test_all_headers.py`.
- Updated `README.md` and `AGENTS/tools/README.md` with a warning about
  `header_utils`.
- Documented the change here.

## Observed Behaviour
`run_header_checks.py` exits with the combined status code from both
utilities.

## Lessons Learned
Clarifying policy in documentation avoids confusion about deprecated
modules.

## Next Steps
Monitor for any remaining imports of `header_utils` across the tree.

## Prompt History
header_utils is never ever allowed to be imported anywhere in any code. Put that in primary repo documentation and remove it from your run header checks script. also verify run_header_checks isn't completely arbitrarily superfluous, adding more confusion to a tools directory that already has confusion about division of tasks
