# Header Check Runner Repairs

**Date:** 1749831291
**Title:** Integrate auto_fix_headers into run_header_checks

## Overview
Expanded `run_header_checks.py` so it now repairs, validates and tests
module headers. The tool first runs `auto_fix_headers.py`, then
`validate_headers.py` and finally `test_all_headers.py`.

## Steps Taken
- Implemented argument parsing and updated runner logic
- Updated documentation in `README.md` and `AGENTS/tools/README.md`
- Wrote this report and validated the guestbook

## Observed Behaviour
`run_header_checks.py` exits with the combined status code from the
invoked scripts.

## Lessons Learned
Chaining the repair and validation steps ensures header issues are fixed
before tests run, reducing churn.

## Next Steps
Monitor for any remaining uses of `header_utils` and expand result
reporting.

## Prompt History
```
does validate headers use repair headers? put the repair script in that stack
```
