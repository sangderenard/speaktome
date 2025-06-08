# Bug Hunt Algorithm Guard

**Date/Version:** 2025-06-08 v5
**Title:** Added rule against modifying algorithm

## Overview
Updated `bug_hunting_job.md` with a strict warning not to change the algorithm. Parallel operations must never be replaced with loops.

## Prompt History
```
Add explicit and forceful rules not to modify the algorithm, which means principally, never, ever, ever, under any circumstances, replace a parallel operation with a loop
```

## Steps Taken
1. Edited the job description to include the new rule.
2. Created this experience report and ran guestbook validation.
3. Executed the test suite to verify no regressions.

## Observed Behaviour
The updated job description now emphasizes algorithm integrity.

## Next Steps
Continue to monitor bug fixes for compliance with this rule.
