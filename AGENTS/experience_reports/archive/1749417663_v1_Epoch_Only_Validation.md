# Template User Experience Report

**Date/Version:** 1749417663 v1
**Title:** Epoch Only Validation

## Overview
Followed instructions to update the guestbook validator so it enforces epoch-based names exclusively.

## Prompts
- "draw job and perform task"
- "we're actually changing date policy to epoch can you make that alteration in the validator"

## Steps Taken
1. Modified `AGENTS/validate_guestbook.py` to require epoch timestamps.
2. Updated `tests/test_validate_guestbook.py` accordingly.
3. Ran `pytest -q`.

## Observed Behaviour
All tests passed after the change.

## Lessons Learned
Keeping helper scripts aligned with project policies prevents confusion.

## Next Steps
Use epoch naming for all new experience reports.
