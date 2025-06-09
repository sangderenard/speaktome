# User Experience Report

**Date/Version:** 1749467959 v1
**Title:** Tests Made Difficult

## Overview
Added more challenging scenarios to several tests to exercise edge cases.

## Prompts
- "go through the tests and make them more difficult"

## Steps Taken
1. Reviewed existing test suite.
2. Parametrized n-gram diversity test over additional n values and longer beams.
3. Parametrized lookahead controller test to run multiple lookahead steps.
4. Added new test for snap_beam_path using insert_under_beam_idx.
5. Added garbage collection limit test for BeamRetirementManager.
6. Ran `python testing/test_hub.py`.

## Observed Behaviour
Tests passed locally with the new cases executing.

## Lessons Learned
Expanding small tests with parameterization quickly increases coverage without large code changes.

## Next Steps
Explore remaining modules for under-tested functionality.
