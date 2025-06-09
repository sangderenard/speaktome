# Tensor backend timing fix

**Date/Version:** 1749431888 v1
**Title:** Fair Tensor Benchmarking

## Overview
Ensured the tensor backend test measures only the operation time using the `benchmark` helper.

## Prompts
- "make sure the new tensor back end testing program reports time for tasks in a fair way that doesn't include tracking the test prep. time tracking should've been added to the abstract class"

## Steps Taken
1. Updated `tests/test_tensor_backends.py` so each operation is wrapped with `ops.benchmark`.
2. Ran `pytest -q` to confirm all tests pass.

## Observed Behaviour
- Test output shows `25 passed, 20 skipped`.
- Timing is recorded per operation when interactive mode is enabled.

## Lessons Learned
Proper placement of the benchmark wrapper avoids inflating timings with test setup overhead.

## Next Steps
Monitor further tensor scripts for similar timing issues.
