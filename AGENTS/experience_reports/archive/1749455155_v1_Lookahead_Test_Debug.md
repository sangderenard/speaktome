# Lookahead Test Debug

## Overview
Investigated why `test_lookahead_controller` skips on the pure python backend and fails on the default backend.

## Prompts
- "I want to know why test_lookahead_controller skips the pure python test and why the default fails because of an assert of none equals none"

## Steps Taken
- Installed `ntplib` and `numpy` to satisfy test dependencies.
- Ran `pytest tests/test_lookahead_controller.py`.
- Observed skip reason and failure.
- Updated the test to enable benchmark timing via `get_tensor_operations(track_time=True)`.

## Observed Behaviour
- Pure Python backend was skipped due to limitations.
- Default backend failed because `ops.last_op_time` remained `None`.
- After patch, all lookahead controller tests pass.

## Lessons Learned
- Benchmark timing is disabled by default in tensor operations; tests must explicitly enable it.

## Next Steps
- None
