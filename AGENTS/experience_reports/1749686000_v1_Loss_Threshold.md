# Loss Threshold Detection

## Prompt History
- User: "well fix it to be a loss based metric with a threshold now please"
- Developer: "always check the files in the repo ecosystem for your benefit..."

## Overview
Implemented a configurable loss-based change detection in `get_changed_subunits`. The function now measures mean absolute difference and compares it against a provided threshold.

## Steps Taken
1. Inspected existing tests and `draw.py` to locate change detection logic.
2. Added `loss_threshold` parameter and loss computation.
3. Wrote a new regression test verifying threshold behavior.
4. Documented the session in this report.

## Next Steps
Ensure the environment setup allows running the full test suite.
