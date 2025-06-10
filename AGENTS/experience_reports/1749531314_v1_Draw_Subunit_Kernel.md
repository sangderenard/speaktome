# Draw Subunit Kernel

## Overview
Implemented integration of pixel diff output with the draw utilities. Changed clock_demo to convert framebuffer diffs into subunit structures and render them through `draw_diff`. Updated `draw.py` to include default ASCII ramp and added tests for subunit diff helpers.

## Prompts
- "before finishing, implement the use of draw.py functions for assembling the image change subunits and casting them with the ascii kernel"

## Steps Taken
1. Added imports and default parameters to `draw.py`.
2. Converted framebuffer diff tuples in `clock_demo` to subunit arrays for `draw_diff`.
3. Created new tests verifying `get_changed_subunits` and the default kernel.
4. Installed missing packages (`cffi`, `setuptools`, `ntplib`, `colorama`, `numpy`) and ran the test suite.

## Observed Behaviour
The new draw integration renders changed pixels via the ASCII kernel. Individual tests for draw utilities pass while the full suite still reports missing backend features.

