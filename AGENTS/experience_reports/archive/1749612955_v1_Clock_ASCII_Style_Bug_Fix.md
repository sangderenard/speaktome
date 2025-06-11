# Clock ASCII Style Bug Fix

## Overview
Investigated crash in clock demo when cycling ASCII styles.

## Prompts
- "please investigate why it might be that the clock demo crashes from an empty ascii ramp when it should be cycling through many ramps all from the config json? you probably can't run the code, it uses control characters and color codes to live video the console, but you should be able to figure out how any empty ascii ramp could get in"

## Steps Taken
1. Searched code base for ASCII ramp handling.
2. Inspected `ThemeManager.cycle_ascii_style` and `clock_demo` key handling.
3. Added `step` parameter to `cycle_ascii_style` to support backward cycling.

## Observed Behaviour
`clock_demo.py` called `cycle_ascii_style(-1)` although the method accepted no argument. This raised a `TypeError`, preventing proper ramp cycling and giving the impression of an empty ramp.

## Lessons Learned
Ensure handler functions match expected signatures when referenced in key mappings.
