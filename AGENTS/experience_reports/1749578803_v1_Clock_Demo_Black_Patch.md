# Clock Demo Black Patch Investigation

**Date/Version:** 1749578803 v1
**Title:** Clock demo renders a black patch by default

## Overview
Investigate why running `time_sync/clock_demo.py` with no
arguments shows only a black area in the terminal instead of the
expected clocks.

## Prompts
```
the clock demo had some adjustments to rendering path. as it stands, to
open with no specificity, by default, what renders is a black patch.
Please trace procedurally what happens when the program is run with
pygame installed and nothing is specified on the command line, find out
why now, having worked on putting in a pygame element, and after putting
in a number of new style features and functions, the rendered "image" on
ascii is black
```

## Steps Taken
1. Viewed repository documentation and explored time_sync package.
2. Attempted to run `python time_sync/clock_demo.py` but imports failed
   because dependencies were not installed.
3. Examined `ascii_digits.py` and related modules to follow the render
   pipeline when Pillow is missing.

## Observed Behaviour
Running the demo without Pillow installed causes the clock drawing
functions to print a warning and return `None`. The frame buffer thus
remains filled with zeros, so the terminal displays an empty black area.

## Lessons Learned
The default clock demo relies on Pillow even when only `pygame` is
installed. Lack of Pillow leads to no rendered clocks, resulting in a
black patch.

## Next Steps
Install the full `time_sync` package (which depends on Pillow) to enable
pixel rendering. Confirm that the demo shows clocks when Pillow is
present.

