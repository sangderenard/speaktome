# Turing native Fortran columnar display

**Date:** 1785883354
**Title:** Isolated RGB proof and columnar fluid compilation through the common shell

## Overview

Added a small compiler-authored RGB probe and compiled the existing columnar
multifluid tick through the same AST/AOT → target IR → SSA → Fortran → C-shell
path. This separated native display behavior from the larger sorting learner
and from the fluid workload.

## Steps Taken

1. Added `native_fortran_display` as a thin compiler entrypoint for RGB tensor
   programs.
2. Fixed the registered Fortran target so module and contained procedure names
   cannot collide.
3. Fixed the generic C shell to find state/output arena files beside its
   executable rather than depending on the caller's working directory.
4. Compiled and visually launched a 320×180 RGB probe.
5. Compiled the 256×176 columnar fluid tick with all planes kept as runtime
   arenas and ran a bounded native frame.
6. Added native probe and relevant compiler regression coverage.

## Observed Behaviour

- The user confirmed the RGB probe displayed perfectly.
- One probe frame took approximately 0.48 ms and produced nonzero sums in all
  three channels.
- One 45,056-pixel columnar frame took approximately 202–213 ms, produced
  strongly non-black RGB output, advanced managed time, and updated the spring
  and entity state.
- Defender activity coincided with generation/compilation of new unsigned
  source, object, and executable artifacts; it was not required for frame
  computation.
- Windows Graphics Capture could see the custom window handle but could not
  snapshot its surface (`0x80004002`); the user performed the visual check.

## Lessons Learned

The blank learner window was not a general Fortran RGB-shell failure. A cheap
display proof is valuable before testing a large compiled graph. Native
artifact state must be located relative to the executable for direct launch,
and Fortran compilation units must not reuse their contained procedure names.

## Next Steps

Investigate the sorting renderer itself if it remains blank now that the same
shell is known to paint both the probe and columnar output.

## Prompt History

> "lets do an isolated test of the fortran producing an image because this appears to maybe have worked but also, windows antivirus went up way high in processing and the screen stayed blank, not sure if those were related, but lets try to use the exact same shell to compile the columnar fluid sim"

> "it worked perfectly, the rgb demo, something else launched for a moment and closed but i'm not sure it was related to your task"
