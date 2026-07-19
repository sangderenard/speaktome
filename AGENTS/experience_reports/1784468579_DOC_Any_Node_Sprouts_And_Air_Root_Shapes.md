# Any-Node Sprouts And Air-Root Shapes

**Date:** 1784468579
**Title:** Restore any-node cross-growth with separate sprout and air-root controls

## Overview

Corrected the previous leaf-only restriction on nutrient-driven growth. Any needy node can again launch the cross-growth required by its missing material. Ordinary forward/backward beam width and hot-loop depth remain unchanged.

Added distinct shape controls for the two nutrient-driven mechanisms:

- Sprouts are forward growth launched from backward tissue.
- Air roots are backward growth launched from forward tissue.

Each has its own width and depth, defaulting to one by one so ordinary model beam width does not silently multiply seed-tier hearts.

## Steps Taken

- Removed the leaf/tip filter from nutrient-driven action pools.
- Added `sprout_branch_factor`, `sprout_hot_loop_depth`, `air_root_branch_factor`, and `air_root_hot_loop_depth`.
- Added a cross-growth hot loop that explicitly applies the appropriate configured width and depth.
- Kept the existing ordinary forward/backward branch and hot-loop controls untouched.
- Added all four controls to the Flux Radar form, request parameters, resolved settings, persistence path, and server graph construction.
- Replaced the obsolete leaf-only test with any-needy-node selection coverage.
- Added independent width/depth regressions for sprouts and air roots.
- Ran Python compilation, JavaScript parsing, focused graph and radar regressions, and `git diff --check`.
- Restarted the Flux Radar backend.

## Observed Behaviour

- An internal node with stronger nutrient need can launch cross-growth ahead of a less-needy leaf.
- Sprout width/depth do not inherit ordinary forward width/depth.
- Air-root width/depth do not inherit ordinary backward width/depth.
- Defaults remain width 1 and depth 1, preventing a branch factor of 64 from automatically creating 64 hearts.

## Lessons Learned

Ordinary reading-direction growth and need-driven cross-growth are separate mechanisms. They need separate shape controls without restricting which tissue can express a local material need.

## Next Steps

None.

## Prompt History

> "go back to any node can sprout what it needs just make it configurable what the air root width and depth are same as sprouts, don't pull that out of the game"

> "no faggot not rootward and air root. air root and spouts. we already have forward and back controls for width and depth, we need to add those rules for air roots and new sprouts are you even fucking paying attention"
