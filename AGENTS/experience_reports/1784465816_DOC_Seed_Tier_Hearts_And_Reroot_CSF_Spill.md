# Documentation Report

**Date:** 1784465816
**Title:** Seed-tier hearts and reroot CSF spill

## Overview

Restricted FluxGraph hearts to live signed-level-zero seed-tier nodes and made
heart reassignment/re-rooting conserve all chamber and reservoir contents by
spilling them into the shared CSF bath.

## Steps Taken

- Replaced orthogonal display-root pump discovery with an explicit level-zero
  seed-tier scan.
- Kept `main` attached to the anchor and mapped other live level-zero seeds to
  `net:<node id>`.
- Added a single full-heart CSF spill operation covering every chamber mixture
  and every reservoir.
- Changed main re-rooting to spill the old heart before moving focus.
- Reconciled all remaining heart owners immediately after signed levels are
  recomputed.
- Made any heart owner change spill the entire old heart rather than merely
  pumping reservoirs into old chambers.
- Reconciled stale off-tier hearts immediately when loading saved fluid state.
- Added conservation, off-tier cleanup, persistence, reroot, and legitimate
  cousin seed-heart regressions.

## Observed Behaviour

- Re-rooting places all old main-heart chamber and reservoir material in CSF.
- The new main heart is empty, owned by the new anchor, and has freshly sized
  empty reservoirs.
- A deliberately distributed level-one heart is removed and its chamber/store
  material is conserved in CSF.
- Legitimate level-zero cousin seeds remain heart pumps.
- Focused lifecycle regressions and Python compilation pass.
- The corrected backend is listening on port 8877.

## Lessons Learned

Orthogonal network roots are display groupings, not physiological seed owners;
they can occur away from the middle layer. Heart ownership must be derived
from signed level zero itself, and owner changes require a complete
conservation boundary into CSF.

## Next Steps

None required.

## Prompt History

- "there's an issue with hearts showing up destributed through the graph. only seed teir items should be hearts and all their contents should dump to the CSF when they are rerooted"
