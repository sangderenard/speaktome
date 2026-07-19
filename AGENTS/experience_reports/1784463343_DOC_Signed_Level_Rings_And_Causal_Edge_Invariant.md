# Documentation Report

**Date:** 1784463343
**Title:** Signed-level rings and causal-edge invariant

## Overview

Corrected the Flux Radar ring coordinate system after investigating a reported
second-ring backward-to-second-ring forward connection. All main and cousin
seeds occupy the same signed middle layer, so ring index is now strictly the
absolute signed level: `0`, `±1`, `±2`, and so on.

## Steps Taken

- Inspected every audited edge and raw parent relationship in the current
  snapshot and the complete rolling history.
- Confirmed ordinary graph growth creates parent/child edges under the global
  backward-to-forward ownership rule.
- Removed the competing per-`center_id` breadth-first ring origins.
- Defined snapshot `display_radius` as `abs(level)`.
- Added `_connect` validation requiring every causal edge to advance exactly
  one signed level.
- Added a regression that explicitly attempts an invalid `-1 → +1` edge and
  requires a `ValueError`.
- Preserved cousin centers: they remain level-zero seeds.
- Isolated base SVG pipe joins (`line.fg-edge`) from pressure-halo line joins
  so repeated renders cannot reuse halo elements as ordinary edges.

## Observed Behaviour

- Focused signed-level, cousin-center, reroot, and invalid-edge regressions
  pass.
- Inline frontend JavaScript parses successfully.
- Before the correction, no structural ring-2 cross-side edge was present in
  the available ticks; the competing visual coordinate origins and polluted
  SVG line join could nevertheless create the reported appearance.
- The corrected server is listening on port 8877. Its saved state currently
  contains no live session, so a post-restart live snapshot was unavailable.

## Lessons Learned

All level-zero seeds, including cousin seeds, share one signed coordinate
system. Assigning separate radial distance origins by `center_id` breaks the
causal visual invariant because two endpoints can receive the same positive
ring number from different origins. Signed level already contains the required
topology: every legal edge changes level by one.

## Next Steps

None required. Start or load a live graph to inspect the corrected rendering.

## Prompt History

- "I'm noticing a connection I don't understand, I'm trying to realize how it could happen. there are second ring backward nodes with a direct connection to second ring forward. that can happen? but it shouldn't be happening"
