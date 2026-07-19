# Heart Token HUD And Radar Efficiency

**Date:** 1784466913
**Title:** Heart token reporting and radar render-path efficiency

## Overview

The Flux Radar heart HUD now identifies each heart with the token string represented by its current seed node. The radar frontend also caches snapshot-derived rendering data instead of rebuilding it during every animation frame, and the server reuses its already-computed orthogonal-node set when determining network roots.

## Steps Taken

- Read the heart HUD, graph snapshot, animation-frame, and orthogonal-network code paths.
- Added the current pump node's decoded token text to each heart heading and tooltip.
- Added snapshot caches for edge influence, pressure, water, ions, fluid volume, and filtered labels.
- Changed resolved link endpoint access to avoid redundant map lookups.
- Allowed `orthogonal_network_roots` to accept a previously computed orthogonal-node set.
- Added focused source and behavior regressions.
- Ran Python compilation, frontend JavaScript parsing, focused regressions, and `git diff --check`.
- Restarted the Flux Radar backend on port 8877.

## Observed Behaviour

- Heart headings use the form `region · “token string”` when token text is available.
- The animation frame consumes cached snapshot arrays and maxima rather than repeatedly sorting labels and rebuilding material/transport arrays.
- The live snapshot computes orthogonal membership once and passes it into network-root calculation.
- Python compilation, JavaScript parsing, and all focused regressions passed.
- The radar server listens on `127.0.0.1:8877`; `/api/live/state` reports that no live session is currently running, as expected.

## Lessons Learned

Snapshot-dependent visual data belongs in snapshot ingestion, not the animation loop. The graph's orthogonal membership is likewise reusable within a snapshot and should not be recomputed immediately by its next consumer.

## Next Steps

None.

## Prompt History

> "make the heart stats report the token string for the heart in the stats please and then do an efficiency pass"
