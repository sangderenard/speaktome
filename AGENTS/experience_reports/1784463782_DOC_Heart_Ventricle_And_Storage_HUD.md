# Documentation Report

**Date:** 1784463782
**Title:** Heart ventricle and storage HUD

## Overview

Added a compact upper-right HUD inside the Flux Radar graph stage that reports
all heart ventricular contents and seed-reservoir storage telemetry for the
active snapshot.

## Steps Taken

- Added a translucent, scrollable graph-corner overlay.
- Unified heart and reservoir region keys so every available heart appears.
- Added per-region heart count and total ventricular volume.
- Added one row per ventricular chamber with total mixture, solvent, and every
  nonzero ion/component amount.
- Added one row per reservoir with total storage volume, ion amount, solvent,
  fullness percentage, concentration, target band, and owner ID tooltip.
- Wired HUD updates into `applySnapshot`, so live mode, playback, and history
  scrubbing all display the selected snapshot rather than only current state.
- Added focused frontend regression coverage.

## Observed Behaviour

- The running server serves the new HUD markup and renderer.
- Inline frontend JavaScript parses successfully.
- The focused HUD regression, Python compilation, and `git diff --check` pass.
- No live session was running during validation, so the served HUD correctly
  starts with its `no heart state` placeholder until a graph is loaded.

## Lessons Learned

The backend already exposed complete `hearts` and `reservoirs` telemetry. The
missing piece was a compact presentation that preserved individual chamber and
store detail while remaining usable when many cousin-network hearts exist.

## Next Steps

None required.

## Prompt History

- "can you put in a little hud in the corner of the graph with stats on all the ventricals and storage of all the hearts"
