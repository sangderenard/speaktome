# Studio timed geometry slice repair

**Date:** 2026-11-12 UTC

## Overview

The fixed:8 timed Studio photo job `1eabe67dfb...` failed with `timed light frame changed native geometry`. Its baseline camera housing was packed from full precision camera coordinates, while its first timed slice used float32 coordinates. This made otherwise static camera triangles differ. The light-only worker guard also excluded captured geometry changes from the exposure contract.

## Changes and receipt

- Turing `7813a6a3` captures changed fixture vertices and material IDs with timed light frames; held frames reuse the same scene mesh. The REST decoder validates bounded geometry slices.
- Spectral `917183f` normalizes the initial timed camera pose, installs changed native geometry and BVH at a sweep boundary, relocates prepared emitter roots by triangle-local coordinates, and keeps sensor accumulation open.
- A slice that cannot retain the prepared geometry/source identity uses the last installed triangles and records `stale_geometry_slices` in job status and Studio UI.
- The Stable service was reloaded with `turing/examples/meowdoku/deploy/Restart-Stack.ps1`. The public DNS Studio and whoami paths both returned HTTP 200.
- A signed-in public Stable fixed:8, 128-pixel, 0.1-second job `5f2ddb91cc...` completed six state frames with `stale_geometry_slices: 0` and image `preview-000006.png`.
- Focused checks: 19 Spectral tests, 27 Turing photo tests, and the timed-frame JavaScript check passed.

## Limits

The slice packet carries Studio fixture geometry, including the room, table, backdrop and lamps. The authenticated saved subject mesh remains the exposure's base object. Incompatible topology falls back to the previous installed geometry; its slice is counted rather than aborting the shutter. The original failed job is immutable and remains an error record.

## Prompt History

> "timed light fram changed native geometry" why am I getting that error and why is that considered an error

> ok someone violated the spec, that's not how that works, we make geometry slices and send them that was the only authorized design. fix it

> let it not update if there's no update I don't want it compiling triangles every .1 frame

> also, like, so what if there is a change even if adapting to change doesn't work yet, it should have done the exposure with the stale triangles
