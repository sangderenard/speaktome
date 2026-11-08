# Marble photo camera and live integration

**Date:** 2026-11-07

## Overview

Canonical Turing now serves a physical 3D toy course in Marble Studio with 1–16 identical modeled marbles, effective contact stiffness, and physical spin. The Spectral thin-lens photo offers a built-in close-up and a snapshot of the current Studio orbit/zoom. The root tracker records the connected work and remaining boundaries.

## Steps Taken

- Committed physical sander/vibrator case motion (`d53e7fec`), shape-contact precision boundary (`13f93a9c`), and effective marble stiffness (`cf5a2fbe`). Integrated agents' per-ID rotation, Studio toy-course, and emoji tumble commits (`a64fe50c`, `167c4434`, `aa9653d0`).
- Added the missing `emoji_tumble.js` static export after a browser check exposed a 404. Re-exported the site. The dev browser then loaded the Phong marble, compiled Studio course, D-pad, and three-marble control without console errors.
- Corrected Studio's marble follow view to look from the backplate normal, lowered its near plane, and set a marble-scale camera distance. The prior view looked through opaque rails; the dev browser now showed the authored cat-eye marble above the backplate.
- Added thin-lens built-in/current-orbit camera choice and changed the built-in field of view from 36° to 18°. An existing 512px photo showed a roughly 140px-diameter marble, so the narrower angle aims for about 280px. The photo camera payload test passes; a fresh GPU photograph of both placements remains to inspect.
- Restarted Python dev 8768 and live HTTPS 8766 on canonical Turing, retaining `--no-wasm`. Both `/api/ready` responses reported the WebGPU toy and all 3D product families ready and settled. Live `/api/marble_preview_spec?count=3` returned a 4k toy spec with three active start ages.
- Focused checks: Node camera, emoji tumble, and case vibration checks passed; six Python marble photo/game tests passed. The source-tree combined Node Studio test imports generated `turing-wasm-shell.js` and cannot run without a generated build file; no Wasm compile was triggered.

## Observed Behaviour

The 3D law carries angular velocity and uses sphere contact torque and rolling resistance. Visual emoji tumble is limited to 8,192 displayed IDs and 64 replay pose frames; 2D law has no angular state. Random saved designs are disabled because the current four-species law cannot assign each saved model's physical properties without additional mapping. Glass/metal/ceramic stiffness presets are stable effective contact frequencies rather than measured Young moduli.

Spectral's photo worker traces triangle BVHs in both forward and sensor GPU passes. Its native parametric sphere contract is a separate path. `spectral-analyzer/MARBLE_PARAMETRIC_PHOTO_HANDOFF_2026-11-07.md` records the real continuous-shell implementation boundary; no misleading toggle was added.

## Next Steps

1. Expose one short GPU photo with each camera choice and inspect marble framing and checker placement, then retain the 10-minute RGB/HDR contract for saved designs.
2. Measure emoji tumble on a large displayed field, and implement explicit per-ID saved-material/species assignment before enabling random saved marbles.
3. Implement the two-pass analytic shell primitive and material-zone hit mapping before offering a parametric photo switch.

## Prompt History

> speaking of which with roll state comes the emojis must tumble

> when is friction and rolling expected, or is that coming with emoji tumble

> let the marble studio camera controls have an option for doing the current camera as placement for the thin lens, for it's built in consider a closer framing by maybe doubling apparent marble size in frame

> the ray tracer should also support a parametric surface mode btw, we should be able to toggle, because sometimes I like the triangulated glass artifacts but we technically have a continuous wya to do it
