# Marble Madness steering and physical nudge

**Date:** 2026-11-07

## Overview

Canonical Turing commit `b08a184b` fixes Marble Madness steering across server state refreshes and adds two visible two-axis controls. The tilt thumbstick changes the existing 3-D fountain law's gravity vector; the nudge thumbstick gives the containing case one physical translation pulse through its existing `case_acc_x/y` runtime inputs. No law source or Wasm product was rebuilt.

## Diagnosis

- Neutral gravity `[0, 0, -1]` is normal to the backplate, so a marble at rest has no deliberate force along the course. This is the requested neutral state. A dev browser check by the parent agent held Down Y at 95%; the marble moved and reached the goal, showing that the compiled law's steering path works.
- `web/app.js` called `compute.setGravity(0, 0, -1)` on **every** server state refresh. This overwrote held steering until the next render frame, producing intermittent control behavior.
- Marble Madness inherited `meowdoku.gscale` from the fountain's saved setting. Its own controls do not display that slider, so a saved zero could silently suppress all gravity steering. Marble Madness now uses scale 1 and restores the saved fountain scale when leaving the game.

## Implementation

- `web/app.js` preserves the current steering vector across ordinary state refreshes and resets it only for a new game or compute epoch. Keyboard and optional phone tilt still feed the existing gravity vector.
- `web/index.html` and `web/marble_controls.js` provide distinct two-axis tilt and nudge thumbsticks. Nudge direction and radial strength are selected by dragging, then fired on release.
- `web/case_vibration.js` defines a compact case displacement `d=A·64u³(1−u)³` for `0<u<1`. Displacement, velocity, and acceleration vanish at both endpoints. The analytic second derivative supplies `case_acc_x/y` to the law; the same displacement animates the visible case. This is one pulse, separate from periodic sander/vibrator motion.
- `web/compute.js` samples the pulse once per snapshot, composes it with any selected case vibration, and clears it on compute reset. A second nudge waits until the first has reached the displayed timeline.

## Verification

- `node tests/meowdoku_marble_controls_check.mjs`: passed, including neutral backplate vector and two-axis thumbstick normalization.
- `node tests/meowdoku_case_vibration_check.mjs`: passed, including nudge endpoint conditions, strength scaling, and finite-difference agreement with analytic acceleration.
- `node --check` for `web/app.js` and `web/compute.js`: passed.
- `git diff --cached --check`: passed before commit. Dev HTTP 8768 returned 200.

The parent agent is performing site export, browser interaction checks, and server restart. Append those observed results here once complete.

## Fountain3D transition regression

After the first steering change, server `/api/log` contained a Fountain3D toy run with `cameraGravity:false`, phone state `waiting`, spin zero, but law gravity `(0,0,-1)` and 4,096 grains settled at almost zero kinetic energy. Another Fountain3D run had its authored `(0,-1,0)` gravity. This is direct evidence that the shared `ComputeRuntime` retained Marble Madness's backplate-normal gravity across a game switch. `compute.sync` resets the product but preserves `this.gravity`, and non-camera Fountain3D frames do not overwrite it.

Canonical Turing commit `14cd208d` restores the active fountain gravity source when leaving Marble Madness: use the available phone vector if the sensor is enabled, or `null` so the fountain spec supplies its authored down vector. Camera-down mode writes its view-derived gravity on the next frame. It also restores the user's saved case spin, motion mode, tilt, and gravity scale that Marble Madness forced off. Direct Fountain3D opening still starts with `gravity=null` and the authored default. The focused Node regression covers no sensor, waiting sensor, camera mode, and live phone vector; `node --check`, case vibration check, and `git diff --cached --check` passed. Browser QA and export remain with the parent agent.

## Next Steps

The current parent task owns deployment and browser QA; there is no deferred task from this report.

## Prompt History

> Can you look into the marble madness gravity i said normal to back plane but that's before the controls the controls still need to move the interior state the ball is hanging out in the top corner right now

> Marble Madness needs TWO visible 2-axis thumbsticks: one tilt X/Y, the other nudge vector and strength (radial magnitude). Please implement using existing physical runtime case_acc_x/y for a bounded nudge, or identify another existing law mechanism if better. No fake mesh motion.

> Please avoid interpreting the nudge stick as continuous 2 Hz vibration; user asked a directional nudge vector and radial strength, distinct from the sander/vibrator mode.

> Urgent regression from user: toy motion shoves everything into a corner again. Suspected Marble Madness -> Fountain3D transition retains `compute.gravity=[0,0,-1]` in shared ComputeRuntime; your applyState exit only restores gravityScale. Trace exact transition and existing gravity source, fix so toy resumes its own default/phone/camera law without a leaked marble vector.

> User correction: my transition theory is not the reported bug. They say old toy corner-shove bug was fixed earlier in development and has returned with recent changes, even outside Marble→Fountain transition. Please trace git history/previous fix and current toy motion path directly. Report measured cause before edit.
