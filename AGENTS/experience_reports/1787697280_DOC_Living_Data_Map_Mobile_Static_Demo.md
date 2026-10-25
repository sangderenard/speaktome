# Documentation Report

**Date:** 2026-08-25
**Title:** Living Data Map mobile static demo

## Overview

Added permissioned phone-orientation and acceleration movement plus always
available on-screen controls to the Living Data Map. Regenerated the complete
self-contained page and copied an identical static artifact into the shared
site demo collection at `site/demos/living-data-map/index.html`.

## Changes

- Added a coarse-pointer/mobile overlay with independent movement and look
  sticks, primary/IN, secondary/OUT, and jump buttons.
- Added an explicit `Enable phone tilt` gesture. It requests both
  `DeviceOrientationEvent` and `DeviceMotionEvent` permission where required,
  supports browsers without explicit permission methods, and explains the
  secure-context requirement when served without HTTPS.
- Calibrates the current handheld pose as neutral. Orientation supplies the
  main forward/strafe signal; acceleration contributes a deliberately small
  correction and can operate as a fallback if orientation is unavailable.
- Keeps the mobile overlay available in full-page shader mode and repositions
  the shader-only exit button to avoid the right look stick.
- Preserved keyboard, mouse, and gamepad behavior.
- Copied the generated single-file page byte-for-byte into
  `C:/dev/Powershell/site/demos/living-data-map/index.html`.
- Follow-up correction retains orientation/acceleration permission,
  calibration, and telemetry but removes sensor values from locomotion.
- Reduced full-deflection mobile look-stick speed from `2.4` to `0.72`
  radians per second.
- Mobile/coarse-pointer layout now collapses the inspector right rail and uses
  a single-column shell.
- The initial active tool is now slot 7, the physics-ball gun, rather than slot
  1, the form/attribute tool.
- Iterated the ball gun into a higher-energy toy: 8.5 m/s launch speed, lower
  drag, 0.78 world restitution, 0.9 ball-to-ball restitution, spectrum-derived
  projectile colors, procedural impact tones, 48 active balls, and 128 shots.
- Added a compact fftfree-derived 64-point radix-2 DIT C kernel. The C source
  is parsed with pycparser, fixed-size-specialized into Turing's FusedProgram
  numeric IR, and emitted as a content-addressed float32 WebAssembly module.
- Added an original generated eight-second PCM music loop. Its AudioContext
  playback cursor supplies each FFT window so the analysis stays synchronized
  with the sound rather than running from an unrelated animation clock.
- Added a user-gesture `Play music room` control and three cyan/magenta/yellow
  Pluck Phong point lights driven by low/mid/high FFT bands. The native shader
  fallback receives band-tinted key and ambient light.
- The music control remains available in mobile and full-page shader layouts.
- Physics-ball expiry is now an entity-to-object transition. Settled, timed-out,
  or capacity-evicted balls leave the physics worker and entity organization but
  remain rendered as static world pickups until walking over them or using the
  gun's secondary action returns one round to the ammunition stack.
- Pluck catalogue exports now apply a matte world-base policy: roughness >=
  0.72, specular <= 0.18, shininess <= 24, bounded metallic, no transmission,
  no self-emission/enamel coat, and a small authored indirect-light floor.
- Added ACES highlight compression and spectrum-aware exposure so music lights
  preserve local gradients instead of clipping broad regions to white.
- Added an opt-in multi-light shadow path to the original Pluck Phong fragment
  source. The static WebGL host allocates five 1024-square depth-array layers;
  each light's layer redraws only when that light transform or scene geometry
  changes. Ball spawn, settlement, and pickup therefore invalidate shadows.
- Added a native `audio/*` file chooser. A selected local track replaces the
  generated loop, starts immediately from the user gesture, and drives the same
  playback-cursor-synchronized embedded Wasm FFT and music lights.
- Replaced delayed player wall correction with a main-loop swept capsule clamp.
  Each collider remembers which side the player occupied, so a penetrated
  worker snapshot cannot accumulate and launch the camera away from a wall.
- The player now publishes an upright-capsule physics body with finite inverse
  mass, strong horizontal friction, and ball-to-player impulse coupling.
- Added downward-crossing top support for the player and for worker-owned balls.
  Collider tops can act as platforms, while crossings from below remain free.
- Replaced automatic settled-ball entity transitions with dynamic solver
  membership. Supported balls below the sleep threshold retain their entity,
  organization membership, geometry, and identity but release their worker
  snapshot slot and leave the active physics set. Player/ball collision and
  physics-field edits restore membership. Explicit secondary collection still
  uses the retained entity-to-pickup event transition.
- Generalized the projectile contract with event transitions for pickup,
  explosion, material deposit, and illumination so sleep remains orthogonal to
  semantic outcomes.
- Added three visible worker gears: full dynamics performs the compiled force
  solve at 120 Hz; confirmed constant-velocity motion enters a guarded 30 Hz
  kinematic coast that skips the Wasm force solve; full quiescence disarms the
  timer. Controls, impulses, contacts/bounds, insertions, collider edits, and
  physics-field edits return the worker to full dynamics.
- Added a generic, persisted tool-mode contract and a visible hotbar mode
  control (also cycled with `M`). The ball gun now has `normal` and `attractor`
  modes across mouse, gamepad, and mobile secondary controls.
- Captured viewport `contextmenu` before the browser host can consume it and
  split secondary input into press/hold/release phases. Normal mode charges
  exit velocity while held and launches on release.
- Attractor mode grows only inverse-square field strength. Its effective radius
  and sparse projectile membership are derived from the force epsilon; only
  projectiles above epsilon wake and rejoin force integration.
- Attractor-field members now transition into the existing ammunition pickup
  lifecycle when they reach the player, immediately restoring one round. Its
  primary action separately locks and continuously pulls only the projectile
  intersected by the crosshair ray; a proper ray-sphere test tracks its live Y.

## Verification

- Living Data Map, FFT, and projectile regression selection: `57 passed`.
- The embedded FFT Wasm maps a unit impulse to unit magnitude across all 24
  published bins in Node.
- Python compilation succeeded.
- The complete generated inline JavaScript parses with Node.
- Local browser verification confirmed live music/FFT activation with no
  console errors, mobile inspector collapse, ball-gun slot 7 selection, and a
  390×844 full-page shader canvas with the music control still visible.
- Chrome WebGL2 verification selected the compiled Pluck Phong program with no
  page console errors, exercised projectile entity→persistent-pickup behavior,
  and rendered the five-layer per-light shadow cache with music active.
- Pluck shader/material tests: `3 passed`; glslangValidator accepted the native
  OpenGL fragment both with and without `PLUCK_SHADOW_MAP` enabled.
- The generated source and site copy have identical SHA-256 hashes.
- The static page has no external script or stylesheet references; embedded
  WebAssembly and worker programs remain inside the single HTML file.
- Follow-up movement/audio selection: `61 passed, 1 stale assertion failed` on
  the first focused run; after updating that handoff assertion, `9 passed` in
  the focused regression. Python and generated JavaScript syntax checks passed.
- Chrome WebGL2 displayed both music controls, started the generated track with
  live FFT, and reported no page-origin console errors. Chrome's extension was
  not permitted to inject a test file into the chooser, so actual device-file
  selection remains a user gesture to verify on the published build.
- Dynamic membership and engine-gear selection: `60 passed` in the combined
  projectile/map/scheduler gate; generated JavaScript syntax passed. Chrome
  WebGL2 retained two settled entity cards and markers while telemetry changed
  from `2 active/0 sleeping` to `0 active/2 sleeping`. A force-free moving ball
  visibly entered `solver kinematic coast`; a quiescent clean page entered
  `solver asleep`, both without page-origin console errors.
- Tool modes and secondary input: `66 passed` in the focused tool, projectile,
  map, and state-loop gate; generated JavaScript syntax passed. Chrome WebGL2
  confirmed a right-button charged launch without a browser context menu, then
  switched the visible control to attractor and exercised its held release.
- Attractor absorption and crosshair pull: `66 passed` in the same focused gate;
  generated JavaScript syntax passed. Chrome WebGL2 confirmed normal firing,
  the attractor-only primary path, and broad-field release telemetry with
  separate influenced and absorbed-ammunition counts.

## Prompt History

> Hey can you get a copy of the whole static page into the c:\dev\powershell\site in an appropriate folder so i can experience it in mobile, and in that, can you ask for mobile positional/accel to let the player move by phone tilt or on screen controls

> Phone tilt was a bad idea, save the ability but don't navigate with it, and the look around control is too sensitive for a phone, then we need the site when it detects mobile to not use the right hand bar, and then we need the tool in hand by default to be the ball shooter not the attribute editor

> can you get those ball shooter changes, and also, lets iterate on the ball shooter for a minute, if we can close a tight loop on a fun to use toy in first person that's fulfilling just to run around playing with, and we could put in an in-page web assembly fft decomposition of an audio track we then play in sync with it's impacts, by using freefft and trying to get c/cpp source input to get that freefft algorithm, the one option that works that we need, compiled into our IR and then web assembly, a room with music and colorful music response plus a gun that is a physics toy, all that runs on a static site with a phong shader, we'd have a nice demo no matter what we do

> the balls need to remain and be picked up as ammo after they stop not just disappear, they stop being entities but should then become objects that can be picked up into inventory
>
> we need some way to normalize the gamma or something, so wild lights like that dim areas around them instead of just blow out
>
> the base materials for everything need to probably have their material catalogue entries made more matte
>
> if we original phong shader we compiled from didn't have any shadow pass we should maybe think about adding one and then seeing that it compiles glsl to opengl properly still

> I'm pretty sure we can afford to do a shade pass for any light any time it changes

> looks like it's working great, but can we please let the user load the music using a file dialogue

> also the player wall rejection is kinda wonky right now kinda waits and then launches isn't soft and doesn't notice, might need a was-on-this-side factor, don't wanna complicate it too much id' rather it remained how it is than get any less effective if it might sslow things down too much. we should give the player a body, too, impacted by physics like the balls, but with a high coefficient of friction, speaking of which, the z detection needs to learn about tops of things, blegh I know but we cant platform without it

> is there any way we could instead of transitioning those balls (but we need to keep the transition mechanics for events, like, they explode, they deposit turf, theylight up, etc.) but can we make the physics engine able to dynamically drop membership for anything slow enough and only pick membership back up if the object is touched by collision or a field change

> we should reveal this physics trick by letting the physics engine sleep when nobody is undergoing any force such that the only work is integration

> I was kinda saying we could shut down the force integration but keep motion, which would be like a second stage in what you're talking about making it even cooler if it knows it can downshift to yes things are moving but nothing is accellerating

> 1\) I'm not sure right click works on any tool, maybe the host isn't ignoring the right click right to let us handle it?
> 2\) I'd like tools/weapons to be able to be set in different modes, and then in the normal mode I want right to hold to charge the exit velocity instead of just one launch, launching on release
> 3\) I'd like another mode for that tool to induce an attractor field for any nearby projectiles, using an epsilon of possible force to weed out membership in the force, and right click would grow the force, pushing that radius out automatically and increasing the impact on each stray projectile naturally

> not strength and radius, radius naturally by strength, I spoke poorly

> the attractor should absorb the balls into ammo, and the left mouse button on attractor mode should pull only what is in the crosshairs

## Operational Note

Mobile sensor APIs generally require the page to be delivered from HTTPS (or
localhost). The on-screen sticks and buttons do not require sensor permission
and remain functional on an insecure origin.

The browser still requires an explicit user gesture before starting audio.
The imported FFT is intentionally the bounded radix-2 butterfly algorithm,
not fftfree's full Eigen/template runtime; the provenance and specialization
boundary are published in the page model.
