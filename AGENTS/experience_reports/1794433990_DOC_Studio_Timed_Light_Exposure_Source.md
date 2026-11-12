# Studio timed light exposure source

**Date:** 2026-11-11 UTC

## Overview

The owner clarified that Studio already streams changing light state and that a long exposure holds one integrator open across those states. The ordinary RGB still-photo path now records changed fixture material/profile states with durations in the browser, validates the ordered bundle in Turing, and processes each state in Spectral's native forward and camera readers without clearing the sensor between states. The path is source-complete but has no GPU execution receipt; the separate field-information prohibition in `turing/AGENT_OBSTINANCE.md` remains in force.

## Changes

- Turing's photo UI records only changed fixture light states during the requested shutter interval, including the elapsed interval boundaries. The MRP1 upload carries the bundle, and server validation rejects gaps, non-light material changes, and unsupported job modes. The original Studio scene blob and its hashes are unchanged when the timed path is not selected.
- Spectral prepares a stable positive native source population for all declared Studio and Shop emitter materials, then remaps each actual state onto the prepared material IDs. It installs current shade, MatBuf, native source roots, and camera area NEE at a complete sensor sweep boundary. Zero emission stays zero; an emitter that starts dark can later light.
- The camera sensor shader multiplies each sample by its interval duration divided by its number of full sweeps. One sensor accumulator spans all intervals. CameraTimeline and ExposureBarrier record emission materialization and detector commits for the single authored exposure frame.
- Timed Shop burst is now admitted only on this path. A sensor-closed interval turns Shop sources off before the configured flash pulse; the existing `camera_software.trigger_bus` and `RoutingGraph` resolve the flash to named burst emitters. Only configured trigger emitters respond; ordinary hot lights remain independent. The graph may be supplied as its canonical serialized config, including bus nodes. Other photo/live paths retain their burst guard.

## Checks

`spectral-analyzer`: 16 focused CPU/mock tests passed (`test_studio_native_emitters.py`, `test_marble_preview_worker_status.py`, `test_camera_emission_scheduling.py`). `turing`: 25 focused photo tests passed; `meowdoku_timed_exposure_frames_check.mjs` and `meowdoku_studio_scene_builder_check.mjs` passed. Python and Node syntax checks passed. No GPU exposure was run. The separate legacy `test_photography_table_contacts_saved_radius_in_real_worker_decode` and `meowdoku_marble_shop_bridge_check.mjs` remain blocked by a `lightField is not defined` error in their preexisting scene-packer VM harness; it is unrelated to the timed-light changes.

## Limits and next steps

The timed path accepts changes to light emission/material profile power on an unchanged fixture mesh and fixed camera. It currently selects ordinary untiled RGB Studio stills without field reception or client intersections. It records the scene in real browser time and then submits the bundle for tracing. The Studio UI does not yet expose the graph mixer board; the default route connects Flash to each configured burst emitter, and serialized RoutingGraph input can replace it. See `todo/studio_triggered_exposure_packet_integration.stub.md`.

## 2026-11-11 exposure tick, ISO, REST and MCP continuation

The owner corrected the clock model: opening the shutter hijacks Studio model
time and advances one chosen substep per completed display tick. This can run
slower than real time. The UI now exposes substep count, simulated ISO, and a
linear highlight clip percentile. MRP1 and server validation carry all three;
the Spectral worker uses the substep count for sensor sweep work and the
existing profiled display transform for ISO gain and percentile white. Raw HDR
is unchanged. The Nodus-backed REST picture builder now accepts an ordered
`lightTimeline` of complete Studio lighting records and builds the same timed
MRP1 bundle. SpeakToMe MCP `show` forwards these photo options, and the
generated REST MCP catalog documents them. The exact fields and limits are in
`turing/docs/STUDIO_EXPOSURE_TICK_CONTRACT.md`.

Source checks: Turing 37 photo/picture Python tests passed with one unrelated
skip; its timed recorder and Node Studio agent product checks passed. The
Node-emitted timed product decoded through Turing's real MRP1 server parser.
Spectral 61 focused CPU/mock tests passed, including 47 color-profile tests.
The SpeakToMe pytest suite could not initialize its repository environment
(`.venv/Scripts/python.exe` permission denied); MCP and REST catalog imports
passed under the available interpreter. No new GPU exposure was run.

The prior paragraph's real-time capture description is superseded: the
browser now follows contracted model ticks. Capture still completes before
the GPU worker begins, so GPU backpressure cannot currently pause an active
browser shutter. Moving geometry/camera remains outside the fixed-scene timed
photo validator.

## Prompt History

> the thing i don't get is this is already a stateful streaming live updating system, it does everything, we only need to not turn off accumulation when changes occur during an exposure for live, and deliver a bundle of frames of state with the ray tracer exposure

> long exposure is just literally holding the integrator open while the state changes

> no worries this is just a you level problem to take up, it was no longer a small agent issue

> please do the work, don't give me updates where there's nothing i can do but say okay well then you didn't finish

> so, if we button this up nicely, that exposure is going to be on the studio tick timeline, we should break the constatnt or whatever is driving dt, and replace it with a chosen number of substeps, allowing regulation of the queue of frames, and a sensitivity or iso-sim response that modifies the linear white curve to crop the top 25% in saturation, or something, then we need to make sure it's in the controls, it's in the contract, the rest api can use it, and the mcp contains our additions to the rest api take picture or other such paths, so you can mcp us exposures

> yeah it will be a little tricky but on exposure we hijack the time let it slow down if it needs to to do a lockstep sequence according to contract
