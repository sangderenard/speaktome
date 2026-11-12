# Studio light dimmer defaults

**Date:** 2026-11-11
**Title:** Studio light dimmer defaults and trigger seam

## Overview

Renamed the rack control displayed as “House lights” to “Studio lights” while retaining the persisted `houseDimmer` key and `house-dimmer` control ID. Fresh rack state now starts with Studio at 100% and Shop at 15%. No exposure trigger routing was installed: the existing Spectral camera trigger graph has no Studio consumer, and Turing's shop emitter explicitly refuses burst exposure until authoritative timing-packet integration exists.

## Steps Taken

- Read the Spectral trigger-bus seam documentation and Turing Studio shutter note.
- Updated the existing rack defaults and control label; retained the stored schema.
- Ran `node tests/meowdoku_lighting_rack_controls_check.mjs` in Turing.

## Observed Behaviour

The focused rack control test passed. It checks the new 0.15/1 defaults and Studio label along with existing nullable override and control behavior. `resolve_camera_trigger_bus` continues to return control signals tied to the original ExposureFrame; no light energy weighting or frame admission changed.

Follow-up regression review updated the Studio lighting check to verify fresh A and B fixture state at `sourceLevel=400`, `dimmer=0.15`, `exposureMode='continuous'` (effective source gain 60). Calibration now resolves B through the exact same exposure profile as A, preserving the equality contract under non-neutral defaults. The rack-control and shop-light emitter checks pass. The broader Studio check passes these new assertions and calibration, then fails at line 744: `migratedTools.roomOverrides.values` lacks the expected `lightTools/attachments/@kicker-left-snoot/enabled` entry with value `false`. This migration expectation remains unresolved; no baseline status is claimed.

## Lessons Learned

The actual Studio still-photo route remains distinct from the thick-lens demo. Turing `packStudioPhotoParts` packs static exposure PBR and `emitter_profiles` into MRP1. `marble_photo.PhotoManager.start` launches Spectral's `camera_designer.marble_preview_worker`; its `prepare_photo_geometry` calls `native_emissive_sources` once, and its camera `SensorAccumulator` uses static scene/source buffers, one `set_nee_spectral_sources(source_records, original_buffer, area_scene)` call, then wall-clock `tick()` accumulation. The worker has no ExposureFrame or slice admission; Turing's `exposure_seconds` is a compute budget. The burst guard remains necessary.

The other path, `thick_lens_focus_lab.RayTracer.trace_forward`, does submit `_flash_emitter_tri_ids_i32` and `_natural_emitter_tri_ids_i32` separately with per-call `flash_exposure_weight` and `natural_exposure_weight` into native `submit_emissive_triangles`. It is not the Studio MRP worker and has no Studio fixture/bus receiver. `resolve_camera_trigger_bus` can resolve arbitrary receiver names to complex control signals on an original ExposureFrame, but no Studio consumer applies those signals to native timed emitters. A source-only adapter would not affect actual Studio capture; a correct bridge needs a source-group timing/admission API in the Studio worker's camera path, preserving the existing frame, timeline, barrier, and physical weighting contracts.

## Next Steps

Add a source-group timing/admission hook to the actual Studio `SensorAccumulator`/worker path, then connect named light/bus receiver signals and Shop cutoff to its existing camera frame lifecycle. Preserve ExposureFrame, CameraTimeline and ExposureBarrier identity and existing flash/natural energy weighting. Until that hook exists, retain the burst guard. Tracking stub: `todo/studio_triggered_exposure_packet_integration.stub.md`.

## Prompt History

> Work in C:\dev\Powershell. User explicitly asks you, a low-cost agent, to implement: change Studio UI/controls from 'House lights' and 'Shop lights' to 'Studio lights' and 'Shop lights'; fresh default Studio dimmer 100%, Shop dimmer about 15%; exposure should drive ordered light events: Shop light cuts out just before Flash trigger pulse reaches a signal router capable of addressing any light/bus; hot-only lights ignore the trigger, fixtures configured with trigger illumination emit at trigger, and only then scene is captured/ray traced. Reuse existing camera software near Spectral thick-lens demo and Turing/Studio machinery. Do not invent duplicate scheduler/router. Read root C:\dev\Powershell\AGENTS.md (user supplied), speaktome/AGENTS.md and relevant guestbook, Turing AGENTS.md/AGENT_OBSTINANCE.md, Spectral guidance if any. Existing relevant seam: spectral-analyzer/camera_software/trigger_bus.py, spectral-analyzer/docs/CAMERA_TRIGGER_BUS_2026-11-11.md, guestbook 1794411620_DOC_Camera_Trigger_Graph_Seam.md. Turing files: examples/meowdoku/web/marble_lighting_racks.js, marble_lighting_rack_controls.js, marble_lighting_rack_override.js, marble_studio_lighting.js; tests nearby. Turing and Spectral worktrees contain substantial pre-existing concurrent dirty edits, especially scene response and camera files: inspect diffs, preserve all unrelated edits, do not reset or commit. Do not run new GPU renders: Turing AGENT_OBSTINANCE forbids new render before missing data contract complete. Make source/config implementation and meaningful focused tests within existing architecture. Carefully distinguish trigger control signal from photon energy, preserve ExposureFrame/CameraTimeline/ExposureBarrier identities and physical weighting. If full integration cannot be done from existing interfaces, implement safe grounded subset and report exact remaining seam rather than fake it. Send me progress/findings, and final changed files/tests/limits. Do not restart production or publish.
