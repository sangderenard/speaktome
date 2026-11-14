# Nodus joined wave kernel schedule

**Date:** 2026-11-13 UTC

## Overview

The Nodus ray-kernel review branch incorporated the existing compiled wave
kernel row proof and added an ordered two-stage `KernelPipelineTool` trial.
This is a state-library work schedule for T4 arithmetic, not yet a T4 arena
or a complete T1–T5 exposure.

## Steps Taken

- Read the Nodus camera-pipeline and T3/T4/T5 guestbook notes, the current
  kernel-pipeline receipt, and the authored Spectral wave kernel adoption.
- Cherry-picked Nodus's existing wave-row execution proof onto
  `codex/thick-lens-t1-t5-wave`.
- Extended `tests/spectral_wave_vulkan_test.cpp` to pass a 512×512 complex
  field from angular propagation to aperture material through the same
  state-library addresses in one two-row schedule.
- Built in user build space at `C:/Users/alber/nodus-build/thick-wave-cmake`.

## Observed Behaviour

The joined trial returned two successful ordered receipts and verified all
262,144 composed complex outputs on Vulkan devices 0, 1, and 2. Combined
dispatch times in this run were 1.561, 1.658, and 1.766 ms. The original
one-row regression also passed on all three. Logs are at
`C:/Users/alber/nodus-build/wave-joined-angular-material-512.log` and
`C:/Users/alber/nodus-build/wave-single-angular-512.log`.

The test also wrote the verified device-0 complex tensor to
`C:/Users/alber/nodus-build/nodus-joined-wave-field.png`. Its RGB channels
show signed real amplitude, signed imaginary amplitude, and normalized
power. The visible diagonal bands come from the authored periodic fixture;
the image is not a scene render or a diffraction claim.

The current Nodus ray proof still has distinct sensor→T1→exact T2→T3 and
T3→sort/pack/scatter→T5 schedules. This new wave schedule establishes an
intra-T4 field handoff, but does not route a T3 branch into the T4 arena or
return a wave continuation to T5. The existing Marble Studio picture at
`C:/Users/alber/nodus-build/marble-service-picture-a3cb.png` is a Studio
service image, not a full Nodus-produced photo.

## Next Steps

Carry the authored T3 branch handle and arena identity in the work contract,
preserve fftfree transforms, aperture mask, power accounting and Jones
coordinate mapping around the compiled arithmetic, then publish the T4
continuation into the shared T3/T5 state library. Join actual sensor and
light subpaths before claiming a full exposure image.

## Prompt History

User: “Can you show me an image and also progress towards total integration
of the ray tracing systems through nodus kernel work schedules/contracts”

User-provided `C:/dev/Powershell/AGENTS.md`: “Never ‘find a better way.’ If
the task names a mechanism, use that mechanism as it is spelled.”
