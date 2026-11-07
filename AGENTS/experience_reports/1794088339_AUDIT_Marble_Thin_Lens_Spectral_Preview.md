# Spectral thin-lens marble preview audit

**Date:** 2026-11-07

## Scope and finding

Inspected `C:\dev\Powershell\spectral-analyzer` and created isolated worktree
`C:\dev\Powershell\spectral-marble-trace-20261107`, commit `983bd7c`.
The detailed implementation handoff is
`MARBLE_PROGRESSIVE_RAY_PREVIEW_HANDOFF_2026-11-07.md` there. No live site,
canonical Turing, or Spectral main files were changed.

Spectral already has a genuine thin-lens backward sensor shader and native
camera/BDPT transport. The cheap C++ forward image call is light-origin and
cannot show checkerboard distortion through glass. A 16×16 native BDPT probe
did not complete in two bounded 20-second attempts, including one with the
demo's batching lifecycle. This blocks an honest progressive overlay today.
The existing GLSL `SensorAccumulator` is the more promising bounded path:
`pump_forward(budget_ms)` plus `tick()` with `rows_per_frame`, analytical
finite-aperture focus, and continuous exposure.

The first preview should use the existing one-line fixed 587.5618 nm
transport contract for a black-and-white thin-lens still. The exported
Meowdoku PBR glass material is not native optical glass: its bright albedo
becomes high spectral reflection under `MaterialDatabase.build_mat_buf()`.
Use Spectral's authored low-scatter glass optical response and verify the
reference-line IOR before claiming refraction. The browser must send the
final current triangle mesh/material IDs plus camera, lens, lights, and room
state under a digest; the server must cancel stale work and publish only
matching revisions. No ray-trace overlay was implemented from the failed
probe.

## Next Steps

Run the known-good Spectral GLSL thin-lens sensor on a small checker/glass
scene with bounded forward and backward work; verify visible distortion and
readback. Then add the current marble mesh/material scene bridge, background
job lifecycle, and digest-gated Phong overlay. Investigate native T5 drain
separately using a process timeout and pipeline telemetry.

## Prompt History

> "there is a ray tracer, it's spectral high everything bidirectional form takes an eternity, but it has so many ways to be run, and many cheap ones could give us the distortions and caustics we want from the marble designer, if we let the server in the background make still images over time gathering rays only as long as the settings don't change, overlaying the area in the phong scene with their data as the data comes in"

> "it's in spectral analyzer and there are a lot of parts, it's extremely powerful, we'll be looking to just use enough of the parts to make enough of a ray trace to give a fast preview of the marble as solid material"

> "User wants marble designer progressive ray-traced still overlay using existing powerful ray tracer in C:\dev\Powershell\spectral-analyzer, cheap path for solid material distortion and caustics; server background gathers rays only while settings unchanged, overlay progresses over Phong scene. Inspect spectral-analyzer AGENTS/guidance, recent guestbook/handoffs, ray-tracing implementations and interfaces, and Meowdoku server/design integration. Do NOT invent separate tracer or stand-in. Identify cheapest sanctioned existing path and implement a coherent minimal end-to-end prototype in an isolated worktree/branch if feasible, coordinating with marble_game agent; no long heavyweight build without approval. Preserve live and main. Report precise files/commands/limits. No Wasm build. Ask one narrow question only if truly blocked."

> "User explicitly says black-and-white photo is acceptable as first ray-traced preview, separate from/reveal potential next to Phong (not replace it). This may use sanctioned one-line fixed spectral contract with true thin lens, refraction, checker background, and capped progressive exposure. Preserve dice bevel/face relief in mesh payload/digest. Please prioritize workable monochrome still if 3-band is too slow; don't claim caustics until visible."
