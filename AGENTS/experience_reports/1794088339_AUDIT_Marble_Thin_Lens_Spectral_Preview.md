# Spectral thin-lens marble preview audit

**Date:** 2026-11-07

## Scope and finding

Inspected `C:\dev\Powershell\spectral-analyzer` in isolated worktree
`C:\dev\Powershell\spectral-marble-trace-20261107`. The detailed
implementation handoff is `MARBLE_PROGRESSIVE_RAY_PREVIEW_HANDOFF_2026-11-07.md`
there. The initial audit commit was `983bd7c`; the validated renderer,
saved-mesh adapter and worker are now integrated into canonical Spectral at
`76749ae` and `19ca18e`. The Meowdoku HTTP bridge and photo panel are in
canonical Turing main at `a48590de` or later.

Spectral already has a genuine thin-lens backward sensor shader and native
camera/BDPT transport. The cheap C++ forward image call is light-origin and
cannot show checkerboard distortion through glass. A 16×16 native BDPT probe
did not complete in two bounded 20-second attempts, including one with the
demo's batching lifecycle. The existing GLSL `SensorAccumulator` proved the
viable bounded path: a real RTX 3060 OpenGL 4.3 worker now uses its analytical
finite-aperture focus and continuous exposure.

The first preview should use the existing one-line fixed 587.5618 nm
transport contract for a black-and-white thin-lens still. The exported
Meowdoku PBR glass material is not native optical glass: its bright albedo
becomes high spectral reflection under `MaterialDatabase.build_mat_buf()`.
The saved-mesh adapter derives the ray material from those PBR records via the
existing `MaterialDatabase` compatibility conversion and registers an optical
one-line glass band at the edited IOR. A controlled 256×256 checker comparison
shows visible shifted/inverted squares through three BK7 spheres while the
same opaque geometry remains dark. A 97,280-triangle final Studio cat-eye
mesh, including three closed dyed ribbons and its shell, was photographed
with the same worker. A temporary saved-object HTTP upload, status poll, and
image GET completed in 2 seconds of exposure after about 4 seconds of scene
preparation, with 1,761,280 forward ray jobs and 3,522,560 sensor sites.
The resulting image is at
`C:\dev\Powershell\spectral-marble-trace-20261107\build\marble_ray_probe\http_actual_saved_marble.png`.

Meowdoku's MRP1 upload includes the final current triangle positions,
per-vertex normals and material IDs. The server verifies the exact saved
geometry SHA-256, reads that saved object's PBR16 table, and launches one
digest-keyed Spectral subprocess. It publishes periodic PNG/status files,
cancels a stale worker on revision change, and caps photo exposure at 600
seconds. The browser offers explicit 30/180/600-second choices beside the
interactive Phong view. No eager Wasm compile is needed.

## Next Steps

Run the live or dev Studio page through one saved design, choosing an explicit
photo exposure, and verify the UI status, digest cancellation and the visible
photo panel. The local temporary HTTP/GPU path was proven; the live site
deployment is a separate gate. The first still is monochrome, uses flat
geometric face normals in the Spectral sensor, and proves refraction through
the shell. It does not yet prove physical caustics, full stage-light parity,
color dispersion or smooth per-vertex optical normals. Investigate native
T5 drain separately using a process timeout and pipeline telemetry.

## Prompt History

> "there is a ray tracer, it's spectral high everything bidirectional form takes an eternity, but it has so many ways to be run, and many cheap ones could give us the distortions and caustics we want from the marble designer, if we let the server in the background make still images over time gathering rays only as long as the settings don't change, overlaying the area in the phong scene with their data as the data comes in"

> "it's in spectral analyzer and there are a lot of parts, it's extremely powerful, we'll be looking to just use enough of the parts to make enough of a ray trace to give a fast preview of the marble as solid material"

> "User wants marble designer progressive ray-traced still overlay using existing powerful ray tracer in C:\dev\Powershell\spectral-analyzer, cheap path for solid material distortion and caustics; server background gathers rays only while settings unchanged, overlay progresses over Phong scene. Inspect spectral-analyzer AGENTS/guidance, recent guestbook/handoffs, ray-tracing implementations and interfaces, and Meowdoku server/design integration. Do NOT invent separate tracer or stand-in. Identify cheapest sanctioned existing path and implement a coherent minimal end-to-end prototype in an isolated worktree/branch if feasible, coordinating with marble_game agent; no long heavyweight build without approval. Preserve live and main. Report precise files/commands/limits. No Wasm build. Ask one narrow question only if truly blocked."

> "User explicitly says black-and-white photo is acceptable as first ray-traced preview, separate from/reveal potential next to Phong (not replace it). This may use sanctioned one-line fixed spectral contract with true thin lens, refraction, checker background, and capped progressive exposure. Preserve dice bevel/face relief in mesh payload/digest. Please prioritize workable monochrome still if 3-band is too slow; don't claim caustics until visible."
