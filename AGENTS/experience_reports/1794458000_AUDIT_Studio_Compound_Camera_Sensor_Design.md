# Studio compound camera sensor design

**Date:** 2026-11-11

## Overview

Read the Studio lens selection, current GPU sensor launch, the exact compound
model, camera software transport graph, and the prior bounded thick-lens trial.
The user asked whether the demo runs, how slowly, and whether a compound camera
can approach the present thin-lens speed while using parametric optical regions.

## Steps Taken

- Reran the prior exact CPU BDPT driver under a fresh output path. It completed
  64×64 with eight pupil samples in 31.993 s, 5.982 s setup and 25.609 s active
  T5, 2,127 lit pixels, finite output, T5 fired. Diagnostic connection records
  truncated; upstream overflow counters stayed zero.
- Reran the existing camera software geometric light-table driver at 128×128,
  30,000 rays. It completed in 2.722 s with 3,865 lit pixels and finite output.
- Inspected the current Studio sensor GLSL. The analytical branch samples an
  aperture per ray. Its manifold branch reads only one direction per pixel;
  it does not represent compound pupil or spectral transfer.
- Wrote `spectral-analyzer/docs/STUDIO_COMPOUND_CAMERA_DESIGN_2026-11-11.md`
  with the proposed exact sensor-side transfer boundary and matched benchmark.
- Follow-up inspection separated native T5 from exact T2 conic traversal.
  Studio's sensor shader uses the forward field plus direct source NEE and has
  no T5 call. The native PLENS `14949` payload supports reverse traversal and
  per-band refractive indices. Its existing spectral Jacobian helper can guide
  adaptive transfer cells, though its geometric helper alone lacks T2's full
  amplitude/OPL behavior.

## Observed Behaviour

The geometric table image is recognizable but passes RGB chart data rather
than measuring spectral sensor physics. The exact native result remains
uncalibrated magenta. The present 128×128 Studio fixed:8 saved job took 5.062 s
for 327,680 sensor sites in a different scene and integrator, so no controlled
speed ratio or GPU compound-camera claim follows from these numbers.

## Next Steps

Build and validate a wavelength-aware backward sensor-side transfer from the
existing CompoundLens, then compare exact, cached compound, and current thin
camera launch in one scene with equal pixels, samples, bands, seed and state.
Measure camera transformation separately from scene tracing and transfer.

## Prompt History

> "I want you to go into the camera settings profiles and how they borrow from the thick lens demos, i want us to design compound cameras, two things i want to test, first one, can we run the thick lens demo and get results, how slowly, and can we by any means try to get something as fast as what we're using now but that also participates in parametric optics regions, can we use a thin lens system inside the aperture integral or sensor integral effectively enough knowing what we know now, that the thin lens demo is working very well"

> "It's very important to me to solve expedient thick lens profiling into transfer functions which may extensively use tables and interpolators, but it has to do real thick lens traversal, from the sound of it you didn't seem to imply that the problem was out paramedic conics, they work per wavelength they lower well, you said it was our bidirectional algorithm that's stage 5 isn't it? But we have the marble studio doing performant matching or a different strategy, so how do we make them work together, swap in the full sim but integrate it with the new method"
