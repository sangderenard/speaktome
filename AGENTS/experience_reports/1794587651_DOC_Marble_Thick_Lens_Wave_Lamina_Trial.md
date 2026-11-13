# Marble thick-lens wave lamina trial

**Date:** 2026-11-13 UTC

## Overview

The Spectral review branch `codex/thick-lens-wave-traversal` now has a
focused exact compound-lens T2 → local T4 exterior wave-patch trial at a
10 mm marble. Per-lane T4 bulk extinction and a Studio region-medium to
complex-index mapping were added. A normal dielectric field pair supplies
coherent reflected and transmitted products for a bubble interface.

## Steps Taken

- Read the preceding Nodus T3/T4/T5 and marble region reports and the
  authored region, optical-media, exact-lens and T4 source contracts.
- Rebuilt `_spectral_kernels` from `csrc` with the existing fftfree Eigen
  dependency at `C:/dev/Powershell/fftfree/build/_deps/eigen-src`.
- Ran the focused lens, wave, authored medium and Jones tests.

## Observed Behaviour

All 94 focused tests passed. The thick-lens trial routes a sensor-side ray
through the exact T2 conics and a local 0.1 mm axial wave patch ending at the
sphere surface. Native power attenuation matched the analytical
Beer–Lambert factor in one-lane and three-lane trials. The saved host/rod/dye/
bubble mapping used the existing `optical_extinction` source law. The bubble
Jones pair conserved reflected plus transmitted power in three lanes.

This is not a wave material bounce, structural-color image, or Studio photo.
T3 does not yet instantiate a wave lamina from a material hit. Bubble
interference requires coherent fan-in, which the current graph refuses.
The full marble cannot be one 256-cell wave arena without undersampling.
Native T4 now reports material attenuation separately from numerical border
loss; reflection, transmission and observed-aperture loss remain distinct.
See `C:/dev/spectral-wave-traversal/docs/MARBLE_WAVE_LAMINA_TRIAL_2026-11-13.md`.

## Next Steps

Use the T3 hit, exact or authored region transition, and normal to create a
local wave lamina. Route its reflected and transmitted field products through
T4 and back to T1/T5. Complete coherent fan-in and account for reflected and
transmitted power before claiming structural color or structural absorption.

## Prompt History

User: “Can you try using the thick lens pipeline and apply a wave context to the marble+.1 mm and pioneer the material responses to wave level analysis”

User: “We want to be able to identify minimal wave boundary through all interfaces within the subject, glass rod style material boundary or dye style dye effects from material properties extending in complexity”

User: “Including the bubbles, structural color would be really good”

User: “And structural absorption”

User: “We could make it a bounce step to do the wave lamina over the mesh”
