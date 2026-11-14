# Thick lens wave context and specialized 3D kernel

**Date:** 2026-11-13 UTC

## Overview

The current native thick-lens pipeline installs the authored T4 ScaleContext
but does not enter it when the region lies inside the exact T2 compound lens.
In parallel, a separate scalar 3D time-domain wave stencil was compiled from
SymPy and executed through Nodus GPU rows. These are separate results.

## Steps Taken

- Built the canonical four-group camera with `build_tracer`, placed a physical
  circular-aperture wave arena at the manifest stop, and submitted a center ray.
- Added a strict expected-failure test and a boundary report in the independent
  `spectral-analyzer` checkout, commit `d998bce3ac0c4640a26cf2bf11eb0eedcb91d82a`.
- Authored a centered 3D scalar wave law with six axial neighbors and two time
  levels in Turing, compiled native f64 and GPU f32 products, commit
  `d3eb1a17accabeb444c5e2eb724bb274d688c139`.
- Added a 64³ native Nodus GPU-row qualification, commit
  `fcee90291ca45db85714bb65a2b5cfe4247d515f`.

## Observed Behaviour

The exact T2 lens moved the ray from an initial X=-0.179 m hit to the next
segment start X=-0.04637295 m. The aperture ScaleContext was centered at
X=-0.143031506 m, inside that skipped interval. T4 processed zero rays and
the arena generation stayed zero. The Spectral test reports `1 xfailed`.

The Turing 3D law's 5×5×5 impulse test passed all five focused tests, including
validated SPIR-V. Nodus evaluated a 64³ impulse on devices 0/1/2 and verified
center 0.625, each of six neighbors 0.0625, zero elsewhere. The second process
run's host dispatch receipts were 1.3472, 2.472, and 1.5197 ms, respectively.
The first run included a 608.952 ms device-0 cache/initialization cost.
Neither receipt measures sustained throughput. The 3D product takes prepared
neighbor buffers; persistent volume state and boundary exchange remain open.

## Lessons Learned

The authored ScaleContext and wave material are real native machinery, but
graph installation does not prove transport reaches a region inside a fused
exact-lens traversal. The 3D time-domain stencil is a different specialization
from the 2D transverse angular-spectrum optical T4 arena, so their timings
cannot be compared as equivalent work.

## Next Steps

See `todo/1794583458_thick_lens_wave_handoff.stub.md` for the exact T2/T4
aperture handoff, then connect restored rays and BDPT side records to T5.
Build a persistent 3D volume row with indexed neighbor reads and an authored
boundary policy before calling the 3D product a complete simulator.

## Prompt History

> "okay well keep it in mind to get it onlien and then continue work trying to integrate the full 1-5 thick lens bdpt pipeline now with an aperture wave volume"

> "you may see code for scale contexts, i think you can intuit what that means, there is a system for drawing the region to be sensitive to transport"

> "I don't know how 3d this one is that you m ade but it should go without saying this requires a 3d kernel too for certain other basic wave simulator needs. I say a 3d kernel because it's probably goin gto make them faster if they are two separate specializde compiles"
