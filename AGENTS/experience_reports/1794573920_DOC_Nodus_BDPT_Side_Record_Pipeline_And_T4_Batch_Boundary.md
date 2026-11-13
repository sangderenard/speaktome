# Nodus BDPT side-record pipeline and T4 batch boundary

**Date:** 2026-11-13 UTC

## Overview

The authored Spectral T5 sort-init, pack, scatter, and connection stages now
execute in order through one Nodus kernel-pipeline row on each of the three
Vulkan devices. This continues the earlier connector-only receipt.

## Steps Taken

- Adapted the original `bdpt_sort_init`, `bdpt_pack_t5`, and
  `bdpt_scatter_t5` GLSL through Nodus's strict uniform-to-storage-block
  relocation and reflection checks. The numerical shaders remain authored
  Spectral sources.
- Enabled BDPT record output in the Nodus T3 Vulkan test. An emissive hit
  produced one vertex, one spectral record, one PDF record, and one terminal
  on each device (`C:/Users/alber/nodus-build/t3_bdpt_records.log`).
- Ran sorted 32-light/32-sensor one-vertex fixture records through GPU
  sort-init → pack → spectral scatter → PDF scatter → T5 connect. The fixture
  deliberately sets fallback beta to 0.25 and omits PDF flags so both scatter
  stages must write for full energy to appear.
- Fixed a camera-side ID lookup failure in Spectral's scatter source by
  reconstituting the 31-bit packed ID from exact 16-bit pieces. The source
  fix and a comment-only sort-init change were committed and pushed as
  `0cb2fed` in `spectral-analyzer`.

## Observed Behaviour

The five-stage Nodus pipeline completed on Vulkan devices 0, 1 and 2. Each
reported 32 light vertices, 32 camera vertices, 1,024 accepted pairs, 1,024
valid connection PDFs, and exactly `(1024,1024,1024)` linear RGB in one pixel.
The test uses subpath IDs above 2²⁴ to rule out a lossy whole-ID float roundtrip.
Receipt: `C:/Users/alber/nodus-build/t5_nodus_sort_pack_connect.log`.

The records were supplied as a controlled fixture already in sorted order.
The bitonic compare/swap sort stage is not yet executed here. T3's output
record shape is verified separately; a live T3→sort→pack→T5 state coupling
has not yet been demonstrated.

T4 is an optional wave-region branch, not an obligatory step for ray-only
transport. Its existing `wave_t4::angular_spectrum_step_wide` already batches
cohort lanes for independent rays inside one arena, using fftfree. The arena
ABI has persistent state and per-slice progress. Cross-arena batching would
need to group compatible grid and step geometry while preserving each arena's
state and claims. `wave_t4.cpp` records two failed attempts to thread the
fftfree executor, including a reproducible race and a separate access
violation; do not re-enable that threading without addressing those failures.

## Next Steps

- Include authored bitonic sort steps in a Nodus pipeline for unsorted records.
- Couple T3-emitted side records and T1/T2 side streams into that pipeline.
- Form a native Nodus T4 row around the existing angular-spectrum contract,
  then measure compatible cross-arena batching without changing the solver.
- Preserve the wave continuation energy and polarization acceptance gates.

## Prompt History

> keep on keeping on, did you need any help? also I just added the speaktome mcp, see if you can use it or if I need a reboot

> whenever you can get to it, t4 should have it's wave algorithm checked and made into a kernel so, just like the rope simulation, we can run all wave simulations at once
