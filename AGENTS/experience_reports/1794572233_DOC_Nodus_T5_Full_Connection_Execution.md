# Nodus T5 full connection execution

**Date:** 2026-11-13 UTC
**Title:** Authored T5 shader writes a linear pixel through Nodus on three GPUs

## Overview

The earlier T5 Nodus proof compiled and validated SPIR-V, but did not invoke
the shader. This pass adopted its two-dimensional dispatch and exercised the
complete authored connection shader through Nodus's tensor state library,
GPU shader job row, and ordered pipeline row.

## Steps Taken

- Extended the Vulkan dispatch entry with a Y group count. Existing callers
  continue to use one Y group.
- Accepted an optional Y invocation field in the GPU shader job and kernel
  pipeline work tokens, leaving the legacy token shapes valid.
- Added `nodus/tests/spectral_t5_vulkan_test.cpp` using the existing authored
  Spectral SPIR-V artifact and its 11 reflected storage bindings.
- Built the Nodus test target in `C:/Users/alber/nodus-build/canvas-cmake` and
  ran it against `C:/Users/alber/nodus-build/spectral-t5/t5_full_connect.vulkan.spv`.

## Observed Behaviour

The fixture has 32 one-vertex sensor endpoints facing 32 one-vertex emitting
endpoints across a unit gap, one spectral band, and an empty BVH. Its 32×32
dispatch should yield 1,024 unit contributions. Vulkan devices 0, 1 and 2
each returned one completed pipeline stage, 1,024 considered pairs, 1,024
unoccluded contributions, and linear RGB `(1024,1024,1024)`. The test asserts
those values. The run log is
`C:/Users/alber/nodus-build/t5_nodus_full.log`.

This proves a full connection, PDF, MIS, spectral, shadow, reduction, and pixel
splat path on a controlled fixture. The camera and light vertex records were
constructed for the test; they did not come from T3/T4. The test does not prove
multi-bounce behavior, image quality, OpenGL parity, or an exposure service.

## Lessons Learned

T5's authored workgroup is two-dimensional (`16×16`), so a one-axis dispatch
cannot traverse its complete light-vertex population. The native row and
pipeline token need to preserve this axis explicitly.

## Next Steps

- Couple actual T3 and T4 vertex output to T5 in a Nodus pipeline and check
  linked path energy against the original shader host.
- Resolve the documented separated-arena T4 energy/polarization xfails before
  claiming a full bidirectional exposure.
- Exercise the pixel result through the image/job contract, including bounds,
  multiple bands, and camera settings.

## Prompt History

> Do you have T5, does T4 function? How close are we to moving all ray tracing inside nodus

> Please proceed
