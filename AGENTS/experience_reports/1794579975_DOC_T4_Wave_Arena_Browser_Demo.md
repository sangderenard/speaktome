# T4 wave arena browser demo

**Date:** 2026-11-13 UTC

## Overview

Spectral Analyzer branch `codex/wave-arena-web-demo` adds a local browser page
for watching an evolving complex field. A thin Python HTTP host retains the
field buffers and calls the existing native `RayTracer.t4_angular_spectrum_step`
on each requested step. Its starting field and display maps come from
`wave_transform_visual_demo.py`; no second propagation solver was added.

## Observed behaviour

- Browser at `http://127.0.0.1:8788/` showed the power, relative phase, and
  change from initial field panels advancing with the kernel step count.
- Changing wavelength to 650 nm and applying reset returned the display to
  generation zero with the new parameter shown.
- `tests/test_wave_arena_web_demo.py` passed, including HTTP page delivery,
  native step, changed frame, and exact reset frame.
- The existing production-transform visual demo test passed (one selected test).

## Boundary

This page is an interactive host for the existing T4 transform with persistent
split-complex buffers. It does not yet operate the ray pipeline's full
`WaveArenaState` lifecycle, lens ports, absorber, or material aperture. The
Nodus compiled math kernels remain a separate review item, not the source of
the FFT propagation shown here.

## Next Steps

- Add a page mode driven by the full native pipeline arena lifecycle and
  expose its per-slice progress and energy ledger.
- Route the accepted Turing wave products through a production Nodus T4 row
  around Spectral's existing fftfree and arena state.

## Prompt History

> Prioritize using pure math compiling to make a premium set of wave kernels for different purposes. Trust that the pure math will make a pure kernel

> Can you make a demo page that let's the user set parameters and watch an arena evolve
