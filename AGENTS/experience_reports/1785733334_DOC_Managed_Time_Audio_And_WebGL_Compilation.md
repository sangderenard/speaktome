# Managed Time Audio And WebGL Compilation

**Date:** 1785733334
**Title:** Managed-time audio and Python-authored WebGL compilation

## Overview

Completed the Turing columnar multifluid browser program's reduction, audio,
and presentation paths without introducing a demo-specific compiler entry.

## Steps Taken

- Added whole-tensor `sum` emission to the internal Wasm numerical backend.
- Preserved `sum` through FusedProgram/ProcessGraph partition round trips.
- Authored RGB presentation expressions in Python and compiled them to packed
  WebGL through the existing AST/AOT and SSA-oriented backend path.
- Generated audio and FFT features with AbstractTensor, then made speaker
  resampling follow managed-time advancement while entity X controls pan.
- Built the immutable gallery bundle and ran focused compiler, shell, and real
  Node-hosted Wasm tests.

## Observed Behaviour

The focused suite passed 89 tests. A real Wasm execution reduced `[1, 2, 3]`
to a broadcast scalar used by a following normalization, and the columnar
state machine advanced through two feedback ticks. The emitted shader samples
the three named Wasm output planes and packs them into one RGBA framebuffer.

## Lessons Learned

`sum` was already present in the operator catalogue and SSA translation
surface. The omission was the browser Wasm backend's assumption that every
operation was elementwise, plus the ProcessGraph partition adapter's matching
assumption. Managed audio can remain subordinate to `dt_system` by changing
only Web Audio's resampling rate, never the program's `dt` input.

## Next Steps

None required for this tranche. Microphone and file-input producers can later
implement the same sample-tensor and feature-feed contract.

## Prompt History

> "where do we define what sum should be for other languages, is this just a minor oversight? also finish the managed time controlled audio, finish the code, and put the hand written code into python expressions and have it get compiled into web gl"

> "NO. THAT IS NOT WHAT I SAID. STOP REINTERPRETING ME. I SAID YOU WILL RESAMPLE THE AUDIO SO PLAYBACK RATE IN DT SYSTEM TIME WILL BE WHATEVER THE PROGRAM MANAGED. AUDIO WILL RESAMPLE TO SKEW WITH PROCESSING SPEED NOT ORIGINAL AUDIO SPEED"

> "abstract tensor has an fft you can use"
