# Compiled wave kernels and Nodus row trial

**Date:** 2026-11-13 UTC
**Title:** SymPy T4 arithmetic through native, SPIR-V, and Nodus GPU rows

## Overview

The owner reprioritized the Nodus ray-tracer continuation toward pure-math
compiled wave kernels. Four arithmetic sites in Spectral T4 now have one
SymPy source in the Turing review branch `codex/wave-math-kernels`:
propagating angular-spectrum transfer, evanescent decay, aperture material
transmission, and 2×2 complex Jones mixing. This is a math-kernel set, not a
replacement for T4's FFT, arena state, or optical continuation.

## Steps Taken

- Read Spectral's `wave_t4.cpp` transfer, aperture, and Jones equations and
  `WAVE_CONTINUATION_ENERGY_AUDIT_2026-11-09.md` before authoring laws.
- Used Turing's `compile_sympy_equations` → AbstractTensor source →
  `piece_from_law` for f64 native batch pieces, and its deployable SPIR-V
  emitter/publisher for f32 GPU products.
- Added the three Pascal-case `Sin`/`Cos`/`Exp` opcode aliases required by
  Turing's own SymPy SSA route; f64 transcendental SPIR-V remains a declared
  shortfall, not an implicit conversion.
- Ran each compiled SPIR-V product through Nodus's existing
  `GpuShaderJobTool` and `TensorStateLibraryTool` on all three Vulkan devices
  for one 512×512 field plane. The companion Nodus branch is
  `codex/nodus-wave-kernel-row`.

## Observed Behaviour

`tests/test_spectral_wave_laws.py` passed eight gates: four f64 native
numerical comparisons and four f32 SPIR-V assembly/validation checks.
Each Nodus row run produced 262,144 checked samples on each of devices
0, 1, and 2. First-dispatch times in milliseconds were:

| Kernel | GPU 0 | GPU 1 | GPU 2 |
|---|---:|---:|---:|
| Propagating | 24.778 | 15.362 | 8.540 |
| Evanescent | 15.621 | 8.360 | 7.326 |
| Aperture material | 18.105 | 10.802 | 8.325 |
| Jones interface | 1.437 | 1.183 | 1.644 |

Those host-call times include initial setup and are not sustained GPU
throughput. The wider SPIR-V test file has an unrelated existing assertion
failure about axis-reduction shortfall wording; the focused wave tests pass.
Artifacts and logs are under `C:/Users/alber/nodus-build/wave-kernel-artifacts/`
and `C:/Users/alber/nodus-build/wave-<kind>-512.log`.

## Lessons Learned

Keeping the source's propagating and evanescent domains separate permits
faithful square-root/decay arithmetic without evaluating an invalid branch.
T4 computes its cached transfer table in f64; the current GPU products are
f32 because the existing SPIR-V backend correctly refuses f64 GLSL.std.450
trigonometric operations. That precision difference must be carried in the
kernel contract and compared in the full solver.

## Next Steps

- Wire these products around the existing fftfree forward/inverse stages and
  arena state in a production Nodus T4 row; retain Spectral's transfer-table
  cache, aperture mask and absorbed-power accounting, and Jones coordinate map.
- Resolve the known separated-arena energy and polarization xfails by routing
  branch handle and Jones continuation state through ordinary ray transport.
- Trial a real thick-lens aperture and measure warm, sustained multi-arena
  work before claiming T4 integration or ray-renderer completeness.

## Prompt History

> whenever you can get to it, t4 should have it's wave algorithm checked and made into a kernel so, just like the rope simulation, we can run all wave simulations at once

> we want to be able to run both ray tracers on many jobs at once through the kernel library, that's the present end goal. intermediate bonus round is getting the wave sim working and test it on a thick lens camera aperture

> Okay so you aren't even close to done please continue

> Prioritize using pure math compiling to make a premium set of wave kernels for different purposes. Trust that the pure math will make a pure kernel
