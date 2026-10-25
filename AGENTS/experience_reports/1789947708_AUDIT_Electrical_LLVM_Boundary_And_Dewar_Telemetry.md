# Electrical LLVM boundary and dewar telemetry audit

**Date:** 2026-09-20
**Title:** Compiler boundary, thermal telemetry, dt labels, and viewer input

## Overview

Audited the live dewar scene's electrical path against the sanctioned Turing
source compiler and the existing Spectral Analyzer circuit implementation. The
runtime circuit is a real complex nodal graph, but its numerical solve currently
ends in Torch. The existing circuit bake boundary already exposes the exact
unnormalized nodal law needed by an LLVM kernel. Also removed misleading live
telemetry and fixed a Pygame key event crash.

## Steps Taken

- Read the live scene registration and publication paths in
  `turing/examples/chamber_raincloud_live.py`.
- Read `ComplexTensorCircuit.nodal_form()` and the tier-1 GraphSolver solve in
  `spectral-analyzer`.
- Measured the instantiated live circuit as 4 lanes, 51 nodes, 68 branches,
  and 134 injections.
- Compared the actual numerical boundary with the existing planar
  real/imaginary `Precision[2]` compiler coverage.
- Removed the unpublished `compressor_temperature_delta_k` fallback that
  displayed NaN under the name `thermal-atlas delta`.
- Removed the hand-authored `dewar-systems`, `participants`, and missing-step
  display metadata. The live profile now describes the actual dt schedule.
- Replaced weapon selection through `int(ev.unicode)` with the authoritative
  Pygame number-key code, because a valid KEYDOWN event may have empty Unicode.

## Findings

- Complex arithmetic is no longer the primary LLVM blocker. Planar real and
  imaginary `Precision[2]` columns have compiled multiplication and division
  coverage through `lower_ast_source_to_ssa`.
- `GraphSolver` itself is not an appropriate compiler entry. It owns dynamic
  dictionaries, string node identities, topology/SCC scheduling, Torch module
  state, autograd buffers, timers, and delayed state.
- `ComplexTensorCircuit.nodal_form()` is the correct existing bake seam. It
  returns the exact complex nodal admittance matrix and source offset before
  GraphSolver rearranges the law for iterative execution.
- The missing compiler artifact is a fixed-shape planar-complex nodal solve
  written on AbstractTensor/Precision and lowered through the sanctioned source
  entry. Its pivoting/order behavior must be proven through the LLVM source
  path; the eager/C `AbstractTensor.linalg.solve` coverage does not establish
  that contract.
- Temperature-dependent impedances and nonlinear tangents can update numeric
  admittance values without changing or recompiling the baked topology.
- The supplied warm profile and local cold runs all identify the tier-1 dense
  solve as the electrical step's dominant cost, despite different absolute
  timings. This makes the nodal solve the useful first LLVM target.
- `thermal-atlas delta` was not a NaN produced by the thermal physics. The live
  scene requested a key no engine published and supplied NaN as its fallback.
- `dewar-systems` and `participants` were presentation metadata rather than dt
  system concepts or measured state.

## Validation

- `python examples/chamber_raincloud_live.py --headless --steps 1 --log-every 1`
  completed one live dt round with finite compressor temperature and the real
  schedule labels.
- `python -m pytest tests/test_chamber_damage_ports.py -q`: 2 passed.
- `python -m pytest tests/test_electrical_tensor_network.py -q`: 10 passed in
  `spectral-analyzer`.
- Python compilation checks passed for the live and viewer modules.

## Next Steps

- Author one fixed-shape planar-complex `Precision[2]` nodal kernel using the
  existing `nodal_form()` arrays and the sanctioned source compiler entry.
- Establish pivoting and near-singular behavior against Torch on the actual
  four-lane dewar circuit before substituting the runtime solve.
- Keep graph discovery, empirical-law callbacks, battery state, and thermal
  publication at their current Python bake/update boundary.

## Prompt History

> "can we get the electrical system into llvm or what are the blockers? also, thermal atlas delta is nan, there's a \"dewer systems\" and \"participants\" on the dt root and neither of those are things afaik"

> "sorry I meant this: [Pygame traceback ending in `ValueError: invalid literal for int() with base 10: ''`]"
