# Turing forward/reverse parameter solver and Fortran cycle

**Date:** 2026-08-04

## Scope

Added a Python-runnable optimization cycle that captures an unpruned
AbstractTensor forward graph, derives its retained-output reverse graph, exposes
target and correction strategies, fuses both directions, and emits/executes the
same cycle through the native Fortran backend.

## Work completed

- Added `capture_forward_reverse_cycle` and `fuse_forward_reverse_program`.
- Added fixed and interpolated target hooks; all retained outputs remain live
  desired-value feeds in Python and native execution.
- Added scheduled gradient, clipped, and arbitrary callable correction hooks.
  Host-only corrections are rejected by native emission instead of being
  silently omitted.
- Added `ForwardReverseSolver` for iterative recapture and correction.
- Added `FortranCycleArtifact` and `FortranCycleExecutable`; native cycles feed
  proposed parameters into the next invocation and accept per-cycle target
  replacement hooks.
- Completed the FusedProgram-to-SSA-to-Fortran bridge for reverse-mode
  `reshape`, `broadcast_to`, keep-dimension reductions, and scalar/array shape
  transitions.
- Added the runnable module and documentation. Generated `.f90`, ABI, and DLL
  artifacts remain under ignored `build/`.

Run:

```text
python -m src.common.tensors.abstract_nn.forward_reverse_cycle \
  --iterations 12 \
  --emit-fortran build/forward_reverse_cycle \
  --compile-fortran
```

Verification:

```text
python -m pytest tests/test_forward_reverse_cycle.py \
  tests/test_reverse_fused_program.py \
  tests/test_backward_program_capture.py \
  tests/test_ssa_fortran_and_optimizing_llvm.py \
  tests/test_machine_target_languages.py -q
34 passed in 6.16s

python -m pytest tests/test_shader_component_abi.py \
  tests/test_machine_chip_layout.py tests/test_webgpu_ssa_backend.py \
  tests/test_ssa_webgl_source_roundtrip.py \
  tests/test_x86_reversible_read_head.py \
  tests/test_reversible_machine_execution.py -q
29 passed in 5.09s
```

Turing commits pushed to `codex/recursive-reduction-bridge`:

- `f0f9f45` Add fused forward reverse solver cycles
- `03143e5` Add cross-shader component ABI
- `e153315` Add fixed reversible machine chip layout

## Prompt History

> can you please make a python runnable that obtains the forward graph and the reverse with hooks that offer different targeting and correction strategies, intuitively this tells me there's a way to solve for parameters arbitrarily with a cycle like that and we can fuse the entire thing in a fortran runnable for both forward and back cycling
