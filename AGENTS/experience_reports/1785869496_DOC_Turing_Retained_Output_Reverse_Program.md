# Turing retained-output reverse program capture

**Date:** 2026-08-04

## Scope

Added an executable reverse projection for an unpruned Turing `FusedProgram`.
The projection uses AbstractTensor's canonical backward registry, exposes every
terminal forward value as a desired-output parameter, and separates proposed
original inputs from the incidental values required to justify the proposal.

## Work completed

- Added `retain_uncaptured_outputs`, which promotes terminal step results before
  output-reachability pruning while preserving declared output names.
- Added `capture_reverse_fused_program`, which constructs target-dependent
  residual operations and fuses them with a captured VJP so target values remain
  live replay inputs rather than frozen saved tensors.
- Added `ReverseProgramCapture.run`, returning proposed differentiable input
  values separately from original feeds and backward-saved incidentals.
- Documented that the operation is a local reconstruction proposal, not an
  algebraic inverse for arbitrary non-injective tensor operations.
- Added tests proving terminal retention, complete target parameterization,
  replay-time target replacement, input proposals, and incidental reporting.
- Added `/build/` to Turing's `.gitignore` and removed 173 generated build files
  from Git tracking while retaining the files locally.
- Preserved, tested, and committed concurrent reversible x86 read-head and
  register-shader source work found in the shared checkout, followed by its
  reversible machine-execution journal and multicore register display layer.

Focused verification:

```text
python -m pytest tests/test_reverse_fused_program.py \
  tests/test_backward_program_capture.py \
  tests/test_fused_program_python_backend.py \
  tests/test_recursive_reduction.py -q
24 passed in 14.50s

python -m pytest tests/test_x86_reversible_read_head.py -q
4 passed in 1.46s

python -m pytest tests/test_reversible_machine_execution.py \
  tests/test_x86_reversible_read_head.py \
  tests/test_machine_code_lifting_roundtrip.py \
  tests/test_machine_turing_graph.py -q
18 passed in 19.41s
```

Turing commits pushed to `codex/recursive-reduction-bridge`:

- `736db47` Stop tracking generated build artifacts
- `d9e3fee` Add retained-output reverse program capture
- `e690c22` Add reversible x86 read-head history
- `3e5e6fe` Add reversible machine execution journals

## Prompt History

> can you put in a function that will take our IR for a program and create a reverse version, that retains and thus splits before the pruning of uncaptured outputs, it would make all outputs parameter to the function - the thing that's... there's some part the path converges on and it escapes me but there's a perfect moment around fused program to just snatch the inverse, logical inverses take parameters that feed down into their predecessors, then we deliver two things, one, all the parameters that are proposed to have gone in, and then all the incidentals that must have also been true, then

> remember that abstract tensor can give you a backward function

> work freely and clean up all content as committed and pushed so you're free to work easy, no work is meant to be unclean

> that's my bad be sure to filter those artifacts through gitignore and untrack them or whatever we need to do when you finish your work, i didn't want logs or images
