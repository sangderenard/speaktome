# Turing native Fortran affine learner and visualizer

**Date:** 2026-08-04

## Scope

Work occurred in the adjacent `turing` repository. The request extended the
affine replacement analyzer into a native experiment that can be pointed at a
Python-described exact algorithm, learn a cheaper affine approximation over
time, and visualize the fitting and pruning process without a Python or Pygame
runtime.

## Work completed

- Added `src.compiler.native_affine_learner` with the callable
  `compile_learning_visualizer(python_file, output_directory, ...)`.
- Defined a small trusted build-time Python protocol that returns training and
  held-out exact input/output pairs plus a transparent reference-operation
  estimate.
- Generated a standalone Fortran 2008 executable containing the baked dataset,
  gradient descent, L1 shrinkage, progressive hard pruning, held-out exactness
  checks, a cost-aware candidate selector, ANSI terminal visualization, and
  model export.
- Kept correctness and cheapness as separate visible facts. Crossing the cost
  budget never causes the candidate to be declared exact.
- Added an eight-value sorting benchmark. Sorting is ordinary and exactly
  verifiable, while a fixed affine transform cannot represent global ordering;
  it is therefore an honest long-running approximation problem.
- Added a runnable CLI and documented the Python callable/file contract.
- Kept emitted `.f90`, `.exe`, and model files under the existing ignored
  `build/` tree.

## Observed native run

An 80-epoch smoke run reduced held-out MSE while the progressive budget pruned
the candidate from 72 operations to 23, below the declared 24-operation sorting
reference. Exact validation remained `0/48`, correctly showing that the cheaper
surrogate had not replaced the algorithm exactly.

## Verification

```text
python -m pytest tests/test_native_affine_learner.py \
  tests/test_affine_matmul_solver.py tests/test_forward_reverse_cycle.py \
  tests/test_reverse_fused_program.py \
  tests/test_ssa_fortran_and_optimizing_llvm.py tests/test_fortran_fidelity.py -q
33 passed, 1 warning in 13.89s
```

Turing commit pushed to `codex/recursive-reduction-bridge`:

- `576c5e0` Add native Fortran affine learning visualizer

## Prompt History

> can you give me a python callable using pygame that can run a visual representation of this process, running a compiled binary of the program you just made that tries to solve a system for a cheaper system, let's make it an - hell forget making the pygame python outer piece compile the whole thing in fortran and we'll point it at python files and we'll make a file that has a common algorithm, something maybe learnable but hard, something that will learn over time but might never perfect and that's a common verifiable algorithm
