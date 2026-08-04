# Turing affine piece solving and whole-program matmul replacement

**Date:** 2026-08-04

## Scope

Work occurred in the adjacent `turing` repository. The goal was to solve each
piece of a captured program as an isolated linear system, compose those systems,
and independently determine whether the complete observable program can be
replaced by one matrix multiplication.

## Work completed

- Added `src.compiler.affine_matmul_solver`, a runnable analyzer for
  `FusedProgram` IR.
- Reconstructs every operation as `y = A x + b` from zero and basis probes,
  then certifies it with deterministic held-out probes.
- Distinguishes variable feeds from captured fixed coefficients, so a fixed
  coefficient multiplication remains linear while a product of two selected
  variables is correctly rejected as bilinear.
- Lifts certified local systems into augmented state-transition matrices and
  composes them in execution order.
- Probes the complete program independently. This can certify an affine final
  result even when nonlinear internal pieces cancel, while retaining the local
  blockers as useful diagnostic boundaries.
- Added `MatmulReplacement`, including direct callable execution, the
  homogeneous `[A b; 0 1]` matrix, and materialization as a replacement
  `FusedProgram` using reshape, matmul, bias addition, and output reshape.
- Connected the analyzer to `ForwardReverseCycleCapture` through
  `analyze_matmul_replacement()`, using the cycle's solve-for parameters as the
  variable feed set.
- Added exact-affine and deliberately nonlinear CLI examples and documentation.

Certification is numerical and empirical over a finite deterministic probe set;
it is not a symbolic proof over all floating-point values or control paths.

## Verification

```text
python -m pytest tests/test_affine_matmul_solver.py \
  tests/test_forward_reverse_cycle.py tests/test_reverse_fused_program.py \
  tests/test_fused_program_python_backend.py \
  tests/test_ssa_fortran_and_optimizing_llvm.py -q
36 passed in 6.22s

python -m pytest tests/test_dream_document.py tests/test_wasm_html_shell.py -q
48 passed in 4.62s
```

Turing commits pushed to `codex/recursive-reduction-bridge` include:

- `7e186f5` Add affine matmul replacement analysis
- `07f66cf` Add sentinel dream document runtime
- `d278f80` Allow interior programs to own HTML presentation
- `10aa121` Wire dream display controller entrypoint
- `fa6e2e0` Emit launchable dream document shell
- `e97b714` Write dream shell to requested file

## Prompt History

> okay well, one thing I'd like to use this to do is to try to solve for each piece as a linear system in isolation then solve them together to see if a semi complex program can be fully replaced by matmul
