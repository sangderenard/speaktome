# Stateful native convolution through ProcessGraph autograd

**Date:** 2026-08-15  
**Repo:** `turing/`  
**Status:** Proven by native execution; worktree remains uncommitted and shared

## Outcome

An ordinary two-layer `abstract_nn` convolutional image model now runs as a
stateful native training loop:

1. AbstractTensor forward and MSE loss;
2. the existing ProcessGraph analytical reverse generator and
   `BACKWARD_RULES` closure;
3. repository SSA;
4. authored LLVM tensor kernels;
5. native forward/loss/backward/SGD steps with persistent parameter buffers.

No tape-autograd lowering and no `FusedProgram` numerical bake participates in
this lane.

Run continuously from `turing/`:

```powershell
python -m examples.pattern_project.native_image_training --steps 0
```

Resume the compiled DLL and saved weights without rebuilding:

```powershell
python -m examples.pattern_project.native_image_training --stage run --steps 0
```

The latest image is written to
`.turing-cache/native-complex-image/snapshots/latest.png`; panels are target,
prediction, and doubled absolute error. State is in `state.npz`.

## The compiler efficiency defect

The first 55-node convolution motion spent more than an hour in repository-SSA
planning. This was not graph size, call-tree explosion, LLVM, convolution, or
memory growth.

A bounded `faulthandler`/`cProfile` sample proved that
`propagate_repository_ssa_call_metadata` was the hotspot. The module contained
only 59 functions and 907 instructions, but made more than 2.1 million
`enrich` calls in one minute.

Two defects compounded:

- whole-function value/constant/return/projection indices were rebuilt inside
  every call-edge visit of every fixed-point round;
- the second `authoritative_returns=True` pass remained authoritative forever.
  Distinct SSA aliases could overwrite each other's shape metadata every round,
  leaving `changed=True` indefinitely. Authoritative settling is now one phase,
  followed by monotonic fill-only propagation, matching the existing exact
  formal-settling contract.

After caching immutable topology indices and making authoritative return
settling one-shot, the identical repository-SSA stage completed in **0.879 s**
with zero shortfalls. Full capture + planning + LLVM emission took about 4 s.

## Convolution completion

- `AbstractTensor.unfold2d` / `fold2d` dispatch through backend hooks while the
  pure implementation remains the fallback.
- SSA and authored C/LLVM provide `unfold2d_double` and `fold2d_double`.
- `BACKWARD_RULES` keeps the existing analytical relationship: unfold's
  adjoint is fold and fold's adjoint is unfold.
- repository tensor lowering derives the twelve static NCHW/kernel/stride/
  padding/dilation operands and calls those authored kernels.
- singleton-batch rank-three matmul is reduced exactly through reshape and the
  existing rank-two matmul, avoiding the large generic indexing helper closure.

Native one-layer parity against an independent NumPy convolution:

- loss error: `0.0`;
- maximum weight-gradient error: `5.55e-17`;
- maximum bias-gradient error: `1.11e-16`.

## Reusable state and GAN boundary

`src/compiler/llvm_training_runtime.py` is a thin coordinator over the existing
compiler stages. `NativeParameterGroup` declares parameter IDs and a learning
rate. One complete forward/loss/backward motion receives one native step entry
per group. A test with groups named `generator` and `discriminator` proves that
invoking either entry changes only its declared parameter buffer. This is the
boundary for later alternating GAN losses/schedules; no GAN-specific optimizer
or compiler was added.

Training motions may also declare `observed_outputs`, allowing predictions or
other existing graph values to cross the native ABI for live reporting without
recomputing them in Python.

## Demonstrated image run

The two-layer RectConv2d/ReLU/RectConv2d/Sigmoid model resumed its saved state
and ran 1,000 further native steps:

- loss: `0.067334718762 -> 0.038956952544`;
- native throughput: approximately **1,394 steps/s** on this machine;
- snapshots and state were written throughout;
- four orphaned Python processes from earlier interrupted diagnostic runs were
  stopped by exact PID; no Python process remained afterward.

## Verification

- `tests/test_c_backend_llvm_ssa.py -k "not tape"`: **31 passed, 5 deselected**
- `tests/test_process_graph_autograd.py`: **20 passed**
- `tests/test_llvm_training_runtime.py`: **1 passed**
- focused unfold/fold LLVM reference + adjoint test: passed
- focused native RectConv ProcessGraph-adjoint parity test: passed

The five deselected tests are the explicitly out-of-lane direct tape-lowering
tests. The known unbounded `tests/test_ssa_llvm_backend.py` module was not run.

## Remaining boundary

This proves named independent parameter groups over one loss motion. A real GAN
still needs two declared loss motions and an alternating schedule selecting the
generator/discriminator entries. The interface supports that next coordination,
but no claim is made that adversarial training itself has been demonstrated.
