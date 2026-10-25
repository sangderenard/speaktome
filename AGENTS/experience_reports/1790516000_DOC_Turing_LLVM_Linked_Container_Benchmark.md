# Turing LLVM linked-container benchmark

**Date:** 2026-09-27
**Title:** A small authored state container measures LLVM linkage inside and outside Python

## Scope

Added `turing/examples/llvm_linked_container_benchmark.py`, a self-contained
benchmark for a small evolving container with amount, temperature, foam, and
spin columns. The benchmark compiles one four-output symbolic law as an
`LLVMPiece`, composes an authored container around it through the sanctioned
source compiler, and statically links the emitted LLVM module into both a DLL
and Turing's existing standalone native host.

## Methodology

The four state updates are simultaneous and interact through feed, ambient
temperature, and dt controls. The benchmark checks 100 evolving steps against
the readable NumPy spelling before timing four execution lanes: NumPy, eager
LLVM-piece calls through Python, a prepared linked DLL called through ctypes,
and the standalone executable whose complete hot loop stays outside Python.

The linkage receipt comes from `NativePackage.linked_llvm`; the measured symbol
was `little_container_law__little_container_law`. Both native artifacts use
`-O2` and the repository's existing compiler and standalone-host paths.

## Results

The scalar call-overhead run used:

```text
py -3.11 examples/llvm_linked_container_benchmark.py \
  --batch 1 --steps 100000 --warmup 1000 \
  --build-dir build/llvm_linked_container_scalar
```

It reported exact numerical parity for all four state columns and:

```text
NumPy                   3.1125 s   31124.84 ns/cell-step    41.99x standalone
eager LLVM piece       11.5650 s  115649.82 ns/cell-step   156.01x standalone
linked LLVM (ctypes)    0.2379 s    2379.44 ns/cell-step     3.21x standalone
standalone link         0.0741 s     741.29 ns/cell-step     1.00x standalone
```

The standalone figure includes process launch plus its one-time input/output
file transfers, so it is conservative. The important observation is that the
same linked container becomes 3.21 times faster when the repeated call loop is
owned by the native host rather than crossing ctypes once per step.

For contrast, an 8,192-cell bulk run amortized the Python boundary and instead
exposed the linked wrapper's extra state-copy loops. That run is useful
telemetry, but it is not a call-overhead benchmark: NumPy measured 28.36
ns/cell-step, eager LLVM 30.66 ns/cell-step, and linked ctypes 97.08
ns/cell-step.

## Recommendations

- Use the standalone lane when measuring small repeated LLVM calls; a ctypes
  loop measures the Python boundary as part of every step.
- Keep the bulk-array lane available because it exposes kernel and state-copy
  structure that scalar call timing hides.
- Report the linked symbol and parity result beside timings so the benchmark
  proves that it exercised the intended artifact.

## Prompt History

> can you hand write a bespoke little container with some stuff going on inside it and benchmark the llvm as a link

> what the fuck would numpy be doing in the c i told you to write you stupid shit

> sorry your multi type benchmark was correct it included the thing i asked for and gave bonus telemetry

> it will only ever be fast outside of python the simple act of calling out to llvm is enormous
