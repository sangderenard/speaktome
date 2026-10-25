# Nested LLVM XOR and Outer Graph Demo

**Date:** 2026-08-14
**Title:** Coordinate a complete inner LLVM step with a graph-derived outer influence step

## Overview

Added a runnable demonstration at
`turing/examples/xor_project/nested_llvm_meta_learning.py`. It coordinates the
existing AbstractNN, ProcessGraph autograd, repository SSA, and LLVM facilities
without adding an autograd method or operator rule.

The inner native entry executes one complete XOR forward, MSE loss,
graph-generated backward, and SGD parameter update. The outer ProcessGraph
uses the saved inner gradients and pre-step parameter state to evaluate the
post-step XOR loss, generates its analytical backward with the existing
ProcessGraph autograd, and differentiates that loss with respect to the inner
learning rate. A host-side outer step updates that influencing parameter.

The runtime coordinator now supports repeated stateful cycles. By default,
the native inner step's updated `W1`, `b1`, `W2`, and `b2` buffers become the
next cycle's inputs, while the outer graph updates the scalar inner learning
rate. `--outer-steps 0` repeats until Ctrl+C; `--reset-inner-each-step` retains
the earlier counterfactual behavior for comparison.

## Result

Run from `C:\dev\Powershell\turing`:

```powershell
python -m examples.xor_project.nested_llvm_meta_learning
```

Observed output:

```text
inner LLVM loss:             0.281849904004
outer graph loss:            0.277837425529
d(outer loss)/d(inner lr):   -0.007727566387
outer step, inner lr:        0.500000000000 -> 0.501931891597
outer loss after outer step: 0.277822498938
outer saved bindings:        41
```

The default command builds inner and outer LLVM artifacts in separate
short-lived compiler processes and runs both in a lightweight coordinator.
Holding both compiler graph states in one Python interpreter caused the
process to be hard-terminated after successful native linking; separating the
stages removes that peak-memory overlap.

## Test Safety

The module-scoped `ssa_module` fixture in
`turing/tests/test_ssa_llvm_backend.py` is disabled at fixture entry. Its
whole-program XOR discovery is not execution-bounded and several timed-out
pytest shells left compiler children running. The small LLVM tests in that
module remain enabled. No further invocation of the disabled test was made.

Focused graph-contract verification:

```text
4 passed in 13.75s
```

Runtime-only state-carry verification over ten cycles reduced the XOR inner
loss from `0.281849904004` to `0.260074586779`; the tenth post-step outer loss
was `0.258933174642`. In reset mode, the reported inner loss remained exactly
`0.281849904004` on all three checked cycles, confirming that the observed
stateful progression comes from carrying the native updated weights.

## Prompt History

> "run the llvm interior forward, loss, backward, and step, because you ingested the whole training program. then evaluate something and , have the fgorward and bbackward process graph pair then run it's backward the graph derived ana lysitcally from unfolding a complete ordinary training process into a stream and then reversing it, then step whatever outer parameters we're going to be teaching to influence the inner llvm"

> "and you don't have to architect any of that. you have a few minor points of coordinating elegance needed"

> "can you make a demo in the design I want tha twe've worked out here"

> "disable that test it's compilation is not bound well and you start fifty of them and leave them"
