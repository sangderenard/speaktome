# Reversible binary read head and machine multicore

**Date:** 2026-08-04

## Scope

Work occurred in the adjacent `turing` repository. The first interpretation
incorrectly approached differentiable `FusedProgram` reversal. The user
clarified that the subject was the `X86TensorReadHead` in the binary-ingestion
and decompilation pipeline, and that the intended endpoint is a real executor
for a fully recompiled binary subject.

## Work completed

- Added exact immutable-state journaling around the tensor x86 read head.
- Preserved its existing vectorization: every batch lane is an independently
  advancing virtual read-head core, updated by one tensor transition.
- Added backward traversal, history seeking, acknowledgement journaling, and
  fork-after-rewind behavior without pretending the lossy decoder transition
  has an algebraic inverse.
- Exposed all twenty mutable decoder registers through a stable
  core-by-register `AbstractTensor` observation ABI.
- Added a WebGPU compute artifact for directly updating the read-head register
  display from that packed state.
- Extended the actual `MachineExecutionOrchestrator` with reversible execution
  journals and a barrier-stepped virtual multicore. Each core versions PC, all
  sixteen general registers, flags, memory, call stack, and step count.
- Exposed the complete architectural register file and a lossless low/high-u32
  packing ABI for a WebGPU display shader.
- Kept missing machine effects fail-closed. The work does not claim the present
  semantic vocabulary is already a complete executor for an arbitrary subject.
- Added a physical-style chip ABI: every core owns a fixed 256-byte bank whose
  twenty architectural/observation registers are individually contiguous
  64-bit cells, and each decoded function owns a fixed cache-line-aligned
  program block with shader-visible occupancy.
- Added separate register-bank and program-cache WebGPU update kernels, plus a
  bounded wall-clock governor that drops excess catch-up time and caps total
  cycles to prevent self-hosted time-dilation runaway.

## Verification

The new read-head and machine-execution tests passed together (8 tests). The
surrounding machine lifting and machine-to-Turing graph suites also passed:

```text
18 passed in 19.56s
```

The subsequent chip-layout pass brought the combined focused count to:

```text
23 passed in 15.33s
```

## Next steps

- Join emitted tensor read-head records to compiled semantic-token execution
  events so decode and execution are one binary-head runtime surface.
- Compile the chosen subject through the existing lift/rewrite pipeline into
  the repository-owned binary ISA, filling semantic handlers until execution
  is complete rather than blocked.
- Bind the emitted WebGPU register shader into the live presentation runtime
  and verify it in a WebGPU-capable browser.

## Prompt History

> I want you to get the binary read head ready to run forward and backward in execution or examination and with multiple concurrent threads of heads, and I want us to make a virtual multicore that, as I mentioned, is reversible, and thus differentiable in a graph sense a little deeper than usual, and we'll make all the registers show their contents and make sure it uses a shader to update as lightning fast as it runs

> that ambiguity was my fault, you stepped into other unrelated work. that's about programmatic reversibility, me and you are talking about binary read head reversibility. the binary head is an object in the decompiling pipeline

> right, for this process, we're going to fully compile to binary our subject, it's going to really be an actual executor

> can we please ensure the registers are all individually contiguous memory for conceptual fun, like, they're on the chip right? we used fixed allocations for our programs right now, that could be a cache we show, blocks illuminated by occupancy, it would be a fun little game of watching the computer compute itself if you let it run itself, but that might run away from itself in time dilation
