# Clocked reversible machine snapshot runtime

**Date:** 2026-08-04

## Overview

Work occurred in the adjacent `turing` repository. The emulator application
and its subject binary were separated correctly: cards/SSA compile the dream
application, while runtime subject bytes enter the existing PE machine
decompiler and remain in register-aware machine execution structures.

## Steps Taken

- Added a shell-regulated clock with mutable transitions-per-second and exact
  per-tick budgets; maximum-speed worker execution remains an optional mode.
- Added three preallocated `TMSNAP01` observation slots using a single-writer,
  single-reader ownership flip. A display may skip generations and cannot see
  a partially written register bank.
- Serialized every core's complete 256-byte-stride register bank, per-core
  execution/history status, and typed subject-output descriptors/payloads.
- Added `BinaryMachineRuntime.load_pe()`, which joins runtime bytes to the
  existing PE token decompiler, reversible executor, external clock, device
  buffers, and snapshot publisher without routing the subject through cards.
- Replaced the dream simulator's fake counter with real load/tick/speed Python
  controls and changed compute/display shaders to consume the snapshot ABI.
- Added an HTML liaison snapshot bus and binary-loader handoff. The interior
  WebGL2 controller uploads the newest register generation plus RGBA8 subject
  framebuffer; colored 1x1 placeholder textures were removed.
- Added tests for snapshot integrity, leased-slot safety, forward/backward
  running, externally changed speed, subject output, and an actual constructed
  AMD64 PE32+ image passing through the existing decompiler before execution.

## Observed Behaviour

- Focused machine/dream/shell suite: 62 passed.
- Snapshot-specific suite including the actual PE loader: 6 passed.
- Generated `build/reversible_chip_clocked.html` launched in headless
  Chrome/WebGL2 with `data-display-owner="chip-present-fragment"` and no shader
  error.
- The complete repository suite exceeded the 180-second command window without
  emitting a failure; it did not produce a completed result and is not claimed
  as passing.

## Lessons Learned

Presentation and execution clocks are separate contracts. The shell may use
its state-machine tick to regulate machine speed, but a shader only observes a
completed snapshot generation. Reversible history remains durable executor
state, whereas triple-buffer observations are deliberately disposable.

CPython supplies atomic object-reference publication for the present SPSC
implementation. Native C and WebAssembly hosts must reproduce the same slot
ownership protocol with their own atomic control words rather than treating
the Python GIL as part of the cross-language ABI.

## Next Steps

- Lower/package the Python binary runtime into the browser target so it installs
  `TuringMachineRuntime.loadBinary` and publishes live `TMSNAP01` buffers.
- Implement the corresponding C/Fortran shell adapter with native atomics.
- Expand instruction-effect, memory-image, import/syscall, and device semantics
  before claiming arbitrary binaries execute beyond the fail-closed boundary.

## Prompt History

> you do not need to make cards from the binary, cards don't use registers they aren't set up for ssa, that was my mistake to even mention

> what remains to be done

> the way this should work is it should free spin run the program as fast as it can, while the shader takes flips of the runner's state buffer which would include the register states

> I guess it should have a clock we can set the speed of where we can externally regulate it with a tick, our shells can handle tick state machines it's just going to run kinda like that

> please make this your goal and pursue it with diligence
