# Turing Fortran C Shell and Dependency-Free Native Display

**Date:** 2026-08-03
**Title:** Whole-program Fortran execution in a profiled C shell with a Win32 RGB blit

## Overview

The adjacent `turing` repository can now package a generated `bind(C)`
Fortran module as a standalone native executable. The generic C shell owns
input/output allocation, calls the compiler-selected whole-program entry,
applies declared state feedback, records launch timing, and writes the final
output planes. When the compiled API carries the shared `shell_io` display
contract, the same generator produces a Windows GUI executable and presents
three declared float64 RGB planes without SDL, pygame, OpenGL, or another
redistributable display dependency.

This work also completed the public distinction between `one_shot` baking and
`whole_program` baking, propagated the public ASAP/ALAP schedule preference,
and published a whole-program ASAP browser bundle. The native artifact is
separate from that site bundle and runs the same managed columnar multifluid
tick as generated Fortran.

## Steps Taken

- Added `src/compiler/fortran_c_shell.py` as an application-blind wrapper for
  emitted Fortran APIs.
- Reused the existing profiled C launch closure rather than creating another
  timing protocol.
- Preserved ABI parameter order, caller-owned allocation, dynamic extent
  specialization, state feedback, output checksums, and binary output files.
- Found and fixed an API-generation defect: a second-pass Fortran control
  wrapper can acquire a transitive callee extent, but the emitted API sidecar
  had described the discarded first-pass signature. The descriptor now uses
  each final emitted subroutine's extent list.
- Attached `display_double_buffer` through the existing `ShellIOManifest`.
  The current native display format is `rgb_f64_planar`, with resolved
  `display.red`, `display.green`, and `display.blue` output bindings.
- Added a Win32 display adapter generated into the C shell. It converts the
  planes to a top-down 32-bit DIB, pumps messages, and calls `StretchDIBits`.
  Display builds link as Windows GUI subsystem executables.
- Rebuilt the managed columnar multifluid executable at 256 by 176 pixels and
  verified continuous execution, finite-frame execution, state feedback,
  Unicode window titles, and clean window closure.
- Published the corresponding whole-program ASAP page version under the
  root `nogodsnomasters` site tree.

## Observed Behaviour

- The native executable depends only on `KERNEL32.dll`, `USER32.dll`,
  `GDI32.dll`, and `msvcrt.dll`.
- A two-frame native run advanced `next_time` from `0.025` to `0.05`, proving
  state feedback occurs between generations rather than repainting a frozen
  frame.
- All 30 one-frame output arrays matched direct execution of the authored
  Python function over 45,056 elements. Maximum absolute error was
  `2.842170943040401e-14` at `rtol=atol=1e-12`.
- The combined Fortran, control, SSA, shell, and ProcessGraph test selection
  passed 34 tests.
- A raw physical framebuffer is not a stable facility for an ordinary Windows
  process. Win32 GDI is the smallest dependable presentation boundary on the
  current target; conversion and pixel ownership remain in the generated C
  shell.

## Lessons Learned

- Calling convention sidecars must be derived from the final emitted
  procedure, not an exploratory signature pass. A missing value extent does
  not necessarily crash; it can shift every pointer argument and produce
  plausible-looking zero output.
- The existing `ShellIOManifest` is the correct common boundary. Display code
  should consume resolved resource bindings and must not know the application
  or infer channels from output names.
- The native display can remain dependency-free while preserving the same
  separation used by Wasm: compiled code publishes buffers, and the host shell
  owns presentation and asynchronous operating-system interaction.
- The current C implementation realizes display only. Merely having keyboard,
  pointer, and file schemas does not mean those native mailboxes are live.

## Next Steps

- Implement the native input-event ring for keyboard and pointer messages.
- Implement the asynchronous native file request/completion broker.
- Add packed `rgb8`, `rgba8`, and scalar/indexed display adapters beside the
  current planar float64 format.
- Move native artifact construction behind a stable command or bundle role so
  reproducing the executable does not require a one-off build driver.
- Consider a presentation thread only after the shared double-buffer ownership
  and generation counters are carried directly into the native adapter.

These items are recorded in
`todo/1785765928_turing_native_c_shell_io_handoff.stub.md`.

## Prompt History

> "let's put a flag in the compiler for one shot bake, which will render all logic and execute as a numeric block, like it used to before we started thinking about user control, or else with a different flag it does what we're trying to do lately and bakes the whole program with logic, and then could you bake a page the usual way for a version that uses asap (if that's been configured yet, the asap/alap flag, don't put it in if not someone is working on it) and for full program not numeric reduction"

> "can you make this in fortran in a c shell for my own pleasure apart from the site"

> "the program output stext, is this because we have no display adapter for the c shell"

> "take a look at how the web assembly pipeline works, how there's a lot of io built in, we need to get there in the c, using at worst the most basic library we can possibly imagine that's really sure to be there, if we can't do what I'd prefer, which would be blit rgb ourselves, no window manager"

> "prepare experience report and handoff document in the speaktome ecosystem and commit and push all your work in every repo you touched"

