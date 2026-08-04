# Reversible AMD64 PE execution and guest system ports

**Date:** 2026-08-04

## Overview

Work occurred in the adjacent `turing` repository. The binary-machine program
advanced from decode-only orchestration to executing the real Windows AMD64
`cmd.exe` entry path with immutable architectural state. PE imports now become
named, capability-gated guest external requests instead of accidental jumps to
unresolved thunk RVAs.

## Steps Taken

- Added sparse immutable copy-on-write page memory with strict unmapped-access
  traps and little-endian typed reads/writes.
- Mapped PE headers and sections at the preferred image base and created a
  Windows-aligned 1 MiB guest stack plus minimal PEB/TEB pages.
- Exposed FS/GS bases as contiguous register-bank cells and applied segment
  override bytes during address calculation.
- Implemented partial GPR rules, address/move, arithmetic flags, compare/test,
  logical, shift/rotate, extend, stack, conditional, and atomic families.
- Made calls update both real `RSP`/memory and the reversible validation stack.
- Parsed PE import descriptors, names/ordinals, and IAT slots.
- Replaced IAT values with stable synthetic guest-external targets and added a
  `WAITING_EXTERNAL` state containing DLL/symbol identity, Windows register
  arguments, stack pointer, and return address.
- Added journaled external completions with scalar `RAX` results and validated
  writes to existing guest memory.
- Added an exact-match capability registry and a deterministic Windows
  bootstrap policy. There is no arbitrary Win32 passthrough fallback.
- Converted memory-semantic failures into structured machine traps rather than
  allowing Python exceptions to escape the executor.

## Observed Behaviour

- The first `cmd.exe` blocker (`sub rsp, 40`) was removed by the generic
  arithmetic/register layer rather than an instruction-specific workaround.
- Execution captured `GetSystemTimeAsFileTime`, `GetCurrentProcessId`,
  `GetCurrentThreadId`, `GetTickCount`, `GetModuleHandleW`, and CRT setup by
  their actual imported identities.
- Serviced system calls are reversible graph edges: stepping backward restores
  their pending request, stack, registers, and memory.
- A `MOVSXD r64, r/m32` width-inference bug was exposed by the real PE-header
  validation path and fixed with an operand-specific regression test.
- The latest probe reached the allowlisted frontier at
  `msvcrt.dll!__set_app_type` after 162 subject instructions/transitions.
- Focused semantic, state-buffer, chip-layout, system-port, and reversible
  execution tests passed. The most recent combined subset was 23 passing; the
  new semantic/system-port subset was 11 passing.

## Lessons Learned

An instruction token name is not sufficient to infer every operand width:
mixed-width operations such as `MOVSXD` require operand-specific decoding.
Real program execution is valuable precisely because it exposes these semantic
composition errors earlier than isolated instruction tests.

PE imports should be resolved into symbolic capability identities before
execution. Treating an on-disk IAT thunk value as a code address loses both
safety and the information a shell needs to capture or emulate the call.

External completion is part of machine history, not an out-of-band mutation.
That makes nondeterministic host observations recordable and reversible while
allowing deterministic virtual policies to replay exactly.

## Next Steps

- Continue the `cmd.exe` frontier loop through coherent semantic and CRT/Win32
  capability families.
- Serialize guest request argument/effect payloads into the physical shell ABI
  rings and connect native and HTML machine-program hosts.
- Add PE base relocations, exported-module mapping, and delay imports.
- Replace independent per-core memory with a deterministic shared-memory
  scheduler and explicit atomic ordering.
- Add translated basic-block caching whose journal boundaries and invalidation
  preserve exact reverse execution.

## Prompt History

> can you loop this as a goal and fill things out trying to reach for the sun

