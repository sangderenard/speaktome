# Reversible binary-machine continuation handoff

**Date:** 2026-08-05
**Repository worked in:** `C:\dev\Powershell\turing`
**Goal status:** active; deliberately not marked complete

## Abstract

This is the continuation record for the long-running reversible Windows AMD64
binary-machine effort. It is written so a new agent can resume from the current
worktree rather than reconstructing intent from chat history or treating an
earlier assurance document as the stopping point.

The project currently loads and executes the system `cmd.exe`, journals exact
instruction and capability edges, compiles safe AMD64 prefixes to authenticated
WebAssembly journals, reverses and cold-replays segmented tapes, exposes all 54
register/observation cells plus device output to native/HTML shader displays,
and fails closed at unknown semantics or capability shapes. The active goal is
broader than the successful `echo hello` proof and remains unfinished.

## Important chronology: assurance documents were checkpoints, not a stop

During the work, these Turing documents were repeatedly updated as assurance
snapshots:

- `docs/REVERSIBLE_MACHINE_CHIP.md`
- `docs/REVERSIBLE_MACHINE_COMPLETION_AUDIT.md`
- `docs/SHELL_SYSTEM_PORTS.md`

Those edits recorded evidence available at that moment. Work continued after
each update. Do not infer that the last paragraph of an older report or an
earlier test count is the final implementation state. The current worktree,
the tapes listed below, and fresh commands are authoritative. This report is a
continuation map, not a declaration that the full goal is complete.

The older guestbook entry
`1785882451_DOC_Reversible_AMD64_PE_Execution_And_System_Ports.md` contains the
long chronology through synchronization, VFS, registry, linking, threads,
segmented paths, Wasm journals, and the first real child executor. This report
starts after that chronology and records the newer virtual-memory, browser
storage/device, pipe, and nonlocal-control work.

## Mandatory orientation before editing

1. Start in `C:\dev\Powershell\turing`.
2. Read `C:\dev\Powershell\turing\AGENTS.md` completely.
3. Read this report and the three Turing assurance documents named above.
4. Inspect `git status --short`. The worktree is intentionally dirty and
   contains many unrelated user changes. Preserve them. Do not reset, checkout,
   force-add ignored build directories, or rewrite unrelated files.
5. Use `apply_patch` for source/document edits. Do not manually install with
   pip. The tests used here run in the already configured Python environment.

## Active objective

Build toward an end-to-end reversible, recompiling binary-machine program that
can load and run Windows AMD64 programs such as `cmd.exe`, resolve or capture
external activities through capability-gated shell system ports, publish real
register/memory/device snapshots, and present machine state plus subject output
in native and HTML shader displays while preserving deterministic replay and
fail-closed behavior.

The goal has no token budget. It must remain active until a requirement-by-
requirement audit proves broad completion; one successful command is strong
integration evidence, not proof of arbitrary Windows compatibility.

## Current verified baseline

### Ordinary real command

Fresh tape:

`C:\dev\Powershell\turing\build\cmd-echo-pipe-tier-20260805.segmented-tape`

Command:

```powershell
python examples/reversible_cmd_probe.py --new --segmented `
  --machine-backend node-wasm `
  --tape build/cmd-echo-pipe-tier-20260805.segmented-tape `
  --maximum-boundaries 1000 /c "echo hello"
```

Observed:

- halted normally at guest step 16,010;
- exit code 0;
- 499 deterministic external completions;
- 16,510 tape records;
- 15,459 instructions committed through authenticated Wasm journals;
- 523 exact interpreter fallbacks;
- `console.output == b"hello\r\n"`;
- all remaining Wasm denials were intentional indirect-control or lifecycle
  sentinel boundaries.

### Real command-pipeline/error-recovery route

Fresh tape:

`C:\dev\Powershell\turing\build\cmd-pipe-card-v2-20260805.segmented-tape`

Command:

```powershell
python examples/reversible_cmd_probe.py --new --segmented `
  --machine-backend node-wasm --demo-card hello-card `
  --tape build/cmd-pipe-card-v2-20260805.segmented-tape `
  --maximum-boundaries 2000 /c "echo data | hello-card alpha"
```

Observed:

- real `cmd.exe` created and closed `pipe.1`;
- the command was not resolved inside the pipeline, so the expected terminal
  diagnostic was produced;
- MSVCRT `longjmp` error recovery executed instead of remaining unsupported;
- halted normally at guest step 23,232 with exit code 255;
- 539 completions and 23,773 tape records;
- 22,632 instructions committed through Wasm journals;
- cold reopen restored position 23,772 and the exact terminal/exit state;
- one `step_backward()` returned to non-halted step 23,231;
- one `step_forward()` reproduced the complete halted tip byte-for-byte.

This proves real pipe setup/lifecycle and nonlocal recovery. It does not yet
prove registered-card resolution within a pipeline. A prior interactive proof
did resolve and execute `hello-card`; the pipeline-specific resolution gap is
the current diagnostic target.

## Latest compatibility inventory

Computed by loading the current system `cmd.exe`, taking its unique
`(library, symbol)` references, and intersecting them with
`deterministic_windows_bootstrap_port().handlers`:

```text
refs=286
handler identities=205
supported cmd imports=153
unsupported cmd imports=133
```

The count distinguishes total port aliases from imports actually present in
this `cmd.exe`. Do not write “205 of 286 supported”; the correct intersection is
153.

## Newer implementation slices

### Virtual memory

Primary file: `src/compiler/virtual_memory.py`.

- Immutable region catalog for initial mappings and managed allocations.
- Bounded reserve+commit `VirtualAlloc` with RW or execute-RW protection.
- Whole managed-region `VirtualFree(MEM_RELEASE)`.
- `VirtualQuery` and bounded current-process `ReadProcessMemory`.
- Dynamic executable pages enter decoding, page-versioning, translated-cache
  invalidation, exact tape serialization, multicore synchronization, and
  trace-SSA `virtual_memory` provenance.
- Reserve-only, decommit, partial release, unsupported protection, overlap,
  and remote-process reads remain fail closed.

### Browser persistence and typed shell devices

Primary files: `src/compiler/shell_io.py` and
`src/compiler/wasm_html_shell.py`.

- Added declared `indexed_db` and `opfs` mount kinds.
- Sources must be relative namespace paths; browser mounts cannot expose host
  paths.
- Persistent stores hydrate into the synchronous virtual-file map before run.
- Writes are immediately visible in that map; `flushVirtualFilesystem()` is
  the durability barrier.
- Added `system_devices` capability and `kind=device` ports with enforced
  direction, byte buffers, subscriptions, handlers, and machine-runtime input
  binding.
- The generic native shell intentionally does not advertise the new device
  capability until it has a corresponding adapter. The binary machine's native
  host already has its separate reversible console/device bridge.
- A real headless Chrome test writes IndexedDB and OPFS files, evicts them from
  memory, rehydrates both, and checks exact bytes.

### Reversible pipes and descriptor aliases

Primary file: `src/compiler/machine_system_ports.py`.

- Pipe objects reuse existing reversible `system_state` for endpoint metadata
  and `device_state["pipe.<id>"]` for bytes. No parallel host pipe exists.
- `CreatePipe`, `DuplicateHandle`, `ReadFile`, `WriteFile`,
  `FlushFileBuffers`, `GetFileType`, and `CloseHandle` share endpoint state.
- Empty read blocks while a writer exists; final writer close yields EOF;
  writing without readers reports broken pipe; capacity is bounded.
- MSVCRT `_pipe`, `_get_osfhandle`, `_open_osfhandle`, `_dup`, `_dup2`, and
  `_close` are aliases over the same handles.
- `STARTF_USESTDHANDLES` routes registered virtual-child stdin/stdout/stderr
  through validated inheritable pipe endpoints.
- Segmented tapes already serialize system/device maps, so endpoint counts,
  descriptor bindings, and buffered bytes resume exactly.
- `machine_trace_ssa.py` adds a distinct `pipe` effect domain when pipe system
  or device resources change.

### Nonlocal CRT control

Primary files: `src/compiler/machine_execution.py`,
`src/compiler/amd64_machine_semantics.py`, and
`src/compiler/machine_system_ports.py`.

- Added `MachineExternalControlTransfer` for a capability-owned nonlocal jump.
- `_setjmp` now records the post-return shadow call stack under buffer-keyed
  reversible system-state keys in addition to writing the x64 jump buffer.
- `longjmp` restores RBX, RSP, RBP, RSI, RDI, R12-R15, RIP, return value,
  MXCSR/FPCSR, XMM6-XMM15, and the saved shadow call stack in one external
  completion edge.
- The control transfer is not implemented as a fake callback or normal return;
  completion removes the pending request and replaces exact control state.

## Current tests

The most recent broad selection passed 253 tests in about 111 seconds:

```powershell
python -m pytest `
  tests/test_machine_system_ports.py `
  tests/test_amd64_machine_semantics.py `
  tests/test_reversible_machine_execution.py `
  tests/test_virtual_filesystem_and_system_tape.py `
  tests/test_machine_trace_ssa.py `
  tests/test_machine_wasm_runtime.py `
  tests/test_machine_state_buffer.py `
  tests/test_machine_snapshot_host.py `
  tests/test_shell_io.py `
  tests/test_wasm_html_shell.py -q
```

Also run `git diff --check`. Line-ending warnings are expected in this Windows
worktree; whitespace errors are not.

## Files in the current slice

The current task touched at least:

- `src/compiler/virtual_memory.py` (new, untracked at the time of this report)
- `src/compiler/virtual_registry.py` (new from the preceding slice)
- `src/compiler/amd64_machine_semantics.py`
- `src/compiler/machine_execution.py`
- `src/compiler/machine_system_ports.py`
- `src/compiler/machine_system_tape.py`
- `src/compiler/machine_trace_ssa.py`
- `src/compiler/shell_io.py`
- `src/compiler/wasm_html_shell.py`
- the corresponding machine, tape, shell, and browser tests
- the three Turing assurance documents named above

There are numerous unrelated modified files in the Turing worktree. Do not use
the size of the full diff as evidence that all changes belong to this goal.
Inspect path-scoped diffs.

## Continuation procedure

1. Re-run the 253-test selection above or a focused subset before editing.
2. Inspect the pipeline tape and determine why the registered
   `/c/work/hello-card.exe` marker is not selected inside pipeline execution,
   although the existing interactive command proof selects it.
3. Instrument through tape metadata or a read-only diagnostic around VFS
   search and command resolution. Avoid adding a generic host exec fallback.
4. Make the registered child receive pipe stdin and return pipe stdout in a
   real `cmd` pipeline, then prove cold reverse/forward at the resulting tip.
5. Continue remaining coherent capability families: process lifecycle,
   console-screen operations, and exception/unwind behavior.
6. Preserve the state/journal ABI when broadening recompilation or adding a
   native backend. Every compiled guest instruction must remain independently
   authenticated and reversible.
7. Keep updating this report or add a newer timestamped continuation report as
   work advances. Assurance docs may be updated along the way, but explicitly
   say when work continued beyond them.

## Known incomplete areas

- Pipeline-specific card resolution is not proven.
- Arbitrary Windows AMD64 instruction and import coverage is not complete.
- Suspended thread/process creation, forced termination/process detach, event
  wait sets, native delay-helper side effects, broad exception/unwind, and full
  dynamic loading lifecycle remain partial.
- Native code generation behind the authenticated instruction journal is still
  absent; current recompilation is WebAssembly via the persistent Node host.
- The HTML persistent VFS is a shell backend; full browser-to-live-machine VFS
  synchronization still needs a controller contract beyond the current map and
  device adapter APIs.
- Cross-bundle discovery/version/signature negotiation remains a larger API
  project. HTML external references are intentionally bundle-only.

## Lessons learned

- Use real subject execution to discover composition errors, then implement a
  coherent family rather than special-casing one RIP.
- Store external effects as immutable state transitions. Pipe bytes and
  endpoint counts then inherit reversal, tape, branching, multicore, and SSA
  behavior without an inverse-operation subsystem.
- A nonlocal jump needs an explicit control-transfer completion. Treating it as
  a normal external return corrupts RSP and shadow-stack provenance.
- Distinguish guest threads from possible-world forks. Threads share scheduled
  process state; forks retain independent exact/SSA suffix DAGs.
- Distinguish assurance checkpoints from completion. Passing a real command is
  a regression anchor, not permission to narrow the active objective.

## Next steps

- Resolve and prove registered card execution inside a real `cmd` pipeline.
- Add exact process lifecycle and wait-set families without host passthrough.
- Continue unsupported import families using the refreshed 133-import list.
- Preserve cold segmented reversal and shader-observable snapshots after every
  capability/recompiler expansion.
- Refresh this handoff after the next material frontier.

## Prompt history

> please make this your goal and pursue it with diligence

> long runtimes are fine let it do it's thing

> A virtual filesystem in the io shell contract system would be useful if you didn't intuit that

> you should make this ystem tape the whole thing, I don't know if you're doing that already, or for the system calls, but we should be taping this to resume

> put in the system a way to annotate features of moments on the tape, so that things like "this might be wrong" can be marked with a color flag

> we may want to use some graph mechanics to build dependency graphs off the tape if that isn't already figured out so resumes are flawless

> can we fill a graph we turn to ssa as we use the binary read, and so, like, develop a decision tree rewinding as we please and starting from anywhere, building out many paths concurrently in existence and having had their operations reduced and their machine simulation available to replay - is that something we would end up reaching because of the loops in programming, or does the binary complicate that too much? Can we retain provenance of binary to ssa to track it's provenance every step through raising?

> this is a good time to suggest you, in this process and uninterrupted from what you were doing, consider enabling the shell and thus the runtime of this program, indexeddb, opfs, and system devices, either by fooling the windows virtual environment or constructing virtual versions of things

> you have about an hour of tokens left, I want you to make continuation and experience documentation in speaktome that allows an agent to resume your work regardless of where you left off, explaining that your documentation was done for assurances but your work continued, and then continue on your work please until tokens run out
