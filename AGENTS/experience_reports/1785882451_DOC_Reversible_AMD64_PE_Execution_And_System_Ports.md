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
- Added an immutable shell-owned virtual filesystem with explicit memory,
  bundle, and native-only host-directory mounts.
- Added an append-only resumable system tape with the subject binary, periodic
  checkpoints, page deltas, virtual filesystem state, external requests, and
  every forward/backward transition.
- Promoted the guest environment map and set/delete operations into reversible
  system-tape state rather than reading or mutating the host environment.
- Added colored tape annotations for moments/spans, automatic red compatibility
  notes at RIP when an instruction lacks a handler, and snapshot/shader
  propagation of the active annotation color to the displayed RIP cell.
- Added durable tape dependency graphs: per-core parent lineage, typed external
  request/completion and branch-source edges, validated runtime-dispatch nodes,
  and a persisted static/dynamic external-link catalog.
- Added reversible terminal device state, guest-side wide formatting, console
  output, MSVC x64 jump buffers, implicit divide/sign-extension semantics, SRW
  locks, exit callbacks, and explicit halted/exit-code process state.
- Added a native OpenGL snapshot viewer with free-spin, clocked, forward, and
  reverse controls, plus terminal/framebuffer presentation below the contiguous
  register-bank display.
- Made dream-document HTML emission optionally cold-resume a system tape and
  embed its dependency-validated `TMSNAP01` state for automatic display on page
  load.

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
- A taped `cmd.exe /c "echo hello"` probe crossed module filename, current
  directory, and environment lookup and reached `msvcrt.dll!_wcsicmp` after
  1,977 subject instructions; after implementing that family, the next taped
  run reached `SetEnvironmentVariableW` at instruction 2,050. That operation is
  now a journaled environment-state effect. The following run reached
  `towupper` at instruction 3,244; deterministic wide-character case conversion
  is now covered as a coherent CRT/classification family. Subsequent VFS path
  and directory-enumeration coverage advanced the frontier to the
  process-environment alias of `SetCurrentDirectoryW` at instruction 3,569.
  Durable resume then carried the same run through string instructions,
  wide-string CRT calls, heap reallocation, executable-search policy, drive
  typing, and extended directory enumeration to `GetUserDefaultLCID` at
  instruction 10,332. After adding the SDK-checked locale table, the run reached
  `msvcrt!setlocale` at instruction 10,496 and wrote its first real amber
  unsupported-external annotation into the persisted tape.
- Focused semantic, state-buffer, chip-layout, system-port, and reversible
  execution tests passed. The latest combined graph, compatibility-annotation,
  snapshot, dream-display, shell-I/O, tape, and reversible-semantics subset was
  108 passing. A missing
  instruction handler now leaves a red compatibility note at the exact RIP;
  its architecture, decoded tokens, and encoded bytes survive tape replay and
  its color is published with the RIP cell in the display snapshot.
- The real `cmd.exe /c "echo hello"` run halted normally at instruction step
  16,010 with exit code 0. A cold restart reconstructed 16,537 dependency-graph
  state nodes and recovered `console.output == b"hello\r\n"`, generation 1.
  A newly published shader snapshot exposed the identical bytes as a
  `TERMINAL` / `UTF8` output descriptor.
- Generated a self-contained HTML artifact from that cold tape; its embedded
  snapshot is published to the dream document's interior-owned shader display
  without requiring a Python process at examination time.
- Reached a genuinely receptive interactive `cmd.exe` state at
  `ReadConsoleW`, injected reversible terminal input, and returned to the next
  prompt without treating an empty input device as EOF.
- Added a fail-closed virtual-program registry and capability handlers for
  `CreateProcessW`, wait, and exit-code observation. Direct tests prove that a
  registered card-set executor can receive a Windows invocation and publish
  its deployment metadata and output without launching a host process. A real
  `cmd.exe` attempt has not yet reached that deployment, so the end-to-end
  claim remains open.
- Lifted validated tape lineages into effect-aware trace SSA whose operations
  retain tape sequence, RIP, instruction bytes/tokens, resource versions, and
  dependencies. Per-output dependencies reduced an observed terminal slice
  from 8,320 operations to 572 while emitting an explicit reduction witness.
- Added possible-world read-head forests distinct from guest threads: retained
  parents can fork at any history position, siblings own private reversible
  journals and constraints, and independent heads can advance on physical
  workers without merging writes.
- Converted the large interactive tape to content-addressed immutable
  segments. The current store resumes 53,299 records from 210 segments while
  caching only 49 decoded records in the observed cold-resume check.
- Added a separate content-addressed trace-SSA segment DAG. Child heads name a
  parent and fork sequence and store only their suffix; iteration streams the
  shared prefix plus suffix with one decoded chunk resident. Identical chunks
  deduplicate by digest. Streaming ingestion also seals an operation iterator
  at the configured chunk capacity, so construction as well as replay is
  bounded. The focused machine/tape/path/SSA/port suite is now 100 passing.
- Corrected `FindFirstFileExW` thread-local error semantics after the tape
  proved that stale `ERROR_ACCESS_DENIED` made `cmd.exe` abandon PATHEXT search
  after `.COM`. With `ERROR_FILE_NOT_FOUND`, it proceeded to the registered
  `.EXE` candidate.
- Added reversible process-attribute-list and startup-info contracts. Real
  `cmd.exe` used the internal additive attribute `0x00060001`; its four-byte
  payload is retained opaquely rather than interpreted as a host facility.
- Completed the real child path. At parent sequence 67,946, `cmd.exe` launched
  `C:\work\hello-card.exe` through `bundle:demo/hello-card@local` and
  `card-set:hello-card:v1`, produced `[card-set:hello-card] alpha beta`, exited
  zero, and returned to `ReadConsoleW` at step 67,459. The linked child tape is
  `67ec12784bd487637814902fbc3126a394c48d68ea80a5975cfd7b1365c9d58b`.
- Indexed runtime dispatch targets in the segmented manifest, removing an
  O(total tape records) cold-resume scan. Legacy stores migrate after one
  validated scan.
- Made dream-document snapshot emission accept segmented stores and serialize
  retained machine states directly, avoiding PE recompilation for display.
  Browser/WebGL verification showed the exact card terminal output and all 54
  contiguous register cells with their 64-bit hexadecimal contents. The
  generated shell's previously stranded `offsetForKey` body was repaired; its
  fatal banner is now empty and hidden. The combined relevant regression suite
  passes 157 tests.
- Connected the continuing executor to the HTML interior without moving guest
  semantics into JavaScript. A loopback-only native host now publishes complete
  monotonic `TMSNAP01` flips from a single-owner controller, retains only the
  newest display generation, and accepts bounded terminal messages through a
  queue that only that controller drains. The dream shell installs polling and
  terminal-input liaison methods plus an on-display command field, and
  `examples/reversible_machine_web_host.py` resumes or creates a segmented tape,
  serves the display, and can register the demonstration card executor. The
  expanded machine/shell/tape/SSA regression selection passes 163 tests.
- Added generation-invalidated, SHA-256-fingerprinted translated basic blocks
  for the single-core free-spin path. Ordinary effects are pre-bound, control
  and external frontiers terminate blocks, and every instruction still creates
  its own reversible state, edge, tape event, and observer callback. A synthetic
  200,000-instruction loop measured about 1.10x over generic per-step dispatch.
- Bounded each binary core's resident reverse journal to 4,096 states by
  default. Absolute history positions survive pruning; backward execution pages
  contiguous exact states from JSONL or segmented tape and evicts the resident
  future rather than accumulating it. The segmented cold-reverse test retains
  four states, reverses twenty transitions to the exact initial register file,
  and reopens a 41-record store. The expanded relevant suite passes 166 tests.
- Made virtual cores actual guest threads with a deterministic, core-index,
  sequentially-consistent memory schedule. Later cores observe earlier writes
  within a barrier; all cores journal the same final memory, overlapping ranges
  retain race provenance, and exact tape rows include the schedule contract.
  Barrier reversal restores every pre-cycle memory. Capability memory writes
  are journal-broadcast as `shared_memory_sync` edges. Forked read heads remain
  isolated possible worlds and never enter this memory merger. The full
  relevant machine/shell/tape/SSA selection now passes 168 tests.
- Added a separate content-addressed exact-state DAG for possible-world heads.
  Each child references its parent/fork position, stores one full anchor and
  only its private suffix, and can reverse across a two-state hot window by
  paging independently decodable chunks. Parallel workers append through a
  locked bounded tail; identical sibling chunks deduplicate, reopen/append is
  streaming, and corrupt objects fail closed. Exact states retain binary token,
  semantic, address, and byte provenance and can stream directly into the
  existing trace-SSA segment DAG without materializing either full path.
- Re-ran the real interactive subject after translation, hot-window, and path
  storage changes using a disposable copy of the authoritative tape. The run
  grew from 69,949/278 records/segments to 77,705/309, published 578 snapshots,
  executed `echo live-check`, and returned to `ReadConsoleW` with no controller
  failure while retaining exactly 4,096 hot states. The expanded relevant suite
  passes 171 tests.
- Added reversible executable-page versions and translated-block coherency.
  Guest semantic writes and admitted capability completions invalidate cached
  blocks; execution decodes the bytes currently in guest memory and fails
  closed on unsupported mutations. Rewind restores both original code bytes
  and the prior version state. Two new regressions bring the expanded relevant
  suite to 173 tests.
- Added bounded PE base-relocation parsing and runtime-base loading. Mapping,
  decoded instruction identities, executable pages, dispatch plans, IAT slots,
  and exact loader state now use one runtime address namespace; tape resume
  recovers that base and rejects conflicts. `GetModuleHandleW(NULL)` was changed
  to return the taped loader base after a complete relocated-subject run exposed
  the old hardcoded value. System `cmd.exe` moved from `0x140000000` to
  `0x150000000`, applied 372 `DIR64` records, executed 16,010 instructions,
  serviced 499 deterministic calls, printed `hello\r\n`, and halted at exit
  code zero. Four new regressions bring the expanded relevant suite to 177.
- Added bounded PE export-directory parsing for module name, address/name/
  ordinal tables, aliases, ordinal-only symbols, and symbolic forwarders. A
  synthetic regression covers concrete and forwarded exports; a real
  `kernel32.dll` check catalogued 1,636 exports (1,449 concrete and 187
  forwarded). The expanded relevant suite is now 178 tests. Mapping approved
  dependency bytes and persisting their module provenance remains the next
  linker step.
- Implemented capability-supplied mapped DLL linking. Approved module bytes are
  decompiled, range-checked, mapped/relocated, and added to a unified executable
  dispatch namespace. The deterministic link plan resolves names and ordinals,
  follows bounded forwarder chains, routes unsupplied leaves back to shell
  capabilities, and records a witness for every IAT slot. JSONL retains module
  bytes and segmented stores use `modules/<sha256>.bin`; both replay paths
  reconstruct and compare the complete plan before resuming. A synthetic main
  PE now performs a real RIP-relative IAT call into a dependency, which executes
  guest AMD64 and returns `RAX=42` before and after both tape formats. Overlap
  unresolved-forwarder, image-overlap, and reserved environment-overlap
  regressions fail closed. The expanded relevant suite is now 182 tests.
- Added a bounded `dependency_provider` capability that is queried only for
  unresolved identities actually reached by the evolving import/forwarder
  graph. Returned images become tape-owned bytes; refusals remain capability
  calls. A two-module regression acquires `demo.dll`, follows its
  `KERNEL32.Sleep` forwarder, acquires a separately based `KERNEL32.dll`, and
  executes the final guest export. Added strict PE delay-import parsing and
  deterministic delayed-IAT/module-handle lowering with explicit delay binding
  provenance. The expanded relevant suite is now 185 tests. Native first-use
  delay-helper side effects remain future work.
- Added strict PE TLS-directory parsing and first-thread initialization. TLS
  templates/zero-fill blocks and the pointer vector are allocated in the taped
  system arena, `GS:[0x58]` and each image index cell are populated, and
  dependency/subject process-attach callbacks run through a reversible loader
  return sentinel before the entry point. A callback regression executes guest
  AMD64, returns `RAX=7`, reverses exactly to its callback-ready state, and
  repeats after JSONL restart. Real `cmd.exe` and `kernel32.dll` were checked
  and presently have no TLS directory. The expanded relevant suite is now 186
  tests.
- Generalized the loader callback sentinel into an ordered startup-call queue.
  Each mapped dependency now runs TLS callbacks followed by
  `DllMain(DLL_PROCESS_ATTACH)` before the subject starts. Queue kind, module,
  target, index, and success requirement are exact state. A false DllMain
  regression traps on the return edge before subject entry; all existing
  mapped-DLL execution and replay checks now traverse successful DllMain calls.
  The expanded relevant suite is 187 tests.
- Added capability-gated `CreateThread` activation within fixed virtual-core
  capacity. Parked heads preserve the physical register/snapshot ABI; creation
  allocates a private stack and TEB, clones the process TLS vector/templates,
  runs thread-attach startup calls, and enters the guest routine with its
  parameter. Outer return records the exit code and parks only that core. Spawn
  records depend explicitly on the external completion, and barrier reversal
  restores both the parked core and pending request. Reversible thread IDs come
  from taped system state rather than a host counter. Two regressions cover
  ordinary return and TLS thread attach, bringing the suite to 189 tests.
- Added reversible thread-handle lifecycle for zero-time polling, blocking
  `WaitForSingleObject`, exit-code reads, and close. A blocked parent records
  its request on tape and parks while the child core advances; child exit is
  merged through process-global state and the exact waiting call then resumes.
  The new integration regression brings the suite to 190 tests.
- Made dormant possible-world storage explicit. `release_head` seals a partial
  exact-state suffix and evicts its append cursor/final state; a later append
  reconstructs only the cold tip. `release_all` also clears the sole decoded
  object cache. Exact state and SSA paths therefore scale with persistent
  segments while RAM follows only the chosen active frontier.
- Completed the clean auxiliary-thread teardown edge. Returning from a start
  routine now preserves its exit code, executes TLS callbacks and dependency
  DllMain entries with `DLL_THREAD_DETACH`, and signals the thread handle only
  after cleanup returns. Module groups unwind in reverse initialization order;
  callback-array order within a module remains stable. The callback sequence,
  pending exit code, and final parked state survive exact tape restart.
- Added the first executable machine-block recompilation tier. Safe
  register-only MOV/NOP prefixes lower to a real WebAssembly binary over a
  contiguous register/RIP/flags/steps buffer. The module emits a complete
  checkpoint and address/semantic/code-digest witness after every guest
  instruction; the Python bridge validates and commits those as ordinary
  reversible edges. A Node regression executes the binary, reconstructs the
  state, reverses it exactly, rejects a tampered witness, and proves strict
  fail-closed behavior at the following RET. The relevant suite is 192 tests.
- Extended that Wasm tier through register-destination `ADD`, `SUB`, `AND`,
  `OR`, and `XOR` with immediate or register sources at 8/16/32/64-bit widths.
  The module computes `CF/PF/AF/ZF/SF/OF` itself and matches the reference
  interpreter across carry, borrow, signed overflow, parity, zero, sign,
  partial-register, and zero-extension boundaries. A mixed three-instruction
  artifact produces and reverses three separate authenticated edges. The
  relevant suite is 194 tests.
- Added bounded static guest-memory mirrors for Wasm block execution.
  Displacement-only and RIP-relative MOV loads/stores carry kind/address/width/
  before/after witnesses in every instruction record; journal reconstruction
  validates reads before applying writes to immutable paged memory. Dynamic
  addresses, mirrors over 64 KiB, stale instruction bytes, and writes targeting
  translated executable pages fail closed. A real Node round trip matches two
  interpreter states and reverses both, bringing the suite to 196 tests.
- Added direct and conditional relative control exits to compiled Wasm. Every
  control witness names its permitted successor set; the actual RIP is journaled
  and rejected if it is not one of those targets. All canonical AMD64 condition
  formulas (`CF/ZF/SF/OF` combinations) differentially match the interpreter,
  commit as one exact edge, and reverse. The suite is now 198 tests.
- Added exact-state-specialized direct CALL and ordinary RET at compiled block
  entries. The Wasm binary and readable WAT both update RSP/RIP and a bounded
  guest stack mirror; every checkpoint binds the stack-memory effect and exact
  shadow-stack push/pop value/depth to its instruction witness. Sentinel,
  outermost, and post-prefix calls fail closed. Differential Node execution,
  commit, and reverse bring the focused suite to 200 tests.
- Installed an explicit `node-wasm` automatic runner policy. One bounded,
  persistent Node worker verifies module digests, caches instances, executes
  safe prefixes, and returns the existing authenticated instruction journal.
  Shell ticks and the live HTML snapshot controller commit/tape each compiled
  instruction normally and fall back at the exact unsupported lifecycle edge.
  Artifact/denial caches are bounded, host failures remain fatal, and module
  memory now scales to the declared guest window. Persistent reuse, CALL/RET,
  cold reversal, controller publication, and a two-page 64 KiB mirror bring the
  focused suite to 206 tests. A real `cmd.exe` probe reached `WAITING_EXTERNAL`
  after 432 transitions with 87 instructions committed from 68 Wasm dispatches;
  a one-operand unsupported-memory planning defect found by that probe now has
  a fail-closed regression test.
- Added exact-state block-entry specialization for dynamic MOV loads/stores
  using AMD64 base/index/scale/displacement plus FS/GS bases. Artifacts validate
  the address-producing state before packing or journal reconstruction, and the
  dispatcher uses the same inputs in its cache key. Internal register-indirect
  CALL/JMP now compile with exact successor and shadow-stack provenance;
  memory-resolved and external targets remain interpreter/capability edges.
  The equivalent real `cmd.exe` frontier now commits 109 of 432 transitions
  through 82 Wasm dispatches (up from 87/68). Five differential, reversal,
  guard, and external-boundary cases bring the focused suite to 211 tests.
- Added runtime LEA lowering that observes registers produced earlier in a
  compiled block, flag-only register/immediate CMP and TEST, and exact-entry
  register/immediate PUSH plus register POP with stack-memory witnesses. The
  bounded dispatcher now reports denial counts by semantic family. On the same
  432-transition real `cmd.exe` frontier, compiled coverage rose again to 212
  instructions through 126 Wasm dispatches; LEA, TEST, PUSH, and POP disappeared
  from the denial inventory. Two integration tests plus expanded flag cases
  bring the focused suite to 213 tests.
- Extended flag-only CMP/TEST through one static or exact-entry dynamic memory
  operand. The same read witness authenticates address/width/value, signed
  immediates extend to the operation width, and operands remain unchanged.
  Comparison/test denials disappeared from the real frontier; `cmd.exe` now
  commits 241 of 432 transitions through 139 Wasm dispatches while reaching the
  identical capability request. Differential static/dynamic and tamper cases
  bring the focused suite to 214 tests.
- Added token- and reason-level bounded denial telemetry, which exposed whole-
  block guest-window planning as the main remaining MOV blocker. Planning now
  emits the maximal contiguous ≤64 KiB prefix and reports the distant access as
  its continuation. Added exact scalar memory read-modify-write effects for
  arithmetic/logical destinations and register/memory MOVSXD/MOVZX-style
  extension. The same real frontier now compiles 326 of 432 transitions through
  183 Wasm dispatches (about 75%) with unchanged capability behavior. Three
  prefix/RMW/extension regressions bring the focused suite to 217 tests.
- Added binary-Wasm and readable-WAT lowering for immediate SHL, NEG/NOT,
  SETcc, INC/DEC, register XCHG, and implicitly atomic 64-bit memory/register
  XCHG. Memory forms retain one authenticated read-modify-write effect;
  INC/DEC preserve CF, and every compiled instruction remains independently
  reversible. A broader fresh `cmd.exe /c "echo hello"` measurement serviced
  167 deterministic capabilities and reached the unchanged unsupported
  `msvcrt!_local_unwind` boundary after 3,043 transitions. Coverage rose from
  2,140 to 2,482 compiled transitions (about 82%) and denial blocks fell from
  236 to 130. Five differential/reversal/tamper cases bring the focused suite
  to 222 tests.
- Added CMOV with unconditional memory-read provenance and correct untaken
  32-bit upper-register preservation; SBB with exact carry-in overflow; SAR
  and ROL flag behavior; and unsigned `MUL r/m64` with the complete 128-bit
  RDX:RAX result computed from exact 32-bit limbs. The same 3,043-transition
  run now compiles 2,647 transitions (about 87%) and has 104 denied blocks.
  Its scalar denial inventory is empty: the only remaining non-control forms
  are one XMM move and one XMM XOR, which require a vector-aware checkpoint
  ABI. Thirteen condition/boundary/memory/reversal cases bring the focused
  suite to 235 tests.
- Extended the compiled checkpoint to `turing.machine-block-state.v2`: all 16
  XMM registers are retained as 32 contiguous qwords, and memory effects carry
  128-bit low/high before and after halves. Added XMM XOR plus aligned/unaligned
  vector loads and stores, with executable-page rejection and exact reversal.
  The existing 54-cell `TMSNAP01` path displays compiled XMM changes without a
  second projection ABI. The wider 512-byte journal record exposed a fixed
  guest-offset overlap in long blocks; the host now page-aligns the guest
  mirror after the actual journal extent, covered by a nine-instruction memory
  regression. The 3,043-transition real run now compiles 2,669 transitions
  (about 88%); all 103 remaining denials are validated indirect calls/jumps and
  the semantic-denial inventory is empty. Four regressions bring the focused
  suite to 239 tests.
- Confirmed that branching exact-state and trace-SSA histories are persistent
  segmented DAGs rather than resident lists. Dormant heads seal their mutable
  tail, evict their final state and append cursor, and later reopen only the
  lineage tip; one decoded immutable object is cached at a time.
- Implemented the bounded x64 MSVCRT `_local_unwind(frame, target)` shape used
  by the real command processor. It parses the guest PE exception directory
  and C scope table, dispatches one guest `__finally` callback with explicit
  RCX/RDX ABI writes, and fails closed on malformed, chained, non-C, or
  multi-callback metadata. Compiled journal commit now authenticates exact
  runtime-decoded instructions instead of requiring every witness address in
  the initial static map. `cmd.exe /c "echo hello"` subsequently halted with
  exit code zero after 15,205 guest steps and 589 deterministic capability
  completions. Four regressions bring the focused suite to 243 tests.
- Promoted segmented-tape bootstrap from a private web-host helper to
  `BinaryMachineProgram.begin_segmented_system_tape`, and exposed segmented
  creation plus the `node-wasm` policy in `reversible_cmd_probe.py`. A real
  compiled `cmd.exe /c "echo hello"` run wrote 16,510 records into 66 immutable
  objects (8,554,601 bytes), committed 15,420 of 16,010 instructions through
  authenticated Wasm journals, and retained `hello\r\n` in the console device.
  Cold inspection exposed and fixed a resume defect: tape loading had restored
  state but reset logical history to zero. Both JSONL and segmented loaders now
  restore the record's absolute executor position; one cold reverse hydrated a
  125-state suffix and one forward step reproduced the halted tip exactly. Two
  regressions bring the focused suite to 245 tests.
- Added binary-Wasm and readable-WAT lowering for CDQE/CQO and the scalar
  BT/BTR/BTS/BTC family. Memory register indices use the architectural signed
  adjacent-bit-string quotient, and the index register joins the guarded
  specialization provenance; modifying forms retain one exact RMW witness.
  This exposed and fixed a shared operand-width bug where the `M8` substring in
  a trailing `IMM8` was incorrectly selected as the memory width for
  `BTR_RM32_IMM8`. Five Node differential, flag, memory, signed-index, and
  reversal cases bring the focused suite to 250 tests; the binary-machine-only
  slice passes 207. A fresh segmented `cmd.exe /c "echo hello"` proof halted at
  16,010 steps with exit zero, 499 capability completions, and `hello\r\n`.
  Wasm journal coverage rose from 15,420 to 15,447 instructions and fallbacks
  fell from 537 to 525 because newly continuous blocks compiled beyond the six
  directly affected route instructions. The only remaining non-control denial
  is one `REP STOSW`; indirect calls/jumps and sentinel returns remain at their
  intentional capability/interpreter boundaries.
- Lowered the final ordinary route denial, `REP STOSW`, as a bounded
  exact-entry Wasm loop with a compact authenticated fill descriptor. The real
  occurrences each have RCX=8,192 and replace 16 KiB, so scalar before/after
  fields or one tape edge per word would be the wrong storage model. The
  descriptor binds RDI, count, AX, direction, and final RCX/RDI; reconstruction
  validates it against the parent state and materializes changed immutable
  pages, while the parent segment already retains the bytes needed to reverse.
  Forward/backward direction, descriptor tamper rejection, and exact reversal
  bring the focused suite to 252 tests and the binary-machine slice to 209.
  The fresh real proof still halts at 16,010 steps with exit zero and
  `hello\r\n`, now committing 15,459 instructions with 523 fallbacks. All 300
  denied blocks are intentional indirect-call/jump or lifecycle-return
  boundaries; the non-control semantic denial inventory is empty.
- Verified the compiled segmented proof as an observable artifact rather than
  only a tape statistic. Cold resume decoded its final segment, direct reverse
  hydrated positions 16,384--16,509, and one forward edge reproduced the
  halted state and `hello\r\n` exactly. The generated HTML/WebGL snapshot and
  loopback live host both decoded a 4,195,584-byte `TMSNAP01` generation with
  all 54 register cells and one UTF-8 terminal descriptor containing that
  output. The Windows desktop capture helper failed with HRESULT 0x80004002,
  so visual validation used installed Chrome through headless Playwright and
  did not attempt blind UI input.
- Added the coherent Win32 single-object synchronization family:
  `CreateMutexExW`, `CreateSemaphoreExW`, `OpenSemaphoreW`, `ReleaseMutex`,
  `ReleaseSemaphore`, `WaitForSingleObject`, and `WaitForSingleObjectEx`.
  Mutex ownership/recursion, semaphore counts, named-object bindings,
  thread-local errors, and previous-count writes are ordinary reversible
  completion effects in shared system state. Zero-time waits return the exact
  timeout code; nonzero unavailable waits yield to the virtual scheduler.
  Security descriptors and unsupported shapes remain fail-closed. The current
  system `cmd.exe` inventory is now 286 references, 169 admitted port
  identities, and 166 unsupported, seven fewer than immediately before this
  slice. The focused port suite passes 52 tests. The broad machine run passed
  212 tests and exposed one unrelated dirty-worktree mismatch: the Fortran
  inventory now contains 70 operations while its existing assertion expects
  65; that user-owned backend work was preserved.
- Closed the entire 15-import `api-ms-win-core-file-l1-1-0` frontier over the
  immutable virtual filesystem: synchronous create/open, share checks,
  read/write, cursor movement, truncate, size/flush/close, attributes,
  creation/access/modification times, file-time comparison/conversion, and
  deterministic volume information/free space. Ordinary file and enumeration
  handles retain distinct lifecycles. Handle access/share mode, position and
  file metadata survive system-tape encoding; read-only mounts/attributes and
  overlapped/delete-on-close/unbuffered/encrypted shapes fail closed. The port
  and VFS suites pass 62 tests, the focused state/tape slice passes 107, and
  the broad machine/tape/SSA/recompilation/VFS slice passes 215. The current
  `cmd.exe` inventory is 286 references, 184 admitted
  identities, and 151 unsupported. A fresh node-Wasm `cmd.exe /c "echo hello"`
  proof remained byte-for-byte behaviorally stable: exit zero at 16,010 steps,
  499 completions, 15,459 compiled instructions, and `hello\r\n`. It persisted
  16,510 records in 66 segments (8,555,052 bytes), then cold-reopened at the
  exact halt without appending a record.
- Replaced the empty registry stub with a first-class immutable virtual
  registry: predefined roots, case-insensitive key identity with display names,
  typed binary values, access-bearing handles, last-write generations, and
  explicit create/delete/open/close/set/delete-value effects. All eight
  registry functions used by the current image (the prior open plus seven
  unsupported imports) now implement Win32 result, access, enumeration, and
  query buffer contracts without touching the host registry. Registry state is
  serialized in exact JSONL/segmented states, lifted as a distinct trace-SSA
  resource domain, shared across guest threads at journalled core-sync edges,
  and reversed exactly. A fresh real proof exposed `MAXIMUM_ALLOWED` at step
  1,320; the machine retained an amber unsupported annotation, admitted the
  bounded virtual-rights resolution, resumed from that exact request, and
  reached the unchanged 16,010-step exit-zero `hello\r\n` halt. The split run
  still totals 15,459 compiled instructions and 523 fallbacks. Cold reverse to
  16,009 and one forward step reproduced the entire tip, six-root registry,
  and device output exactly. Current coverage is 191 admitted of 286 `cmd.exe`
  references, leaving 144; the broad regression slice passes 219 tests.
  Deleting a key with an outstanding handle deliberately remains fail-closed
  until deferred-delete/tombstone semantics are modeled; the implementation
  does not silently revoke that handle.

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

Chronology alone is insufficient for replay. Dynamic link identities,
runtime-discovered code, request/completion causality, and repair/branch source
states need explicit durable dependencies. Missing link-catalog persistence
caused a real synthetic-target collision; graph validation and durable catalog
restoration now fail closed instead of silently rebinding it.

Persistence should be split by authority. Machine-state segments are the
lossless replay source; trace-SSA segments are a reducible analysis index.
Content addressing lets possible worlds share all pre-fork storage, while a
small mutable tail and one decoded-object cache bound live memory without
weakening rewind or provenance.

## Next Steps

- Serialize guest request argument/effect payloads into the physical shell ABI
  rings and connect native and HTML machine-program hosts.
- Add suspended creation, forced-exit/process-detach behavior, events and wait
  sets, and native first-use delay-helper semantics.
- Continue coherent import families with CRT/process pipes, process lifecycle,
  and bounded exception/unwind capabilities.
- Extend dynamic-memory and indirect-control lowering beyond exact block-entry
  register specializations, broaden call/return beyond block entries, and measure
  when automatic dispatch becomes profitable on real subjects; add embedded
  native emission behind the same journal ABI.
- Lower cached blocks to native code or Wasm while retaining an instruction-level
  journal/provenance witness.

## Prompt History

> can you loop this as a goal and fill things out trying to reach for the sun

> A virtual filesystem in the io shell contract system would be useful if you didn't intuit that

> you should make this ystem tape the whole thing, I don't know if you're doing that already, or for the system calls, but we should be taping this to resume

> put in the system a way to annotate features of moments on the tape, so that things like "this might be wrong" can be marked with a color flag

> add instruction set compatibility notes to RIP on no handler

> we may want to use some graph mechanics to build dependency graphs off the tape if that isn't already figured out so resumes are flawless

> are we retaining the rest of execution as binary or converted graph at all - can we capture a live terminal and start a program in the receptive state and tailor enough to compile that into a website, a working terminal in isolation, can can we correctly convert the tape or use the head to make an ssa version, a more faithful lifting and filtering of all these machine concerns into equivalent math we can use basic reduction to distill fewer operations from?

> I was thinking we would intercept it's calls to exec things and we would load card sets and their executor, you know? like complex programs from our system?

> can we fill a graph we turn to ssa as we use the binary read, and so, like, develop a decision tree rewinding as we please and starting from anywhere, building out many paths concurrently in existence and having had their operations reduced and their machine simulation available to replay - is that something we would end up reaching because of the loops in programming, or does the binary complicate that too much? Can we retain provenance of binary to ssa to track it's provenance every step through raising?

> I would prefer then also that you build in the ability for the program to intentionally branch onto the different read heads, we wanted multi core, right, well, if the program isn't using more than one, or if we keep register caches, you know, and do thread managing, we could make a 100% valid parallel path tracer

> I think I see, so, distinction between threads and forks right? perfect

> are you serious? like, no way, we're tracking the child processes?

> oh, on size, we can store segments, discrete segments of the, you know, ssa branching tape subpaths, we don't have to keep it all in memory
