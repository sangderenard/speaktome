# Reversible machine demo and graph-crop continuation update

**Date:** 2026-08-05
**Repositories:** `C:\dev\Powershell\turing`, `C:\dev\Powershell\speaktome`, and the root coordination repository
**Goal status:** active; demo vertical slice proven, general machine incomplete

## Assurance checkpoint, not a stopping point

This report was written on the user's request as a durable assurance checkpoint.
Work continued after the report was created, so the named worktrees may contain
later changes, tests, generated site files, or commits not reflected word-for-word
below. Treat this document as a resumable lower bound. Inspect Git status, the
latest commits, and the generated bundle before changing anything.

The earlier full handoff is
`1785935145_DOC_Reversible_Binary_Machine_Continuation_Handoff.md`. Read it first
for the architecture and pre-pipeline chronology. This update records the work
after that checkpoint.

## Outcome at this checkpoint

The actual Windows AMD64 `cmd.exe` machine now completes this capability-gated
pipeline from a real binary instruction stream:

```text
echo data | hello-card alpha
```

It halts with exit code 0 and publishes:

```text
[card-set:hello-card] alpha
```

The proof tape is:

`C:\dev\Powershell\turing\build\cmd-pipe-card-v3-20260805.segmented-tape`

The successful future was created by rewinding to tape position 34614 (a real
failed `HeapAlloc` request), servicing it with the corrected page-managed heap,
and allowing the executor to form a new branch. The old failed future remains
in the graph. Final proof: 42,526 records, guest step 41,001, exit code 0.

A history-free graph crop of that exact successful tip is:

`C:\dev\Powershell\turing\build\cmd-pipe-card-success-crop-20260805.segmented-tape`

It has one position-zero root, preserves the complete state, subject and link
catalogue, and carries a content-addressed origin receipt. It loads without the
source history and correctly refuses to step backward beyond its new root.

## Intended feature matrix

| Intended feature | Status | Evidence / limitation |
|---|---|---|
| Actual subject compiled/loaded as binary executor | Working for Windows AMD64 PE | Real system `cmd.exe` runs; not arbitrary ISA/OS coverage. |
| Binary read head forward and backward | Working | Hot and cold tape reverse/forward; HTML controls now expose both directions. |
| Exact reversible tape | Working | Registers, memory, VFS, registry, devices, external completions, annotations and branches are journalled. |
| Segmented, disk-backed subpaths | Working | Content-addressed bounded segments and cold hydration. |
| Intentional forks versus threads | Working foundation | Rewind creates graph branches; virtual threads/cores are distinct. Broad guest scheduling remains partial. |
| Graph crop / instantiate mid-run | Working | `SegmentedMachineTapeStore.crop()` creates independent position-zero roots with origin receipts. |
| Multi-head / multicore | Partial | Multicore state, thread spawn and barrier mechanics exist; real `cmd` proof is principally one guest core. |
| Parallel path tracer | Partial architecture | Branch forest/provenance exists; automatic exhaustive path scheduling and reduction are not complete. |
| Binary-to-SSA provenance | Partial, useful | Trace SSA retains tape/instruction/effect provenance including pipe domain; whole arbitrary-binary lifting and mathematical reduction remain research work. |
| Recompiling VM | Working bounded tier | Safe prefixes execute through authenticated Node/Wasm journals; indirect/control boundaries fall back or fail closed. |
| Instruction compatibility notes | Working | Missing semantic handlers annotate RIP and ISA identity. Arbitrary AMD64 coverage remains incomplete. |
| Register memory as contiguous chip cells | Working display ABI | All 54 scalar observation cells are fixed-stride and individually visible. |
| Page-aligned memory visualization | Working in snapshot/demo | Snapshot output now publishes 4 KiB page index, byte occupancy and executable/image/managed flags. Heap subobjects remain realistically sub-page. |
| Finite bootstrap memory span | Working | Fresh system arena doubled from 1 MiB to 2 MiB. Overflow uses bounded, page-aligned managed virtual allocations. |
| Subject output shown beside machine | Working | Terminal/framebuffer output descriptors feed the interior shader. |
| GLSL authored in Dream Document | Working | Compute and fragment blocks live in `reversible_chip_simulator.dream`; WebGL2 fragment owns the display. |
| GPU active indicator | Working | Fragment lamp and canvas state indicate snapshot-driven GPU updates. |
| HTML-shell interactivity | Working vertical slice | Pause, forward, reverse, one-step each way, speed slider, register HUD, memory pages, terminal and binary file parameter are document-owned UI. |
| Keyboard to guest shell | Working | Accessible command field plus shader-surface key editing; Enter posts CRLF to `/input`, then becomes a taped `console.input` effect. |
| Browser binary file parameter | Working host seam | Declared `subject-binary` system port posts byte-exact contents to `/subject`; controller replaces the machine through an admitted factory and starts paused. |
| Generic HTML shell, not bespoke page | Working | Emitted with `emit_dream_html_shell()` plus the generic snapshot liaison. UI remains Dream blocks. |
| Publishable bundle | Working local/site artifact | `index.html` and `bundle.json` generated from the Dream Document. It currently requires the Python loopback runtime for live execution. |
| Fully standalone browser executor | Not complete | Python machine runtime has not yet been compiled into the page/WebAssembly. Static HTML alone renders only when given snapshots. |
| Native shader viewer | Working earlier slice | Pygame/OpenGL viewer has registers, output, direction and speed; page-band parity with HTML should be added. |
| Virtual filesystem | Working foundation | Memory VFS plus IndexedDB/OPFS HTML mounts. Native host-directory admission remains capability-gated. |
| Virtual registry | Working bounded Windows subset | Exact reversible registry state and principal `cmd` operations. |
| Pipes and CRT descriptors | Working demo tier | `_pipe`, duplication, close, EOF, inherited child stdio, remote child handle close and tape/SSA effects. |
| Child process tracking | Working for declared executors | Virtual process handles, exit state, deployment metadata and child tapes. No ambient host `exec`. |
| Shell/card interception | Working | Pipeline child `cmd` built-in trampoline and registered `hello-card` executor. General shell grammar/built-ins remain incomplete. |
| External linked binaries | Partial | PE dependency acquisition/import binding and provenance exist; arbitrary system DLL execution/automatic lifting is not complete. |
| System activity shunting | Partial | Explicit capabilities cover current command tier. Unsupported shapes remain blocked. |
| System devices | Partial | Console and typed device buffers work; broad Windows device virtualization is unfinished. |
| IndexedDB/OPFS | Working browser contract | Hydration, persistence and Chrome proof exist. |
| Clock / externally regulated ticks | Working | Free spin, exact transition ticks, direction, and adjustable speed exist. |
| Tape annotations / colored flags | Working | Features such as unsupported calls and verified exits can be marked and rendered. |
| Dependency graph for flawless resume | Working core | Parent/dependency edges, lineage validation, checkpoints, branches and crop receipt. Reachability pruning of child assets can be improved. |
| Programmatic differentiability / graph mathematics | Partial | SSA/effect graph enables analysis; full differentiable semantics for arbitrary machine behavior is not claimed. |
| Cross-language comment sentinels | Working foundation | Dream Document segments, parallel declarations and in-place shader blocks lower to deployment-aware IR. |
| Common shader/shell ABI | Partial | Snapshot and shader component ABI is shared in design; exact WebGPU compute parity and all native ports remain unfinished. |
| C shell around Fortran / other bundle targets | Existing compiler capability, not integrated demo proof | Do not conflate the HTML demo with completion of all shell targets. |

## New implementation details since the previous report

- Fixed Windows case-insensitive directory resolution at the system-port edge;
  the common VFS intentionally remains case-sensitive.
- Added `lstrcmpW` through the obsolete Windows string API set.
- Added a capability-gated virtual `cmd.exe` built-in trampoline and direct
  redirection of non-built-ins to the declared virtual program registry.
- Carried inherited CRT descriptor bindings into child standard streams even
  when `STARTF_USESTDHANDLES` is absent, matching the observed pipeline shape.
- Added virtual-child `DuplicateHandle(DUPLICATE_CLOSE_SOURCE)` semantics and
  masked Win32 DWORD/BOOL stack arguments to their architectural width.
- Added deterministic heap reuse capacity, page-managed overflow, and the 2 MiB
  fresh bootstrap arena.
- Added `--rewind-to-position` to the real `cmd` probe.
- Added graph cropping with source manifest, subject, source sequence/position,
  and state digests.
- Added generic `/control` and `/subject` HTML-host contracts, bounded queues,
  owner-thread mutation, binary replacement, and Dream-owned controls.
- Added memory-page snapshot output and GLSL visualization.
- Added bundle publication support to `reversible_machine_web_host.py`.

## Demo/site artifacts

Build preview:

`C:\dev\Powershell\turing\build\reversible-chip-demo-bundle-20260805`

Actual tracked/live site location (corrected after this assurance checkpoint):

`C:\dev\Powershell\site\programs\reversible-binary-machine\versions\v1-528d10a4d394e4a5`

The bundle is generated, not hand-authored. It now uses the common immutable
`turing-program-bundle-v1` publisher and root gallery scan. Its artifacts include
the Dream source, the authored PE generator source, and the exact 2,048-byte
AMD64 PE subject. The page embeds two authentic executor-produced `TMSNAP01`
frames so GitHub Pages supports forward/backward/pause/step/speed replay without
a Python process; the native owner remains responsible for arbitrary binaries,
terminal execution, system ports, and unbounded tape operation.

Launch command after checkout:

```powershell
python examples/reversible_machine_web_host.py `
  --tape build/cmd-pipe-card-v3-20260805.segmented-tape `
  --machine-backend node-wasm --demo-card hello-card --open
```

For a fresh interactive `cmd.exe`, use a new tape and omit `/c`; the browser
terminal then feeds the receptive shell. Build tapes are ignored artifacts and
are not expected to be in Git.

## Verification at this checkpoint

- `165 passed` across snapshot host/state, HTML shell, shell IO, system ports,
  segmented tape, and related suites.
- `26 passed` for focused AMD64/heap semantics.
- `253 passed` in the earlier broad machine selection before the newest demo
  changes; rerun the broad selection after the final commit.
- Real pipeline: guest exit 0 at step 41,001.
- Graph crop: position 0, same halted state/output, no ancestor available.
- Chrome screenshot: generated WebGL2 shell rendered successfully under
  SwiftShader; an early screenshot caught and led to correction of control-row
  layering.

## Next-agent checklist

- [ ] Read this report, the prior `1785935145` report, Turing `AGENTS.md`, and
      the three reversible-machine docs before editing.
- [ ] Inspect all three Git statuses. Preserve unrelated dirty changes.
- [ ] Check the commits made immediately after this assurance document; work
      explicitly continued after the report was written.
- [x] Confirm the root versioned bundle is generated from the current Dream
      Document and that `bundle.json` subject/shell digests match.
- [ ] Run the generated shell with the loopback host and test pause, forward,
      backward, both single steps, speed, keyboard Enter, and file replacement.
- [ ] Add a deterministic browser integration test that waits for one snapshot
      without being confused by the intentional long poll (`snapshot-once=1`
      was introduced for this seam).
- [ ] Make the control status report the current core position/step as well as
      total flips.
- [ ] Add page-occupancy parity to the native OpenGL viewer.
- [ ] Decide whether browser subject replacement should start paused at the PE
      entry or optionally accept a cropped state bundle.
- [x] Add a tracked project-authored AMD64 demo subject and executor-produced
      finite replay; no Microsoft system binary is published.
- [ ] Keep unsupported host execution fail closed. Do not add ambient `exec`.
- [ ] Continue compatibility work only where it blocks the interactive demo or
      a named program proof.

## Known risks / honest limitations

The published HTML now has an interactive finite executor replay, but it is not
yet the full Python machine compiled into browser Wasm. GitHub Pages cannot run
the loopback Python owner. Arbitrary subject replacement and terminal/system
activity therefore remain native-owner features. Binary replacement uses the
explicitly admitted endpoint and starts a fresh in-memory tape; it does not
overwrite the canonical proof tape.

The worktrees contained unrelated edits before this session. Commits must stage
only the reversible-machine and documentation paths. Never reset other work.

## Post-publication continuation

Work continued after the original report and first publication. The native
OpenGL viewer now consumes the shared page-occupancy channel, opens segmented
proof tapes, provides forward/reverse/single-step function-key controls, and
journals typed commands into `console.input` without racing its free-spin owner.
The root bundle now includes a provenance-bound machine-block Wasm artifact,
packed state/guest window, WAT and dispatch plan. Chrome executed the subject's
lowered `0x90` entry instruction and visibly reported `WASM BLOCK ·
AUTHENTICATED` after checking the journal witness. The next continuation
connected register-only Wasm journals to the common snapshot transport.
Chrome's deterministic `?recompiled-step=1` proof shows RIP advancing from
`0x140001000` to `0x140001001`, the steps register changing from zero to one,
and the HUD reporting one flip. Reverse returns to the initial snapshot.
Memory/device-effect journals deliberately keep retained frames until their
browser commit path exists; arbitrary browser-owned loading and capability
dispatch remain incomplete.

## Prompt history

> "you have about an hour of tokens left, I want you to make continuation and experience documentation in speaktome that allows an agent to resume your work regardless of where you left off, explaining that your documentation was done for assurances but your work continued, and then continue on your work please until tokens run out"

> "I am not opposed to some finite doubling of the container memory span if our infrastructure and syntax allows it"

> "would you be willing to make it aligned memory in pages so the memory display shows actual hardware alignment areas?"

> "put in architecture while you're doing this for obtaining and beginning fully fresh from a graph crop, shedding the old tape and existing as if the program just intantiates mid run"

> "prioritize the seams that will allow us to generate a published bundle with interactivity over absolute completion of program potential, you may have another hour, so I want you to focus on demo potential"

> "keep in mind you are to make this runnable as the html shell not a bespoke page"

> "the is the keyboard linked to input, the user must be able to use the shell for the demo to be complete"

> "create a continuation update report, list all intended features and their status in the build, a checklist for thnext agent with a note you will continue on after writing the document so some lingering changes might exist, I ant you to commit and push turing, speaktome, and the root after making the page and putting it in the root site bundle folder so we can try it live even if it's not finished. then you may continue and attempt to exhaust your credits"
