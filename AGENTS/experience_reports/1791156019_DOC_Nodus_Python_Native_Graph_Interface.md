# Nodus Python Native Graph Interface

**Date/Version:** 1791156019 v1

## Overview

Implemented and verified a Pythonic construction/execution surface over Nodus's
native Canvas, ThreadManager, row stack, and optional EdgeTensorFifo semantics.
Work was isolated in `C:\dev\Powershell\.nodus-python-graph` on branch
`codex/pythonic-graph-interface-2026-10-04`; the dirty primary Turing checkout
was not modified. A detached Turing reference worktree provided bridge tests.

## Steps Taken

1. Read workspace, Speaktome, and Turing guidance; relevant reports; Nodus and
   Turing bridge code; branches, PR state, and dirty worktrees.
2. Based an isolated Nodus branch on the open handler-identity work.
3. Added POD ABI access for row programs, FIFO configuration, stable identity,
   configured ports, scheduler mode, and a headless observation barrier.
4. Added `nodus_graph`, a native-backed example, tests, Nodus documentation,
   and an indexed root research report.

## Observed Behaviour

- Value tests exposed contact writer, reader subscription, asynchronous
  observation, and canonical-vs-tool-row identity defects hidden by the former
  quiescence-only smoke. Fixes stayed in Canvas/ThreadManager/FIFO authority.
- Headless canvas creation inherited and could autosave the interactive default
  workspace. It now disables autosave and clears imported state using native APIs.
- Workspace clearing omitted UUID storage; this was fixed with stable-ID lookup.
- Native scalar stack, repeated tick, first token, two-module FIFO, and
  save/fresh-load tests pass. Pointer/scalar mixing and absent FIFO execution
  are explicitly refused.
- Focused Nodus tests passed; GraphIR passed with DLL paths set; detached Turing
  GraphIR/Canvas-KPN tests passed 9/9.

## Lessons Learned

Canonical canvas pairs and per-table tool-row IDs are observably different.
Fallback is safe only after checking the canonical representation because a
boundary self-edge can be a valid but wrong result. FIFO subscription timing is
semantic: configuring storage after subscribing destroys first-token readiness.
CANVAS text is also not the whole persistence artifact; `.gpmod` sidecars remain
part of full table state.

## Final Native Review Update

Independent review found that the sidecars preserve topology, not configured
FIFO specs or live queue state. Reviewed head
`4a9d147fef4a940d426500eec3d5b9f5253b76ef` therefore adds explicit restored-
edge reattachment and refuses unknown or absent FIFO configuration. It also
fixes submission-count/tick-ID confusion in the scheduler barrier, waits through
snapshot publication, propagates stopped/failed waits, rejects duplicate UUIDs,
preserves bindings on failed load, invalidates stale handles after successful
load, and refuses headless construction beside an existing primary canvas.

The Python acceptance executable now reports eight named cases and adds saved
two-module FIFO reattachment (`slots=7`, `top_k=0`, numerical result `21`),
failed-load preservation, stale-handle, and duplicate-identity checks. Combined
bare CTest passed 5/5, including GraphIR and ThreadManager. The former Windows
`0xc0000135` was DLL discovery: existing Python PyTorch 2.5.1+cu124 supplies
`torch_cuda.dll`, `torch_cpu.dll`, and `c10.dll` under `site-packages/torch/lib`.
CTest now prepends that configured directory for each affected child process;
no global PATH or dependency install changed. This proves loader/test success,
not general LibTorch numerical parity.

Full report:
`C:\dev\Powershell\docs\research\2026-10-04\NODUS_PYTHON_GRAPH_NATIVE_REVIEW.md`.
Agent package Library ID: `libfile_69c1c72c635c8191a34d5078451ba200`.

## FIFO and Stack Inspection Update

Continuation head `a3a3986f8a2f97d2c1de7a69e03569de27a8fa89`
adds only contracts already owned by native Canvas/Table code: configured FIFO
spec/unread inspection, publish accepted/dropped results, fixed-size byte-copy
helpers, native `StackDisplay` snapshots with type/element-size metadata, and
the scheduler-published final scalar stack tail. Absence remains distinct from
a configured FIFO. Stack access is read-only and live state still does not
persist.

Tests now cover lossless full rejection, lossy reader advancement/drop,
subscriber/writer identity rejection, invalid calls, and the exact native
bottom-to-top snapshots around `(9-4)*3`. The expanded compatibility selection
passed 6/6; canonical identity compatibility passed 4/4; Python compileall and
`git diff --check` passed. A test-local Torch DLL path was added to the existing
standalone FIFO contract test after Windows loader code `0xc0000135`; no global
environment was changed.

## Next Steps

No required follow-up substitutes for this implementation. Future expansion
should bind existing native edit, scheduler, typed-token, and AbstractTensor
ownership contracts and add round-trip tests before exposing each surface.

## Prompt History

> User requests: 'i would love for you to work on the nodus table and canvas graph system with all its optional fifos and the stack and everything with all its api oddities and work on making a pythonic version of the system so we can work with nodus graphs in nodus and in python for translation ease'. Own this engineering task end-to-end: inspect current Nodus and Turing APIs, graph tables, canvas, stack, optional FIFO and scheduling semantics, existing Python bindings/serialization/translation; implement a Pythonic graph interface faithful to existing authorities with tested native/Python interchange. Start by reading AGENTS.md and relevant .agents/skills, current plans, dirty state, open PRs and existing bindings. Preserve concurrent edits, work in isolated worktree/branch. Use local Codex memories only as context if useful, starting memory_summary.md at runtime Codex home; never read raw sessions or modify memories.
>
> Important previous evidence to REVERIFY against current source: Nodus canonical catalog is authority, CT IDs differ from generic handler IDs; ambiguous Call mapping was fixed by Nodus59/Turing781 proposals (check current integration). Native Canvas, ProcessGraph, GraphIR and SSA representations have different scalar/tensor contracts; don't collapse them. Existing Canvas↔SSA straight-line bridge drops shapes/has control-flow restrictions; GraphIR export may lose identities. Optional FIFO presence, drop policy, slip readiness, skip cadence, stack order, dtype/shape, ownership, effects, stable graph identity and concordance can be observable semantics. Layout can be metadata, but preserve supported semantics or reject unsupported translations explicitly. Do not create a competing operator catalog or unrelated scheduler. Do not flatten optional FIFOs into mandatory queues or silently replace native behavior.
>
> Deliver useful implementation, not only architecture prose. Send early findings and a staged plan after inventory; select a faithful initial vertical slice while retaining explicit unsupported features. Aim for idiomatic Python construction/inspection/editing and native round-trip, using existing C API/bindings where appropriate. Test identity, ports, stack behavior and optional queue configuration, round-trip fidelity and error paths. Expand through existing well-defined contracts; ask parent only for genuine unresolved design choices. If native runtime tests cannot run, provide executable/mock/schema tests and exact limits without claiming runtime parity. Preserve user active orbital, compiler and RGB work; no hardware or lighting actions. Riemann solver remains shelved.
>
> Save source, tests, minimal examples and agent-facing documentation. User wants documentation in root docs/research as well; index new report there. Return exact paths, diffs/commits and verified results. Do not merge or deploy; local implementation first, no publication in this initial task. A separate read-only peer will inspect cross-language semantics and send useful cautions. Report useful results right away, before scratch notes. If work remains, send progress to parent; otherwise return result. If authorization is missing or an action denied, report exact action,target,blocker; do not route around restrictions.
