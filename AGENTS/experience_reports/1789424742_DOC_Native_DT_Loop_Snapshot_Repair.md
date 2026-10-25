# Native dt-system loop snapshot repair

## Work

Resumed the tire/controller compiler effort from the latest continuity ledger
in `turing/docs/PATCH_SEQUENCE_2026-09-07_ALL_19.md`, including the v167 frontier.
The older implementation report ended at v146 and was not the current frontier.

Saved v167 provenance identified the exact bug: the late conditional
continuation pass rewrote the `run_superstep` raw proposal snapshot (274) to
the later conditional cap (565). The fix preserves a uniquely produced,
explicitly selected loop update that dominates its backedge. Ordinary stale
captures still follow conditional continuations. Decisions retain provenance
and incumbent tie policy; the pass remains idempotent.

Also changed structural insertion-list membership to use exact instruction
identity. Repeated read-only stack samples found recursive dataclass equality
at the former membership expression. No physics laws, acceptance tolerances,
compiler optimizations, or existing user changes were replaced.

## Verification so far

- New late-boundary regression failed before the fix.
- Return-state and loop/conditional batch: 42 passed, 72 deselected.
- Adjacent shell checks: 3 passed, 2 failed; both failures reproduced with HEAD
  compiler modules loaded into an isolated process without checkout changes.
- Current compiler replay of trusted source checkpoint v147: six frame rounds,
  4538 formals, three result rounds, zero incompatible contracts and zero
  structural findings. Final snapshot backedge is 274 and recurrence backedge
  is 565; the historical producerless value 328 remains absent.
- Explicit O0 one-step standalone executable: 48/48 public buffers match eager.
- Two consecutive short outer calls also pass 48/48 with persistent state.
- Both full-window executions complete exactly 1/120 second: 169 successful
  native substeps, 167 eager, zero critical or nonfinite attempts in either.
  Strict pointwise buffer parity is still 38/48. The failure and mismatch
  arrays are preserved. No tolerance or acceptance verdict was weakened.
- The running native process's module list contained only its executable and
  Windows/C runtime DLLs. Python was not part of its runtime.

The next numerical boundary requires a fresh causal trace of this checkpoint;
the historical v145 trace does not prove the cause of the current difference.
All processes launched for this continuation are terminal.

The original pre-fix replay was explicitly superseded after saved provenance
and the regression proved the defect. Only that owned process was stopped;
no elapsed-time cutoff was applied. Other active Python jobs were preserved.

## Continuation

The canonical current notes and exact commands are in
`turing/docs/CONTINUATION_2026-09-14_NATIVE_DT.md`. Follow-up acceptance remains
tracked in the companion stub and the existing complete patch-sequence ledger.
No commit or push was made.

## Prompt History

User:
> you could pivot to some old unresolved work on getting dt system to run as a fully native compiled system, a great deal of work was attempted over many days. it's weapon unrelated, I think you'd find it in the tire simulation

User, clarifying outer and inner timestep semantics:
> this would be a strange circumstance where the question would be, why didn't you compile all the things python was doing into the dt system compile, but if dt compile is only made to be modular, the outer tick is meant to be the expected update, and inner processes are allowed any sub-tick. the overall tick and inner ticks I think just consequently work out

Relevant preserved historical instructions:
> YOU CANNOT KEEP USING TIMEOUTS

> if the priority is equal the tie goes to the incumbent
