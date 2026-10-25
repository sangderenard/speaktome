# Turing effect-order repair

Continued from the user's acceptance of the reachability increment and their
fresh 19-finding review. Found the late scheduler's any-ready queue could move
reads and accept effects before earlier blocked control. Added on-demand
prerequisite scheduling, resident-sequence access order, terminal-before-call
constraints, and diagnostic refusal of contradictory cycles. Removed obsolete
flat ranks only for regions proven pure by the existing inventory. Restored
non-loop clear effects omitted by lexical mutation recovery.

The bounded step control replay is clean; native sequence observations match
Python across all eight append/clear combinations. Two broader repros expose
pre-scheduling dropped record writes and dropped call-only loops, retained as
narrow expected failures rather than passing proof. The current full diagnostic
outcome and test measurements are in
`turing/docs/REPAIRS_2026-09-06_EFFECT_ORDER.md`.

Verified the user's optional-field concern: the managed controller really has
dt_min=None, and the feed adapter silently replaced it with a numeric sentinel.
It now refuses None in numeric record fields without optional presence. This
protects semantics but does not implement the missing optional ABI.

The reusable capture/inspection helpers are in `AGENTS/tools/`. No existing
compiler work was reset or stashed; no authored DT code was changed.

## Prompt History

> The next repair should be effect ordering, with the return-merge wiring of 577/593 folded into the record-field step as planned.

> One thing to confirm on the pruning itself: the failure return disappeared because `ctrl.dt_min is None` folded to `False` from the ABI scalar, which makes `retries_exhausted` constant under `max_retries=None`. That is sound only as a declared specialization on the controller instance.

The prior authorization remains:

> begin implementing repairs while taking your own time to verify they are the right move

## Next Steps

See `speaktome/todo/turing_second_opinion_repairs.stub.md`, the Turing effect-order
repair report, and its continuation document. Record returns and optional
presence need independent physical/native proofs; no full parity claim.
