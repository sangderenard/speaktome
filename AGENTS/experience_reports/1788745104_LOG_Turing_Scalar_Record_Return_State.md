# Turing scalar record return-state increment

Added exact per-return receiver/field/value receipts in the reducer, with
canonical identity remapping. Physical record return-merge expansion now has
a conservative scalar lookup: one source candidate, one non-formal SSA
definition, matching scalar type, and dominance of the return predecessor.
This changes the field Phi argument, not earlier reads or the global record
layout. Ambiguous receipts retain prior behavior; those paths remain repair
work, not an implied correctness claim.

Whole-source verification caught two missing pieces beyond the first helper:
projection aliasing must remap the receipt with the return slots, and the
Boolean ledger Phis still have provisional float64 result types. Added receipt
remapping and an explicit return-edge conversion guarded by an acyclic
Boolean-leaf Phi proof, preserving the intermediate storage ABI. Both native
dtype variants pass. Source receipts are retained in function metadata for
review. Intermediate diagnostics and the lexical-reducer retry are separated
from the final result in the repair report.

The repeated lexical failure was reproduced deterministically by evicting a
resolved static-reference projection before reusing its symbol. The cache now
checks node liveness and reference identity before reuse; existing reference
factory logic reconstructs an evicted compiler symbol. The repro and related
checks passed after reproducing the same missing-node exception. Return Phis
also retain source record identities so linking can revisit them when callee
field/effect information becomes available, without changing output slot IDs.

Final publication runs after reachability and paired signature cleanup; the
per-round selector alone remained too early. Saved-module replay changes four
step return fields, including hard_failure to Cast(593), with zero undefined
operands and zero changes on repetition. The actual publication pass is now
covered by both native dtype variants, not merely a manually wired Phi.
The final focused group passed 18 tests with one expected failure in 26.38s.
The fresh source diagnostic subsequently confirmed those four publications:
returned hard_failure1482 consumes Cast9564 of ledger593; the earlier bool
region still consumes1378. It completed with19 formals, zero undefined
operands/unresolved calls, exit1 at the strict gate. Repetition changes zero
fields. All launched jobs are terminal; no full native build/parity ran.

The scalar SSA mechanism has a bounded native regression across both branch
outcomes and reused storage. The authored identity-returning child regression
instead exposes existing unaccounted record formals; it remains an explicit
expected failure, not native proof. Keyed field assignment, duplicate effect
Phis, aliasing, and optional presence remain unresolved.

See `turing/docs/REPAIRS_2026-09-06_RECORD_RETURN_STATE.md` for measurements and
the fresh whole-source diagnostic, and the continuation document for next work.
No authored DT changes, full native build/parity, commit, or push.

## Prompt History

> begin implementing repairs while taking your own time to verify they are the right move

> Return-merge wiring is the right next repair. 593 and 577 are in place with correct physical initials, function_exit has one incoming edge, and 496/497 are exact duplicates against fabricated initials.

> The managed input builder constructs the controller with dt_min=None and dt_max=None, so the adapter will refuse the real validator input until optional presence storage exists.

## Next Steps

See `speaktome/todo/turing_second_opinion_repairs.stub.md`. Complete physical
keyed assignment and the authored two-outcome native proof before removing
duplicate effect Phis or claiming the whole record-return repair complete.
