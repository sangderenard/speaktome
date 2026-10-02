# Turing second-opinion repair entry

Read the workspace guidance, Turing test hazards, prior DT receiver-identity
report, continuation brief, and second-opinion review. Preserved the existing
dirty worktree. Started general compiler repairs with conservative scalar CFG
reachability and predecessor-aware Phi pruning, integrated before the existing
signature transaction. No authored validator workaround or formal allow-list.

The saved-SSA replay reduces unaccounted formals from 23 to 18, not the review's
predicted 16. Calls already hoisted into reachable blocks cannot be deleted
merely because a later Boolean result is constant. C backend inspection also
disproves the tentative inference that scalar dtype means by-value native ABI:
internal formals are opaque storage pointers. Advance-result storage identity
still needs its own proof.

Validation and the current frontier are recorded in
`turing/docs/REPAIRS_2026-09-06_SECOND_OPINION.md` and the continuation document.

## Prompt History

> begin implementing repairs while taking your own time to verify they are the right move

The preceding user message supplied the second-opinion review summary, with
23 formal findings, silent scheduling/record defects, proposed general repairs,
and an explicit caveat that the native ABI had not been inspected.

## Next Steps

See `speaktome/todo/turing_second_opinion_repairs.stub.md` and the Turing repair
document for uncompleted effect, record, producer, and parity work.
