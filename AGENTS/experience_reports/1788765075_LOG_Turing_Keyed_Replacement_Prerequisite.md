# Turing keyed replacement prerequisite

## Consolidated follow-up audit and patch sequence

Traced all 19 baseline findings into
`turing/docs/REPAIR_ACTIONS_2026-09-07_COMPLETE_FRONTIER.md` and consolidated
them into `turing/docs/PATCH_SEQUENCE_2026-09-07_ALL_19.md`. Nine patches cover
every baseline finding exactly once; the tenth includes optional presence,
loop state, return identity, stronger gates, and integrated native parity.
This is an implementation specification, not a completed repair claim.

Working mapping-copy edits remain unfinished. The new native mapping-copy
regression lowers and compiles but returns an empty tuple instead of four
lookup results. The strict gate misses structural output shortfalls. Keep
the regression failing until direct return lookups and their source ordering
are repaired. No new full-source diagnostic was run after those edits; the
19-finding result below remains the earlier baseline.

Additional user prompts:

> please finish the work

> nothing was preventing you from finishing your previous work too, don't leave half made things

> can you trace the nature of every one of the remaining problems into a list of specific actions that can be taken all at once in a single turn

> put in all 19 fixes into one patch sequence in a single turn and then we will shift to working on the fixes

Continued the scalar-return repair toward keyed error_channels ownership.
The existing replacement path cleared before copying, so it was unsafe for
self-aliasing and capacity failure. Added checked row replacement to the
shared SSA sequence lowering and wired existing conditional replacement to it.
Native tests cover empty/self/repeated copies and failure preservation.

Focused4 passed15.68s. Broader42 passed,1 failed41.39s at the existing append
test's expectation that the generated C shim file be empty. The assertion
was retained; that fixture does not exercise replacement.

Saved full SSA also shows missing dictionary materialization and two keyed
stores in the floor branch. The table resolver requires lexical binding names
even for an already declared dict arena; its IndexedStore aliases and lexical
scheduling need a coupled repair. Do not wire an unpopulated local arena into
the returned field or remove the duplicate formals as a shortcut.

Details and fresh diagnostic result are in
`turing/docs/REPAIRS_2026-09-07_KEYED_REPLACEMENT.md`.
The fresh source diagnostic completed with19 formals and zero undefined
operands/unresolved calls, exit1 at strict gate. All four previous scalar
publications remain, including hard_failure1482 ->9564 ->593 and early1378.
All launched jobs are terminal. No full native build/parity or commit/push.

## Prompt History

> continue the work please

> begin implementing repairs while taking your own time to verify they are the right move

## Next Steps

See `speaktome/todo/turing_second_opinion_repairs.stub.md` and the repair notes.
Recover constructor copy and chained keyed stores at their authored control
position, then bind owned-field copy/alias semantics with a real capacity ABI.
Optional presence and authored child-record proof remain necessary for parity.
