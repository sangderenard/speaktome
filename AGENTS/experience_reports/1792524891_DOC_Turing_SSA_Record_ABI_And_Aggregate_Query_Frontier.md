# Turing SSA record ABI and aggregate-query frontier

Canonical continuation report:
`C:/dev/Powershell/turing/docs/compiler_convergence/29_ssa_record_abi_and_aggregate_query_frontier_2026_10_20.md`.

The isolated bootstrap `_result_layout` unit now has complete source regions,
complete root parameter accounting, resident `any`, and canonical conditional
aggregate member Phis. Nested Program-ABI tables are admitted as child handles.
The compiler deliberately refuses multi-column `next(iter(layouts))` rather
than dropping the shape column.

Continue in `C:/dev/Powershell/turing` at two exact mechanisms: attach the
declared `SSAValue` child-pool storage to the `Instr.args` handle, then publish
the full two-column row for resident `first`. Do not replace either with an
opaque object, callback, scalar-only load, or source-name special case.
