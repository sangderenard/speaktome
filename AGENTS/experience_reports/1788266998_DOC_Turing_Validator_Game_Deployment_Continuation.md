# Turing validator and game deployment continuation

**Date:** 2026-09-01
**Title:** Compiler dispatch, managed return ABI, and three-product status

## Overview

Audited and repaired generic repository-SSA control partitioning, deployment
planning, and mixed scalar/record call ABI behavior while preserving authored
vehicle Python. Recorded the distinct status of the validator, Living Data Map
game, and validator-participating upgraded game across web and native targets.
The detailed handoff is
`turing/CONTINUATION_REPORT_VALIDATOR_GAME_DEPLOYMENT_2026-09-01.md`.

## Steps Taken

- Corrected branch compartments that overlapped through locationless AST
  singleton nodes.
- Added repository-SSA deployment planning/execution and O3 LLVM closure build
  support using the existing host deployment pool.
- Made deploy/fast work contracts request automatic deployment.
- Repaired ordered semantic output correlation, repeated physical return ids,
  scalar-versus-storage result handling, and aggregate-view pruning.
- Added opt-in structural-output, linked-call, and selected aggregate-callsite
  receipts to the translation decision tree.
- Inventoried current build artifacts and historical qualification reports.
- Ran focused dispatcher/ABI gates; the final selected gate passed 32 tests.

## Observed Behaviour

The managed validator progressed from an overlapping-control failure to zero
unresolved calls. Its last full-native failure was one undefined `ok` boolean.
Diagnostics proved that the source-call semantic contract contained `(bool,
record)`, but the callee's actual physical `Ret` contained only the expanded
record fields. The generic output authority now reconstructs the physical
return in semantic order. Focused tests pass; the complete several-minute
managed lowering has not yet been rerun after this final repair.

The web validator is a runnable O0 one-step worker page, not the complete
assembly machine. The native validator/viewer is runnable O0 and historical
reports reach 18/19 and 19/21 passing stages, but no report is overall passing.
The native viewer itself is still a non-change dispatch failure: Space toggles
a flag on the sole SDL/render thread, that thread calls the entire synchronous
physics step before polling or swapping again, the GLSL path has no compute
dispatch, and the new repository dispatcher is not wired into the executable.
Consequently a stuck frame yields neither responsive pause nor a new display
snapshot.
The last generated web game predates this work, the nominal current game build
directory is empty, and no native self-referential-map game exists. The
validator rig is integrated into the neutral web game model as a participating
object, but neither a current web upgraded-game artifact nor an integrated
native upgraded-game artifact is complete.

## Lessons Learned

A viewer that appears frozen can be blocked before any adaptive-dt statistics
exist. Control partition overlap, source-call linkage, physical return ABI,
O0 execution cost, and genuine subdivision need separate receipts. In this
case the evidence stopped at compiler ABI failures; it did not show infinite
metric subdivision.

Semantic output order must be the return ABI authority. Expanding one record
cannot replace its scalar siblings, and a scalar result must not be renamed
through a tensor storage-alias ledger. Deployment manifests also are not proof
of parallel execution; the final host must invoke real pool work and join it at
the frame boundary.

## Next Steps

See `todo/1788266998_validator_game_deployment.stub.md`. In short: rerun the
complete managed gate, compile the whole closure through O3 LLVM deployment,
wire that generic ABI into the complete scientific viewer, add on-screen
substep telemetry, qualify/soak it, rebuild the current web game, then create
the integrated upgraded-game products.

## Prompt History

> “DONT MODIFY THE PYTHON SOURCE”

> “FIX THE COMPILER DISPATCHING”

> “THE FAILURE IS THAT DISPATCH DEFINITELY SHOULD FIND PARALLEL WORK IN THE EXPANDED CODE”

> “the deployer should be such that it can tolerate trivial internal closures”

> “an actually optimized release version of the entire validator rig not just the balloon physics”

> “update documentation, translation trouble tree (add any diagnostics you created), give me status on the web and the native versions of: the validator ... the game ... AND the upgraded game”
