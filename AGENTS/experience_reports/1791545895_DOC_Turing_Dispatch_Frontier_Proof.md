# Turing dispatch frontier proof

Date: 2026-10-09. Compiler baseline: `54c67fbd`.

The user authorized stopping a large orbital compile unless it served the
graph compiler architecture. The N=35 probe PID 24648 was verified by its
exact command and stopped. It ended in deployment instantiation after
2,417.556 seconds wall, with recent RSS around 20–23 GB. Completed piece
caches were retained; there is no completed combined N=35 artifact.

In the live compiler, `_dispatch_subgraph` rebuilt semantic parent and child
maps for every region. `glsl_deployment_strategy` now derives one immutable
index per carving phase from the same graph edges and operand references.
Each region consults its members and incident neighbours. The index is a
temporary derived view, holds no new authoritative identity facts, and is
released at phase completion. Existing region/book receipts remain the
identity authority. A wrong-source index is refused.

The tiny probe `tools/compiler_probes/probe_graph_region_frontiers.py` invokes
`lower_ast_source_to_ssa` on an authored repeated-law program, emits through
the repository's existing artifact route, and runs the result. Two, four,
and eight calls pass exact parity. Four calls replace 164 per-region adjacency
queries with 31 index queries; eight replace 556 with 47. Region extraction
makes zero source adjacency queries after building the index. Weak references
verify all indexes are released at lowering completion. Four/eight-call
emitted C hashes are identical before and after. Structural concordance is
zero; the existing 148/248 unsourced facts remain OPEN and unchanged.

The scheduled-dispatch test file gives 29 passes / 4 failures, versus
27 passes / identical 4 failures on the clean `C:\dev\tg-base` worktree.
The new hidden-edge/frontier, traversal-bound and source-ownership checks
pass. Adjacent copy-on-read, kernel deduplication and SSA dispatch tests
give 18 passes. Details are recorded in `turing/TEST_BASELINE_AND_HAZARDS.md`.

This proves one shared immutable graph boundary in the actual compiler.
Full region copies, whole-function residency and serial book writes remain.
No whole-system bounded-memory or parallel-compiler result is claimed.
Existing dirty files were preserved. No packages were installed.

## Prompt History

> enjoy the breadth of the most advanced model to architect a more purely graph refined compiler

> how is paying attention to a serialized unoptimized process instead of architecting a pure graph form from first principles serving you or me

> if the large compile is paying genuine dividends in making the pure system, let it run, elsewise, kill it so you can more swiftly iterate on a very small task to prove the architectural expansions in place
