# Turing dt span migration

Date: 2026-09-19. Repository: C:/dev/Powershell/turing, branch
codex/recursive-reduction-bridge, starting HEAD b74c1874. Changes uncommitted.

## Work and evidence

Implemented the full Metrics/Targets channel and participant span migration,
including live producer/consumer call sites, explicit channel order, presence
masks, controller diagnostic spans, and P/P*C PieceState extents. Preserved the
existing dt state/save/restore mechanism. Read the original handoff and local
compiler guidance. The detailed receipt is
[turing migration report](../../../turing/docs/TENSORIZED_DT_SPANS_2026-09-19.md).

Measured: 78 fast dt tests; six span regressions including four C native cases;
37 runtime/engine/scientific checks; 37 metadata/pure-call compiler checks pass.
Two keyed regression failures reproduce at clean b74c1874. Long lowering was not
run. Publication stores retain two unresolved alias-audit findings. The real
coercion-to-proposal reproducer still reports nine identity findings and an
unsupported reference-evaluator llvm.fcmp.ord. Do not claim full native acceptance
or measured compilation speed from the passing small slices.

A write trace found cross-function local-id collisions: energy-helper bool 68
retyped binary_value's unrelated double 68. Metadata now uses existing source-owner
scope and call edges. Cached tensor reference mutation was observed too; whole
source compilation now owns one copy shared by its regions. Tests preserve the
cached input and run consecutive energy/proposal compilations. Existing pure-call
and entry cleanup now retire an unused coercion receiver after field forwarding.

Cleaned wtp/wtb/wt_head after binary patches and SHA256-verified untracked files
were saved under turing/artifacts/worktree_preservation/20260919. A temporary clean
baseline checkout was also removed after its two checks. Never used contaminated
wtp as clean b8e69910. Preserved prior RAR scheduler edits, the extent audit,
original handoff and backup file.

## Prompt History

> yes, dt system and it's users must adapt to full tensorized format with spans, at some point that was taken out of it, hilariously, many years ago, because it was too slow, now we need it back because it's the only thing that will compile fast

> Two things in there a reader should act on rather than just read: the leftover worktrees need cleaning up, and `wtp` is contaminated — I copied HEAD files into it for the bisect, so it isn't a clean `b8e69910`.

Applicable turing AGENTS.md instruction:

> You never launch the long lowering to find out; the user launches it, when they choose, and you hand them the exact command.

## Next Steps

The migration report contains the exact existing-entry seven-law b64 compile
command for the user. Remaining audit findings, full native extent/physics parity,
and actual lowering time require measurement; no render or speedup is claimed.

## 2026-09-19 retained-loop projection ownership repair

The controller repro's `region dependencies cross atomic control boundaries
cyclically` refusal was traced with the existing
`tools/audit_ancestry_retained_loop_graph.py` and the refusal's own control tree.
The five detached `GetAttr` nodes were synthesized initial record-field states
for `coerce_metrics(metrics)`. They were in the retained loop's invalidation
cone, but lacked the `source_span` that `loop_composer.is_lexical_body_node`
uses to admit synthesized work into the authored loop body. The reducer's
`new_node` API already carries this provenance through `source=`; the initial
record-field seed was the missing caller.

`src/common/tensors/topological_reducer.py` now creates that seed with
`source=body_statement`. A focused loop-composer regression proves the seed's
span and loop membership. The complete abstract-tensor reducer file passed
(65 tests), and the two focused seed/ownership checks passed. The real
`tools/repro_step_with_dt_control_used.py` crossed its former approximately
63-second atomic-boundary failure and remained CPU-active until its dedicated
session was stopped after roughly 50 minutes; the separate whole-compiler
bootstrap process was not signalled. A subsequent audit reached the later,
separate incomplete-source refusal for unresolved `state.restore` and
`ctrl.update_dt_max` effects rather than the atomic-boundary cycle.

## 2026-09-19 identity-log correction

The four `emptied` rows in
`artifacts/compiler_evidence/identity_logs/step_used.20260919T211225.failed.log` are the two
occurrences of
`(() if rejected else tuple(soft_reasons))` at `dt_controller.py` lines 353 and
403. Each occurrence contributes the literal empty `Tuple` and the dynamic
`tuple(soft_reasons)` materializer. A bounded interception of the same entry,
source assembly, and contract reached all four in 4.17 seconds and then
stopped. The literal is legitimately born with no leaves. A second bounded
write trace showed that each materializer is initially assigned its argument's
lexical Name occurrence as `aggregate_leaf_value_ids`. Later input resolution
rewires the Call's `arg:0` edge to the resolved value without updating that
cached ledger. Canonical relabeling retains and maps the resolved source as
`materialized_source_value_ids=(52,)`, but filters the obsolete lexical
occurrence out of `aggregate_leaf_value_ids`, producing the empty tuple in the
log.

The lookups come from `_publish_conditional_tuple_members` while
`_propagate_callsite_tensor_specializations` scans both conditional arms. It
cannot publish fixed per-position Phi members because both leaf lists are
empty, so it leaves this variable-length reporting value alone. Thus the four
rows contain two legitimate empty tuple literals and two stale cached-ledger
transitions. These occur in diagnostic `attempt_log` values that are outside
the specialized `attempt_log=None` runtime path, so this log does not establish
them as the cause of the later compile time. The other 165 rows are broad
aggregate probes, not 165 faults. The interrupted identity book contains no
terminal refusal and no account of where the later runtime was spent. Its
`.failed` suffix came from manual interruption.

Prompt that caused the correction:

> I feel you're inadequately invested in finding reasons

The user then removed the need for that aggregate entirely: `attempt_log`
does not require an immutable tuple. Both records now store the freshly
allocated per-attempt `soft_reasons` list directly. The focused behavior test
passes (1 passed in 0.71s). The full dt-superstep file produced 16 passes and
then remained in an existing long-running case until its owned test session
was interrupted; it reported no failure before interruption.

> attempt log also doesn't seem like it would matter what it got can you make it not require any conversion

The exact documented seven-law b64 link command was then run after this change.
Its owned Python process was PID 19416. It remained responsive and continuously
CPU-active for slightly over 30 minutes, growing to approximately 3.32 GB
resident memory, but produced neither a build artifact nor a refusal. Its only
stdout was the pygame greeting. The session was stopped at the user's stated
30-minute bound. The independent compiler-bootstrap PID 17436 remained active
and was not signalled.

No `llvm_dt_system.*` identity log was produced. The wrapper writes only when
the IdentityBook has pages, and PTY interruption may also prevent normal Python
unwinding, so the absence of a log is not a stage receipt. No success, failure,
or causal attribution is claimed from this run.

> proceed

## 2026-09-19 completed seven-law b64 linked attempt

The earlier 30-minute interruption was not a valid terminal measurement. The
same documented seven-law command was restarted and left to terminate on its
own. It ran from 22:22:08 to 23:45:09 (about 83 minutes), remained responsive
and CPU-active, and reached approximately 6.4 GB resident / 8.7 GB private
memory before exiting with status 1. It was not interrupted.

The terminal refusal is in `precompile_to_ssa._schedule_loop_callsites`:

```text
ValueError: control effect order conflicts with value dependencies
```

The reported ten-position cycle is:

```text
23 -> 8 -> 22 -> 54 -> 33 -> 35 -> 39 -> 37 -> 42 -> 23
```

Positions 4 and 8 are `ConditionalBlock`s, 35 is a `LoopControlBlock`, 39 is
`__plan_callsite_138__`, and the other positions are scheduled numerical
regions. Five links can be identified directly from the printed signatures as
value dependencies: 23→8 through value 382, 8→22 through 402, 33→35 through
524, 39→37 through 138, and 37→42 through 294. The remaining four links were
ordering constraints whose exact categories were not printed by the old
diagnostic. Do not infer their categories solely from the topology.

The complete compiler identity log is:

```text
C:\dev\Powershell\turing\artifacts\compiler_evidence\identity_logs\llvm_dt_system.20260919T234509.failed.log
size: 149070 bytes
SHA256: 6396BFAEF554FC6204E2764E42B32B05CBAE4E07267B4E8B2BD83380C3890FBB
```

It contains five pages: 1527 aggregate-ledger rows, 0 argument-binding rows,
58 cross-function-reference rows, 0 member-formal rows, and 16
tensor-shape-enrichment rows (22 cells). No native build artifact was written;
the refusal occurred during SSA control scheduling before emission.

A bounded compilation-unit-plan capture mapped many of the value ids back to
the current controller source in 12.54 seconds. Among the cycle-relevant ids,
382 is the dt-floor retention conditional, 402 is the
`metrics.control_present[3]` store, 294 is the normalized `dt` tensor, and 524
is the post-write `metrics.control_present` field value. Compact graph ids are
not universally transferable to the full specialization: in particular, node
138 in the compact graph is a string constant while the full refusal names
`__plan_callsite_138__`. Preserve that distinction.

The scheduler refusal now reports provenance for every dependency edge
(`value`, `hierarchy`, `sequence_raw`, `sequence_war`, `sequence_waw`, or
`terminal_guard`) without changing scheduling. Its focused six-test file
passes. A one-law b64 linked diagnostic was launched through the same public
entry to obtain that evidence earlier in the law sequence. Its process and
logs are recorded in `turing/build/one_link_diagnostic.pid` and
`turing/build/one_link_diagnostic.paths`. It must be allowed to finish on its
own.

That one-law run terminated on its own after 475.01 seconds and reproduced the
seven-law cycle exactly. The four previously unidentified links are now
measured: 22→54 is hierarchy rank 49→68, 54→33 is hierarchy rank 68→70,
42→23 is hierarchy rank 28→40, and 35→39 is the terminal guard. Region 37→42
also carries hierarchy rank 25→28 alongside its already sufficient value-294
edge. No sequence RAW/WAR/WAW edge participates. This disproves the remaining
possibility that the current refusal is another sequence-arena dependence
error: flat hierarchy order crosses the retry control boundary and closes a
cycle with the terminal guard and real values.

The exact one-piece pre-deployment graph identifies planned callsite 138 as
`_scalar(ref)` inside the numeric-exhaustion calculation. A second one-law run
was started with the refusal label extended to print the `PlanCall` callee and
the `LoopControlBlock` action/site identity. As with the first run, it must
finish or fail by itself; elapsed time is not a stop condition.

The second diagnostic terminated on its own after 468.54 seconds. It confirmed
callsite 138 is `_scalar`, while the conflicting terminal is `return` site 524:
the fast-lane return inside `if not rollback`. A bounded exact one-piece graph
inspection mapped site 524 to
`return metrics, _restore_type(dt_next, ref), _restore_type(dt_tensor, ref)`.
The retained-loop descriptor still carried that return's predicate before
deployment, so return discovery had not lost its arm.

The concrete scheduler defect was the flat hierarchy chain crossing terminal
controls. All three hierarchy-only edges in the cycle crossed position 35's
return; the one hierarchy edge remaining on one side duplicated value 293.
Hierarchy precedence is now constructed separately for each lexical segment
delimited by `LoopControlBlock`. Value dependencies and the existing terminal
guard continue to order actual cross-boundary requirements. A focused
regression reproduces the old four-block cycle (before-region → return →
source-placed call → after-region → hierarchy back to before-region). The
control-order batch passes 10 tests.

The post-fix one-law compile crossed the old scheduler failure and terminated
on its own after 479.09 seconds at the next gate. Exact refusal:

```text
returned record formal lacks receiver field 'pub_tau'
```

The callee `Metrics` record published `pub_tau`; the caller receiver contained
the scalar/control/channel fields plus `pub_limits` and `pub_limits_present`,
but none of the other participant spans. An enriched repeat identified the
boundary exactly as caller
`step_0__step_with_dt_control_used__specialized_bcc30322fc03`, callee
`step_0__coerce_metrics`, callsite 378. This is lazy record-field
materialization: `coerce_metrics` returns its exact `Metrics` formal, while the
caller has not yet materialized every span that formal carries. The compiler's
fixed-point record-demand pass grows such caller formals, but it runs after
returned-record reconciliation.

Returned-record reconciliation now attaches missing span storage using exact
PlanCall argument bindings when present; otherwise it allocates the ordinary
ABI-keyed caller formal which the later record-demand pass is already designed
to reuse and propagate upward. The refusal remains for incompatible storage or
field shapes. Two existing exact-return/coerce record tests pass. A new
one-piece linked run is in progress and must terminate by itself.
