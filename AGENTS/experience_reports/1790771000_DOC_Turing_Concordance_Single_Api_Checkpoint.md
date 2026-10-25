# Continuation: the concordance becomes a causal flow graph (checkpoint 2026-09-30)

Written at the user's request as a model-change checkpoint. Read this, then
`turing/docs/CONCORDANCE_SINGLE_API_DESIGN_2026-09-30.md` (the decided spec)
and `turing/docs/concordance_census/00..70` (evidence and the two plans).
Compiler numberings are ephemeral: none appear here; never write them into
docs.

## Standing rules (user, verbatim intent)

- The concordance is the graph realization of all morphing. Every value that
  has ANY known source gets an edge from that source at the moment it is
  written. A fallback chosen from absence of information is recorded as
  unresolved, never as a fact. A raise on disagreement means an edge was
  missing; never call it the concordance's fault.
- One api: a post is admitted only as NOVEL (minted id + the transform edge,
  so later stages can re-identify it if topology changes) or DERIVED (exactly
  which concordance cells it came from). "Actual causal flow graph, not a
  spine of definition clusters."
- Decisions taken: source reference is the full vector (page, row, column);
  Unsourced is a TYPE behind a LATCH (open now, closed once everything is
  fixed); no strings decide anything at runtime (pages/stages/transforms/
  reasons/modes are registered objects); steps 1-3 are worked together.
- Process rules: no builds or long lowerings without the user's word; one
  compile at a time; seconds-long probes; `git grep` only; dispatch Fable
  workers in parallel on disjoint files, read-only where possible; commit
  locally, push only when told; do exactly what was asked, nothing more.

## Where the line is

turing `main`, local, NOT pushed since `5abab673`. Commits today, newest
first: step 1 of the api (this checkpoint), design + four decisions
(`7a678092`), census + design (`a17b3dde`), scalar-deferral removal
(`92744b87`), dt-system contract fix (`6f3d667d`), and earlier today the
dtype authority, book-backed struct/union tables, struct intake, the
forwarding-edge fix, the frontend path trace.

### Done

1. **Step 1 landed.** `IdentityBook.post(page, row, fact, *, stage,
   provenance, mode) -> Ref` in `src/compiler/identity_concordance.py`.
   Derived writes edge + reverse index in one clock tick; Novel mints (or,
   for a root row, writes the origin edge only); Unsourced is latched
   (`book.latch`, OPEN). Registry of Page/Stage/Transform/Reason objects.
   Every raw primitive write is auto-tagged `Unsourced(RAW_PRIMITIVE)`.
   `record_shape_transformation` goes through `post`. Read api for viewers:
   `book.registry.pages`, `edges_into`, `edges_out_of`, `mint_of`,
   `unsourced_rows`, `latch`. Audit: `unsourced-fact` /
   `unsourced-identity` on a separate report line; the six audit cases'
   first lines are unchanged. Worklist (facts/identities): view 3782/75,
   toplevel 5606/132, energy 3829/105, controller 8211/141,
   controller_untyped 5359/140, mapping 153/15. Private page names:
   `concordance_edge`, `concordance_dependents`, `concordance_mint`,
   `concordance_unsourced`.
2. **Census** (six read-only scouts): ~265 write sites, 36 with an edge,
   ~120 bare, ~45 fallback-as-fact, ~100 mints with no transform, ~60
   unaudited pages, roots not on the book. Census 50 follows
   `Metrics.hard_failure`: 42 facts, 2 edges; the dt_min-floor `False` write
   is lost because the reducer publishes the field value under the SetAttr
   node id while `lower_conditional` looks the arm up by the RHS id and
   takes the pre-branch snapshot as the arm.
3. **Plans** for step 2 (`60_plan_step2_ingestion_roots.md`: 20 pages, 27
   writer sites, root row = `source_span (module, qualname, ast path)`,
   canonical relabel DERIVED not NOVEL, top risk = `identity_table` mutated
   after reduction by planner/loop_composer/autograd) and step 3
   (`70_plan_step3_reducer_field_state.md`: page `REDUCER_FIELD_STATE`
   keyed (scope, receiver, field) in REVISE mode, the exact phi-arm cell,
   12 Unresolved reasons for `scalar_return_field_versions`, 8 `_set_operands`
   callers; top risk = Refs entering cloudpickle digests and the host-code
   cache pickler).
4. **Scalar-deferral rule removed** (`92744b87`, with
   `docs/DECISION_scalar_expression_deferral_2026-09-30.md`): annotated and
   defaulted scalar parameters now compile; 14 programs equal CPython
   natively; audit identical. Baseline proved it is NOT the cause of the
   dt-system failure below.

### Landed after the report was first written

- **Viewer committed** (turing, `tools/view_identity_concordance.py`, first time
  tracked): real edges via the read api, `--diffuse`/key D diffusion mode,
  `--infer-edges auto|on|off`, HUD. With the latch OPEN 153/162 mapping rows
  are raw-tagged unsourced, so diffusion over real edges is nearly empty until
  steps 2-3 land; heuristic inference stays on by default until the latch
  closes. Shots under turing/shots/ (scratch).

### In flight when the limit hit (now resolved above)

- **Viewer lane** (Fable) extending `tools/view_identity_concordance.py`
  (UNTRACKED local file; the pygame/OpenGL globe): real DERIVED/MINT/
  Unsourced edges via the read api, a diffusion colour mode (`--diffuse`,
  key D) for history/consequence flow from focused nodes, HUD with latch
  and edge counts. Its report was not received. Check the file's state
  before trusting it; the user's two example commands are
  `python tools/view_identity_concordance.py --case mapping --list transformation_decision`
  and `... --focus "#117" --focus "id 3  scope=root" --depth 5 --out shots`.

### Next (in order, each one commit + seconds-long probe + audit tool)

1. Review the viewer lane's edit; commit `tools/view_identity_concordance.py`
   (first time it is tracked) if it is sound.
2. Execute step 2 from plan 60 and step 3 from plan 70 (the user chose to
   work 1-3 together). Plan 70 needs plan 60's root rows first. Proof:
   `probe_annotated_scalar_parameter`, `probe_struct_intake`, a new tiny
   dataclass-field-on-one-branch probe (plan 70 section 5), audit tool with
   the `unsourced` counts dropping.
3. Then design steps 4-8 (planner structure, control builder environment,
   return versions, frame linker, tables), closing the latch at the end.

### Held for the user

- Plan 70's finding: the reducer's `ast.For`/`ast.While` branch never
  snapshots or merges attribute state (a loop body's last write is the
  post-loop state). Proposed `Unresolved(LOOP_EXIT_FIELD_STATE_UNMERGED)`.
- Plan 60: function addresses as MINTED ids or not; `identity_table`
  post-reduction mutation (fix belongs to steps 4/5).
- Spec silence: `Unsourced` with `Mode.REVISE` is admitted while OPEN (raw
  `revise` callers need it); confirm.
- Everything in `docs/PARKED_2026-09-29.md` section 7 (laid-out-type
  direction confirmation, N11 spelling, union U1-U5, etc.).

## The dt-system failure (unchanged, diagnosed)

Native lowering of `examples/llvm_dt_system.py` over the drift piece
(`lowered_system`, ~10 min) fails in `materialize_record_phis`: the
return-merge arm for `hard_failure` first takes the record descriptor's
value, then a later round's `scalar_return_field_versions` substitutes a
conditional-carried phi (both arms = initial value; see census 50 section 2
for the lost write) cast to the field dtype, and
`_concord_record_return_phi_inputs` raises. Same on the baseline with the
scalar rule present. It became reachable when `rollback` stopped being a
folded constant (`bc34adba`: `dt_system_over` passes `state.rollback`;
`_propagate_callsite_planner_specializations` had folded the omitted
argument to its default). Steps 2-3 are the fix at the identity; do not
patch the merge check.

## Scratch state to know about

- `C:\Users\alber\AppData\Local\Temp\wtb2`: git worktree at `b8cd8c2a`,
  idle; remove with `git worktree remove` when done. `wtb` is an older
  worktree with someone's uncommitted test edits; leave it.
- `.venv` lacks PyYAML; probes run with the system Python 3.11.
- Root repo (`nogodsnomasters`) and speaktome are clean apart from this
  report.

## Second checkpoint (later on 2026-09-30): steps 2-3 landed

Committed in turing (local, not pushed): declarations module `ca79244b`,
design correction `7bde183c`, REVISE-rule change `90fc38b0` (a different
set of derived cells is a cause), and steps 2-3 in one commit (see its
message for the full list). Vocabulary for steps 2-3 lives in
`src/compiler/concordance_declarations.py` (objects, no strings).

State: annotated-scalar, struct-intake and the new
`tools/compiler_probes/probe_branch_written_field.py` probes pass (the last
prints field schema -> OBSERVED/WRITTEN/MERGED cells -> exit state -> SSA
field versions with their edges); all six audit cases lower; first lines
identical except the generic `[unsourced-fact]` group counts; unsourced
counts baseline +2..+21 (residual = `new_node` callers with no source, the
next worklist). Lane C reported after the commit: its edits are in
`12a14051`.

Lane C's two "not as spelled" items: the twelve return-version Reasons are
DORMANT until a return-merge-Phi identity cell exists (step 6, S18: selection
rows read `attributes["identity_cell"]`, absent today); the record-return
Cast is not yet minted NOVEL (no declared page carries a VALUE_ID for it; the
selection row records the conversion). Two more pre-existing defects it
surfaced: a function returning a tuple with a record is rejected by the
execution contract, and an `if` arm holding only a scalar field write is
dropped so its Store lands unguarded (the honesty path now reports it as
ARM_VERSION_MISSING).

Findings to act on next:
- Lane B's section-7 scratch programs fail the full-native contract with
  unaccounted formals at `ca79244b` too: pre-existing, the class of defect
  steps 2-3 feed; the linker does not yet consume field-state cells.
- The single-exit rewrite upstream of the reducer turns two returns into one
  `return name`, so a record receiver is not a return-slot value; the field's
  state at the exit is the MERGED cell on the exit phi (plan 70 section 2.3
  assumed per-return sites).
- After a loop that wrote a field, the post-loop cursor is Unresolved
  (LOOP_EXIT_FIELD_STATE_UNMERGED); a later read gets no after_write ordering
  operand. Held for the user with the loop-merge question.
- Not on the book yet: `lexical_read_binding` (undeclared page, 24 raw rows
  per case), `identity_transition` (so `_set_operands` NOVEL posts are not
  yet possible), `selected_class_identities`.
- Test to update per plan 60 E13: `test_process_graph_function_linking.py::
  test_callable_dataclass_field_preserves_function_identity` expects history
  `(0,)` and two rows (ingestion + canonical).
- Scratch worktree `Temp\wtb2` now at `ca79244b`; remove when done.

## Third checkpoint (2026-09-30, later): plans 4-8, two diagnoses, six decisions held

turing commits since the second checkpoint (local, not pushed): `2acc155b`
shape re-resolutions derive from their edge (controller: cells with a
DERIVED edge 38% -> 53%; plan 60 E13 test edited, not run); `74b8b219`
plans 80 (steps 4-5) and 90 (steps 6-8) + `probe_scalar_write_only_arm.py`;
`65d62903` `probe_record_in_tuple_return.py`. The user's other session
committed `658daed6`, `5c48c89a` (viewer: --drift, oscillator audit case,
mass toggle) and `42b99e87` (AbstractTensor comparisons) on main in between.

Record completeness (measure_completeness.py in the session scratchpad;
re-create from its description if lost): per case, every cell is edged,
minted or tagged (silent = 0); cells with a DERIVED edge 23-53%; unsourced
55-68%; 100-125 pages written, 20-24 declared; MINTED SSA ids with a mint
record: 0 (steps 5-7). Api adoption (static, src outside the book module):
48 post sites vs 107 raw-primitive writes vs 130 mint sites.

Two pre-existing miscompiles diagnosed to their pass, repro probes committed,
NOT fixed (the user decides):
- Dropped scalar-write arm: `_ordinary_conditional_control_programs` counts
  its retention reasons before scanning the reducer's field-state merge
  phis, so `if c: m.f = x` (arm = one field write) gets no ConditionalBlock
  and the write is appended to the enclosing arm; before steps 2-3 the Store
  landed unguarded, now the control builder refuses
  (carried-field-arm-missing). Fix: the merge phi (source_conditional_id ==
  this conditional, field_state_arms) is a retention reason.
- Guarded tuple return: `_normalize_top_level_guard_returns.result_assignment`
  assigns the whole tuple to one single-exit name; arms bind aggregates with
  no SSA producer; the merge is promoted to an unnamed formal. Fix: split per
  lane as `_normalize_direct_tail_recursion.ExitReturnRewriter` does.

Six decisions HELD for the user (recommendations in the chat log and plans):
plan 80 -- per-copy scope for forked callee specializations (recommended);
refuse silent snapshots for name-carried arms (recommended). plan 90 --
CELL_SET rows for many-source posts (recommended); record, not raise, on
argument_binding's storage-from-absence (recommended); RESIDENT_CHOSEN_BY_ORDER
recorded as its own reason (recommended); latch closes only when both generic
findings are zero on all seven audit cases and the audit tool exits on them
(recommended). Also open from earlier: loop-exit field-state merge.

Next: user's word on the six; then execute plans 80/90 with function-level
ownership splits (steps 6 and 7 both live in fortran_c_shell.py); the raw-only
page inventory (`75_...md`, lane still running at this checkpoint) gives the
declaration blocks.

## Fourth checkpoint (2026-09-30, model switch imminent): lanes in flight

Decided by the user (design section 7, commit `4249738b`): every callee copy
is its own specialization variant with its own scope, always (the fold that
made `rollback` a constant was HONEST: the source omitted the argument, so
under the contract it was the default); the scope ladder is correct -- an
arm the book records as not writing takes the entered version; refuse
`ARM_VERSION_MISSING` only when the book records a version the builder
cannot find. Plan 90's four decisions remain recommendations (checkpoint 3).

Concept the user stated (to carry forward): the concordance is the ENTIRE
description of the compilation -- its input is the IR graphs (process graph,
control graph), its output is the emitted language. Today the graphs are
pinned at the node level (`ingestion_value`/`canonical_value` cells from
`source_span` roots) but graph EDGES and control BLOCKS are only partly on
the book, and NO edge exists from an SSA value to the text a backend emits.
Step 9 (plan `100_...`, lane running) makes the graphs views of the book and
adds an `emission` layer, so a C token diffuses back to its source span in
the globe viewer.

Lanes running at this checkpoint (each was told to write its own
continuation note; look for these files, committed or untracked):
- Step 4 planner (plan 80 part A) -> `docs/concordance_census/CONTINUATION_step4_planner.md`.
  Owns glsl_deployment_strategy, hierarchical_plan, loop_composer,
  transformation_priority, shell_reference_tables, process_graph_function_linking,
  reducer `fork_read_scope`; declarations in the "Step 4" section of
  concordance_declarations.py; carries the fix for the dropped scalar-write
  arm (`probe_scalar_write_only_arm.py` must pass).
- Step 5 control builder (plan 80 part B) -> `CONTINUATION_step5_control_builder.md`.
  Owns precompile_to_ssa, ssa_call_input_adapters, ir_identities, reducer
  `_set_operands` body; "Step 5" declarations section; `cell_set` page for
  many-source posts; MINTED ids gain mint records.
- Guarded tuple return fix -> `CONTINUATION_guard_tuple_return_fix.md`.
  Owns only `_normalize_top_level_guard_returns` in fortran_c_shell.py;
  `probe_record_in_tuple_return.py` must pass.
- Raw-only page inventory -> `75_raw_only_pages_inventory.md` (with a
  Continuation section).
- Step 9 plan -> `100_plan_step9_graph_input_and_emission_output.md` (with
  a Continuation section).

Gate for committing any lane's work: `probe_annotated_scalar_parameter`,
`probe_struct_intake`, `probe_branch_written_field` (+ the lane's probe)
pass; `tools/audit_identity_concordance.py` seven first lines identical to
baseline (view 0, toplevel 1, energy 0, controller 1, controller_untyped 5,
mapping 0, oscillator 0 findings); report the `unsourced:` counts and the
measurement script's DERIVED percentage before/after. Commit each lane
separately; never `git add -A`; do not push unless told.

Scratch: `Temp\wtb2` worktree at `ca79244b` (baseline for steps 2-3);
session scratchpad scripts (`measure_completeness.py`, `check_post_api.py`)
are described in this report if lost.

### Landed after checkpoint 4 (budget restored, lanes continued)

- Inventory committed (turing `93d46bda`): 148 raw-only page names; per
  owner step 4 = 28 pages, step 5 = 18, step 6 = 17, step 7 = 27, step 8 =
  12, unowned 21 (tensor_ssa_lowering, ssa_call_input_adapters,
  ir_identities, ssa_self_check, sequence-contract helpers, fcs shape seam),
  steps 1-3 remainder 25. Eleven pages embed a process id in their scope
  (`id(caller.G)`; the `name@control:<id>` scope minted in
  `lower_control_sections_to_ssa`) -- the step 5 lane was told to book-mint
  that scope and re-key them. DRAFT declare_page blocks in section 10.
- Guarded tuple return FIXED (turing `a9555bd5`): the guard rewrite splits
  tuple returns per lane like the tail-recursion rewriter; receipt gains
  result_names/tuple_result_arity; repro probe passes; native correctness
  14/14; audit first lines unchanged.
- Still running: step 4 (planner + the dropped-arm fix), step 5 (control
  builder), step 9 plan. Their continuation notes appear at the paths listed
  above when they finish.
- Step 9 plan committed (turing `901a758d`, `100_plan_step9_...md`): Part A
  graphs as views (operand edges = identity_transition Append rows through
  `_set_operands` as the ONE edge writer; ~24 hand-written edge writers to
  route; CONTROL_BLOCK/PLACEMENT/PROGRAM/SSA_BLOCK pages), Part B emission
  layer (EMISSION_UNIT/FUNCTION/ARTIFACT per backend). Top risk: backends run
  after `end_identity_book` and must use `identity_book(module)`, never
  `current_identity_book()` (mints a detached book silently). Needs step 5's
  identity_transition and step 6's SSA value identity first.
- RECONCILE before step 6 executes: plan 80's `ssa_value` page (step 5,
  fresh_value mints) and plan 90's `SSA_VALUE_IDENTITY` (step 6) are the SAME
  page; step 6 must adopt step 5's declaration, not add a second.
