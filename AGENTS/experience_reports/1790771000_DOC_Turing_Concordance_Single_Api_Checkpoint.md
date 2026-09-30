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
