# Turing Woodshop concordance / row-shape handoff

Date: 2026-09-29

Repository: `C:\dev\Powershell\turing`

Branch and remote at handoff:

- branch: `main`
- HEAD: `fa4b127a Separate record source discovery from reachability`
- `origin/main` is also `fa4b127a`
- remote: `https://github.com/sangderenard/Turing.git`
- the work described below is **not committed or pushed**

## WARNING: compiler numberings are ephemeral

Every graph node id, SSA value id, resident sequence id, planning alias,
region number and minted id quoted in this report (e.g. `168 -> 261`,
"sequence 264", "node 28", "regions 24, 26, 28") is valid **only for the run
that printed it**. Any change to the compiler, and sometimes nothing at all,
renumbers them. On 2026-09-29 the very next run after this report was
written renumbered the four `_advance_newton_dt_system` collections:
`momenta` moved from 262 to 263 and `centers` from 264 to 261.

Do not document compiler numberings beyond a single session, and do not act
on numbers from an earlier run. Identify a value by what it is: its source
expression, owning function, operation and declared attributes. Then find
its current number in the current run's identity log
(`turing/artifacts/identity_logs/woodshop_outer_physics.<timestamp>.*.log`),
for example the `LoopResult` whose `value_source_id` is the `center_xyz()`
call.

## Project direction and next frontier (added 2026-09-29, from the user)

- The compiler must write tensor operations **indifferent to the nature of
  the storage**, negotiating index math automatically. A double access such
  as `momenta[:, j]` over a dynamic sequence of 3-vectors is ordinary index
  math over one resident: the length cell owns the leading extent and the
  row layout owns `(3,)`.
- When an access looks impossible (for example "too many indices"), first
  find where the process got confused about what to expect, and fix it
  there. If the confusion turns out to be honest and inevitable given what
  the process knew at that time, discuss how to interpret the situation
  with that knowledge, always by way of the concordance.
- Once the Woodshop emits **clean C with no errors or shortfalls**, the next
  step is its **correctness rating**.
- Checkpoints are committed and pushed as work proceeds (`42a689a2`, a
  checkpoint with temporary diagnostics still present).

## Session log 2026-09-29 (continuation of this handoff)

### Found: the transformation book never ran (Python scoping defect)

After this report was written, `_tensor_descriptor`
(`turing/src/compiler/glsl_deployment_strategy.py`) was changed to read the
shape-transformation concordance first. The change imported
`concordant_shape_transformation_state` locally in its post-query block, and
Python makes any name that is imported inside a function local to the
**whole** function. The preamble used the name before that line bound it, so
every query raised `UnboundLocalError`. The preamble's `except` swallowed the
error and set `row = None`, which skipped the concordance read, all
transformation-edge recording and every sequence row-layout commit. One
diagnostic run counted 253,847 swallowed failures.

This was a Python implementation defect, not a flaw in the compiler's logic.
It did, however, hide the logic entirely: no dynamic collection received a
row layout, so `momenta[:, 0]` reached index normalization carrying its
element shape `(3,)` as the collection shape and raised "too many indices".

Fix: the preamble's own import now binds both names
(`concordant_shape_transformation_state`,
`descriptor_from_shape_transformation_state`) before first use. The rerun
with `TURING_DEBUG_SEQUENCE_ROW_LAYOUT=1` shows no swallowed descriptor
failures.

Lesson: that `except` hides everything. When descriptors look stale, run once
with `TURING_DEBUG_SEQUENCE_ROW_LAYOUT=1` before reasoning about logic.

### Also changed this session

- `identity_concordance.py`:
  - `record_shape_transformation` now also indexes each edge by its source
    identity on the page `shape_transformation_dependents`.
  - When a target's state changes, `withdraw_superseded_shape_derivations`
    withdraws every derivation recorded from the superseded state: shape
    state, proven extents and sequence row layout. Self-edges are skipped,
    and structured (non-int) source identities are kept as they are.
  - Row layouts gain an `invalidated` event, so a re-derived layout is a new
    generation rather than a disagreement.
- `glsl_deployment_strategy.py`: the call-site re-proof of a returned shape
  is now recorded as a callee-return transformation edge.

### Fixed: loop-continuation rewires were unrecorded morphs

`loop_composer.rewire_continuation` moved a consumer's operand from a
comprehension's materializer node to its collection `LoopResult` without
recording anything. Consumers described before loop composition therefore
kept their old answers. In `center_xyz`, `np.asarray([...])` stayed scalar,
so `.mean(axis=0)` returned a scalar. Rewires are now recorded on the page
`loop_continuation_rewire_concordance`, and every derivation below the port
is withdrawn. Verified: inside `center_xyz`, `asarray` is now rank 2,
dynamic, row `(3,)`, and `mean(axis=0)` is `(3,)`.

### Fixed: comprehensions had no shape law before loop composition (7 -> 1)

The item below turned out to be downstream of a confusion one step earlier.
Call-site return specialization reads a copy of the callee taken **before**
loop composition, and in that copy a list comprehension had no descriptor
law at all; only the composed collection `LoopResult` had one. The
comprehension node and its port now share one law (commit `5fbf693b`).
Woodshop: `OUTER_NATIVE_EMITTED complete=False shortfalls=1`. Only the
`argmin` shortfall in `_resolve_pair` remains.

Cost: the compile took about 650 s to finish lowering, against about 330 s
before. The large phases were ABI settlement round 2 (~199 s), the
sequence-schema survey (~144 s), and final legalization/reconciliation
(~163 s). This needs a phase-by-phase comparison against an earlier run,
whose full log was overwritten, before it is accepted.

### Milestone: complete C emission (commit `0b0594e6`)

The last shortfall (`argmin` in `_resolve_pair`) came from generator rows.
`part_bounds_xyz` yields `(part, lo, hi)`, and a generator had no
caller-visible product, so the destructured `lo_a`/`hi_a` were never
described in the graph phase. Fixed by publishing each yield column as a
`yield_row` transformation state on the caller's call value; destructured
loop targets now read column k. The late SSA writer
(`publish_projected_iterable_layouts`) now records its shape writes.

Result: `OUTER_NATIVE_EMITTED complete=True shortfalls=0`. The native build
then failed only because array initializers spelled infinity as Python
`inf`. That is fixed too: the emitted C compiles to an object with zero
errors.

**Native library built (09:56, same commit):**
`build/woodshop_outer_native/woodshop_outer_physics__woodshop_physics_step.dll`
(343 KB) links from the emitted C plus three `.ll` law pieces. Exported ABI:
`void woodshop_outer_physics__woodshop_physics_step(void **buffers, long long *extents)`,
fed through `CModuleArtifact.prepare_execution(feeds_by_ssa_value_id)`.

**Next frontier: the correctness rating** (the user's stated step after
clean C): compare the native step against the Python Woodshop per step,
per item, per lane, with the abs/scaled/ULP report shape of
`engine_toy/time_trials/compare_woodshop_newton.py` (whose inner-Newton
rating already reached 0 ULP). Before any rating means anything, resolve:

- **Red flag — CONFIRMED (slot map `build/woodshop_outer_native/public_slots.json`,
  module snapshot `module.pkl`, both written by the probe):** 21 public
  slots = `dt`, `items.length`, `items.keys` (17 tokens), `items.values`
  (a *scalar* int64 — wrong for a 17-entry keyed table), `restitution`,
  `friction`, eleven `contacts` cells, `last_metrics` handle + presence,
  `_newton_batch`, one `void **` pointer table. 392 of the root's 413
  formals carry no `program_abi_field` and are private `calloc`+`memset 0`
  arenas — among them the `float64 [3]`, `float64 [3,3]`, `int64 [17]`
  spans that are the items' positions, rotations, momenta and masses. The
  native step runs on an empty world. **First rating defect:** nested
  keyed-record rows (`items` → `WorldMachine` → `parts` table,
  `linear_momentum_kg_m_s`) never became public ABI inputs/outputs, and the
  record relocation prologue emitted nothing.
  **Traced (module snapshot + code):** one defect. In
  `materialize_parameter_record_abi`, rows of a `keyed` field with a
  `value_record` are materialized only for `indexed_value_candidates` of
  *this function's graph* and only for leaves with `record_field_candidates`
  (GetAttr reads) here. The binding root only calls `world.step(...)`, so it
  indexes nothing; the 17 `WorldMachine` rows and their nested
  `sim.machine.parts` columns get no root ABI identity; the frame linker
  (`_linked_caller_member`) finds no caller resident and leases callee
  workspace (`linked_call_frame_storage`, `program_abi_*` stripped by
  design), which the C wrapper correctly treats as private. All 392 private
  arenas carry that lease. The nested-table branch additionally requires the
  contract to spell `columns`; the probe declares `parts` by `row_record`
  only, though `_record_row_sequence_columns` can derive them. Fix point is
  the keyed branch of the binding function's materialization: rows from the
  declaration, not from local reads.
  **Fixed (turing commit "Own keyed-record rows at the binding function"):**
  the binding function owns the rows from the declaration as row-pooled
  `items[].<leaf>.column` spans; rows are a pairable RECORD field of the
  parent on every frame; the pooled column is a member of the callee's row
  record; fixed-shape span leaves are pooled too. Result: 37/37 callee
  column formals bind (was 0/37), six row columns (`custody`, `slot`,
  `pose_state`, `sim.machine.identity` `[17]`; `orientation_deg_xyz`,
  `linear_momentum_kg_m_s` `[17,3]`) are public root inputs, `items.values`
  is `[17]`, emission complete, library links. Still deferred and reported
  on the book: the nested `parts` table (needs the `lengths`/`row_stride`
  child-table layout at the declaring function) and the `edges[]`
  reference leaves. 391 private arenas remain; their classification
  (authored state vs compiler temporaries) is the next rating step.
- No marshaller exists for `keyed -> record rows` (`items`) or `table`
  (`parts`) fields; `_managed_native_feeds_by_id` handles flat fields only.
- `contacts` and `last_metrics` are `reference` storage in the probe ABI, so
  they have no native read-back unless the ABI is extended.
- Pickle the lowered `module` beside the DLL so slot -> field names are
  recoverable (idiom in `compare_woodshop_newton.py --module-snapshot`).

### Gauntlet track (examples/python_semantics_gauntlet.py, probe build/python_gauntlet_probe.py)

Twenty CPython-semantics torture cases with frozen expectations, compiled
through the same entry and contracts as Woodshop. Fixed so far (each a
compiler defect, per the rule that valid Python must compile):
- source pursuit follows calls through a loop over a static table of
  functions (`for case in CASES: case()`);
- ragged literal tuples no longer crash the literal descriptor rule;
- a module table of source functions is a static tuple of
  `FunctionReference`s (found by declared `module.qualname` identity);
- `functools.reduce` folds into a carried loop at binding install.

Remaining rejection: the `case()` call result is unavailable. Lowering it
as a dispatch table (runtime loop + switch over the 20 callees) is blocked
on a design item: the compiler has **no union/variant value**, and the 20
callees return 20 different types. User direction (2026-09-29): establish
the SSA for unions in a type table first, then the memory/alignment model
(aligned at small and large scale; nodus's tensor arena as reference).
Design notes are being gathered by workers; see the memory note
`project-union-type-design`.

### Superseded: the callee-return edge is a pseudo-identity

After the rewire fix, Woodshop still emits the same 7 shortfalls. The
`center_xyz()` call value in `_advance_newton_dt_system` took its shape
from the callee return during the call-site fixed point, which settled
**before** `center_xyz`'s own loop composition corrected its return. The
caller's edge names `("return", callee)` as its source rather than the
callee's actual returned value, so the corrected return has no edge to
travel to the call site. It reaches the call value only at SSA enrichment,
after the `centers` row layout was committed as `()`.

Direction: in `call_result_descriptor`
(`glsl_deployment_strategy.py`), record the edge from the callee's
returned value identity to the return identity, so that the caller's
call-value derivation is withdrawn and re-derived when the return changes.
Also check what re-derives a withdrawn call value once the call-site fixed
point has ended.

### Open, next to verify

`loop_composer.add_port` copies the element's `tensor` onto every new port,
including collection `LoopResult`s. That stamps the row descriptor onto the
collection as if it were the whole shape, and records no concordance edge.
If the rerun still shows a collection answered with its element's shape,
fix it at this origin: the collection port starts empty, and the collection
law derives its descriptor from the element through the concordance.

## User intent and non-negotiable constraints

The goal is to make the real Woodshop program compile through the complete
source -> graph -> control -> repository SSA -> C path, without changing the
Woodshop source to accommodate compiler defects.

The user has made these constraints explicit:

1. Work on the Woodshop only. Do not run pytest, including targeted pytest.
2. Read the implementation deterministically. Do not run unrelated or stale
   probes to discover behavior by accident.
3. There is one identity concordance. Do not add private/local semantic cache
   dictionaries beside it. Facts must travel on the transformation graph and
   through the concordance.
4. Do not solve a compiler problem by applying Python-source workarounds to
   Woodshop.
5. Preserve optimization. Dynamic shapes and runtime loop bounds should lower
   as real loops/index math when exact static extents do not exist; they must
   not force host materialization or duplicate physical objects.
6. Work connected failures together, not serially as isolated error messages.
7. A Woodshop validation process that is still running must be allowed at
   least **eight minutes** before considering interruption. If it exits on its
   own sooner, that is fine.
8. Before final handoff, remove temporary diagnostics, validate the Woodshop,
   then commit and push. This has not yet happened for the current worktree.

The top-level and local agent instructions apply. In particular, this is not
an invitation to redesign the compiler or add a stand-in next to its existing
systems. The sanctioned compiler entry is
`src.compiler.fortran_c_shell.lower_ast_source_to_ssa`.

## Only sanctioned end-to-end validator

From `C:\dev\Powershell\turing`, in PowerShell:

```powershell
$env:TURING_DEBUG_DUMP_FUNCTION='woodshop_outer_physics:build\woodshop_outer_native_failure_dump.txt'
python -u build\woodshop_outer_native_probe.py 2>&1 | Tee-Object -FilePath build\woodshop_outer_native_probe.log
```

Do not run pytest. The probe is ignored build material. It currently includes
large diagnostic dumps after lowering; those dumps account for much of the
runtime and log size.

For CMD rather than PowerShell, the earlier question about seeing and logging
output can be answered with:

```cmd
python -u build\woodshop_outer_native_probe.py 2>&1 | powershell -NoProfile -Command "$input | Tee-Object -FilePath build\woodshop_outer_native_probe.log"
```

The actual work in this handoff has been run from PowerShell with `Tee-Object`.

## Current result

The most recent Woodshop run exited naturally with process exit code 0 after
more than eight minutes. It completed whole-program lowering and C emission,
but the emitted artifact is incomplete:

```text
OUTER_NATIVE_EMITTED complete=False shortfalls=7
```

There is no compiler exception now. The seven C-emission shortfalls are two
connected metadata-flow failures:

1. One `argmin` input is still recorded as scalar/unknown shape:

   ```text
   argmin requires static source shapes in
   woodshop_outer_physics___resolve_pair__planned_region_2
   ```

2. Three `positions[:, i]`-style selectors over sequence value `%t264` survive
   normalization, and each causes a second missing-operand consequence. The
   affected regions are 24, 26, and 28, hence six shortfalls total.

Exact final lines are in:

- `C:\dev\Powershell\turing\build\woodshop_outer_native_probe.log`
- `C:\dev\Powershell\turing\build\woodshop_outer_native_failure_dump.txt`

The latest log's final shortfalls begin near line 8679.

## What is conclusively proven about the collection failure

Woodshop's source in `C:\dev\Powershell\engine_toy\woodshop.py` contains:

```python
centers = np.asarray([
    self.items[identity].center_xyz() for identity in identities
], dtype=np.float64)
momenta = np.asarray([
    self.items[identity].linear_momentum_kg_m_s
    for identity in identities
], dtype=np.float64)
```

Both elements are known 3-vectors, but by different legitimate proof paths:

- `center_xyz()` is a linked source call whose exact returned value is proven
  shape `(3,)`.
- `linear_momentum_kg_m_s` is a Program-ABI span field explicitly declared as
  rank 1, shape `[3]`.

The physical collections are dynamic in their first dimension and should be
represented as one sequence with row shape `(3,)`, not as two physical
objects. The correct logical shape is `(runtime_length, 3)`. The sequence
length cell owns the dynamic leading extent; the row descriptor owns `(3,)`.

The relevant exact planning aliases in the latest module are:

```text
168 -> 261
174 -> 262
186 -> 263
189 -> 264
```

The latest run has:

```text
sequence 262 row shape = (3,)
sequence 264 row shape = ()
```

and therefore only `%t264[:, 0]`, `%t264[:, 1]`, `%t264[:, 2]` fail.

Crucially, the previous run had the opposite useful fact: 264 had row `(3,)`
while 262 was empty. A change intended to publish declared span shape made 262
work while 264 stopped working. This is not evidence that one Woodshop field
needs a special case. It is evidence that the same semantic fact is being
published/read at inconsistent identities or stages. The defect is in the
transformation/concordance path and is sensitive to publication order.

The current graph descriptor trace shows source call node 28 (`center_xyz`)
with a stale local tensor descriptor of scalar shape, but the shared
`proven_shape` page correctly contains:

```text
('_advance_newton_dt_system', 28) -> ('proven', (3,), 'float64')
```

The linked call frame also carries shape `(3,)`. The missing step is carrying
that exact fact through the comprehension/materializer/LoopResult identity to
the resident sequence's row layout. The analogous Program-ABI field fact must
walk the same spine.

### Important architectural distinction

Do not publish `(3,)` as the **whole collection shape**. That was one source
of confusion. A collection `LoopResult` with dynamic length must answer:

```python
{
    "shape": (),
    "rank": 2,
    "metadata_state": "dynamic",
    "sequence_row_shape": (3,),
}
```

The leading dimension comes from the exact sequence length formal at lowering
time. The row shape is `(3,)`.

## Suspect edit that should be corrected, not blindly retained

In `src/compiler/identity_concordance.py`, inside
`publish_program_abi_graph_identities`, the current worktree added logic that
calls `record_proven_shape(...)` directly for a Program-ABI `span` GetAttr.

That logic was meant to place declared `[3]` on the exact element edge before
the collection `LoopResult` asks for an element descriptor. It is suspect
because graph value IDs are subsequently aliased to resident sequence IDs.
Recording a row fact on an identity that later means the collection can turn
the row shape into the collection's complete shape or make the result depend
on query order.

The replacement should follow the existing graph/concordance transformation
identity explicitly. Likely routes to inspect:

- `planning_value_concordance`
- the `LoopIterationOutput` fields `value_id`, `result_value_id`, and
  `materializer_node_id`
- collection `LoopResult` attributes `collection_iterable_value_id` and
  `materializer_node_id`
- the exact `value` parent of the collection `LoopResult`
- `commit_sequence_row_layout` / `committed_sequence_row_layout`

The correct change should let `_tensor_descriptor` ask the concordance for the
element identity's proven shape, then commit that as the resident sequence row
layout. It should not infer from a binding name, add a Woodshop-specific case,
or create a parallel dictionary.

Relevant implementation locations:

- `src/compiler/glsl_deployment_strategy.py`, `_tensor_descriptor` and
  `_tensor_descriptor_rule`, especially the collection-LoopResult branch near
  the current lines 18086-18307.
- `src/compiler/identity_concordance.py`,
  `publish_program_abi_graph_identities`, alias resolution helpers, and
  sequence-row layout APIs.
- `src/compiler/loop_composer.py`, collection materialization and
  `LoopResult` construction/rewiring.
- `src/compiler/fortran_c_shell.py`, publication of
  `planning_value_concordance`, sequence declarations, and control handoff.

Temporary trace currently only included selected node IDs and did not include
186/189, so the latest log does not fully print the second materializer path.
Code reading should be preferred. If another Woodshop run is genuinely needed,
expand the trace to include both complete paths and keep the eight-minute rule.

## What is conclusively proven about the `argmin` failure

The failing function is:

```text
woodshop_outer_physics___resolve_pair__planned_region_2
```

Its `argmin` operand is SSA value 21 and remains:

```text
dtype='float64', shape=()
```

Upstream, region 0 has vector inputs (including shapes `(3,)`) and calls
repository elementwise arithmetic (`binary_double`). Its results 19, 20, 21
were created with empty/scalar shapes. Shape preservation was extended in the
current worktree to understand repository calls:

- `binary_double`: broadcast first two operands
- `binary_scalar_double`
- `unary_double`
- existing `Cast` and `Phi`

This settles instruction-local result shapes and updates matching tensor
descriptors. However, the final module ordering currently ends with:

```python
propagate_repository_ssa_call_metadata(module)
settle_shape_preserving_value_metadata(module)
settle_repository_ssa_static_extent_operands(module)
```

If the last settlement makes a producer result shaped, there is no subsequent
call-edge propagation to carry that new fact across region return/projection
and into region 2's formal. This is the likely remaining `argmin` defect.

The architectural fix is a real monotone fixed point between:

- local/canonical shape settlement; and
- exact repository-SSA call-edge metadata propagation.

Do not add a private shape cache. These passes already operate on SSA values
and concordance facts. Iterate until neither pass changes anything, with care
to retain the compiler's disagreement checks. The shape-preserving pass only
fills previously empty extents, so it should be monotone. Confirm that the
call-edge pass does not oscillate before writing an unbounded loop; it already
has internal convergence/disagreement behavior.

Relevant implementation:

- `src/compiler/tensor_ssa_lowering.py`
  - `settle_shape_preserving_value_metadata` near line 884
  - `propagate_repository_ssa_call_metadata` near line 1034
  - early settlement sequence near lines 2098-2122
  - final settlement sequence near lines 5658-5668

## Changes accumulated in the current worktree

The current intended diff is large: approximately 3,084 insertions and 206
deletions across 12 files. It represents the connected compiler work since
`fa4b127a`, not just the last two edits.

Intended changed files:

- `src/common/tensors/accelerator_backends/c_backend_llvm_ssa.py`
- `src/compiler/control_source.py`
- `src/compiler/extraction_contract.py`
- `src/compiler/fortran_c_shell.py`
- `src/compiler/glsl_deployment_strategy.py`
- `src/compiler/identity_concordance.py`
- `src/compiler/loop_composer.py`
- `src/compiler/precompile_to_ssa.py`
- `src/compiler/ssa_c_backend.py`
- `src/compiler/tensor_ssa_lowering.py`
- `src/transmogrifier/graph/graph_express2.py`
- `tests/test_ast_parent_ingestion.py` (regression source only; do not run
  pytest)

Two files show modified status because of line-ending/worktree state but have
no textual diff in `git diff`; they were identified as user-owned/unrelated and
must not be staged or altered:

- `src/transmogrifier/graph/python_identity_programs.py`
- `tests/test_python_identity_programs.py`

Before committing, stage the intended paths explicitly. Do not use `git add .`.

### Main implemented capabilities in this uncommitted diff

The worktree includes much more than the final two failures:

- LLVM repository implementation and registration for authored C `mean_dim`.
- Dynamic/unknown loop-bound preservation so a real loop is retained rather
  than rejected or silently executed once.
- Typed uninitialized optional/None representation.
- Record and call-frame identity linking through the concordance, including
  optional presence members.
- Record field union/linking improvements without cheap stand-ins.
- Span/sequence indexing lowering as index math over one physical resident.
- Namespaced sequence helper symbols to prevent collisions.
- Generator-expression lowering as real nested control without changing
  Python source.
- Removal of accidental semantic dependency on host NumPy/NetworkX values in
  the paths encountered by Woodshop.
- Concordance-backed formal/actual linkage that cleared the previous five
  formal parity failures.
- Projected iterable row-layout publication.
- Late tensor settlement after projected iterable publication.
- Dynamic `basic_index` fallback that can use an exact resident `sequence_id`
  and its `sequence_length_for` formal.
- C backend diagnostics and lowering support accumulated while following the
  Woodshop through the pipeline.

These should be reviewed as one connected compiler change before commit; do
not discard them just because the newest blocker is narrower.

## Earlier blocker progression (useful evidence that the pipeline advanced)

The session began with a hard deployment exception:

```text
CompilationSubdivisionRequired: ... blockers=('unresolved-loop-bound',)
```

After loop work, the next blocker was missing scalar ABI dtype for optional
`last_metrics`.

After typed optional work, record linkage failed because the caller record did
not expose `items`, then `last_metrics.__present`.

After concordance-based record work, `_newton_dt_pieces` disagreed as one
opaque reference in the callee versus a three-member sequence in the caller.
That led to the one-resident span/sequence indexing strategy rather than
publishing multiple physical objects.

After generator and linker work, the whole program now lowers all 79/79 local
functions, materializes record/keyed-sequence ABI, builds source-call records,
links repository SSA, and reaches C emission. The present seven shortfalls are
therefore materially farther along, not a recurrence of the first exception.

Recent already-committed checkpoints on `main` are:

```text
fa4b127a Separate record source discovery from reachability
febb36c6 Lower consumed generator expressions into nested control
a49c0a0c Concord source records and sequence row layouts
f1243cac Lower dynamic sequence tensor slices through SSA
1d55ae45 Complete Woodshop outer concordance lowering
b8cd8c2a Complete native compiler repairs and Woodshop diagnostics
```

## Temporary diagnostics that must be removed before final commit

1. `src/compiler/glsl_deployment_strategy.py`
   - page/block named `temporary_woodshop_descriptor_trace`
   - currently around line 17920

2. `src/compiler/fortran_c_shell.py`
   - environment-gated block `TURING_DEBUG_SEQUENCE_RECORD_ABI`
   - currently around line 5167

3. `build/woodshop_outer_native_probe.py`
   - extra identity/module diagnostic dumping was added locally
   - the file is ignored and must not be staged
   - it can remain as local diagnostics or be cleaned once no longer needed

Search before commit:

```powershell
rg -n "temporary_woodshop_descriptor_trace|TURING_DEBUG_SEQUENCE_RECORD_ABI" src
```

Other general-purpose diagnostics should be judged by whether they are real,
useful compiler diagnostics rather than Woodshop-only tracing. The user asked
to fix diagnostics after the functional fix.

## Validation already performed

- `python -m py_compile` was clean for the edited compiler files at the last
  validation point.
- `git diff --check` is clean except expected LF/CRLF warnings.
- No pytest was run.
- The latest complete Woodshop command exited naturally after more than eight
  minutes and reached C emission with seven shortfalls.

After further edits, repeat syntax compilation and `git diff --check`, then run
only the Woodshop command above.

## Recommended next sequence

1. Read the exact collection materialization path in `loop_composer.py` and
   identify how element `value_id` becomes collection `result_value_id` and
   then resident sequence 262/264 in `planning_value_concordance`.
2. Remove or narrow the suspect direct Program-ABI GetAttr
   `record_proven_shape` publication. Route both the linked-call return shape
   and declared span shape through the exact element identity to
   `commit_sequence_row_layout` on the resident collection.
3. Make final tensor metadata propagation a monotone fixed point so region 0's
   newly shaped elementwise result reaches `_resolve_pair` region 2 before
   `argmin` legalization.
4. Run `py_compile` and `git diff --check`.
5. Run the Woodshop validator and give it eight minutes minimum if still live.
6. Treat all new Woodshop failures as one graph-state report and address the
   connected set, not just the first line.
7. Once complete, remove temporary diagnostics and rerun Woodshop.
8. Inspect the staged diff carefully, explicitly excluding the two unrelated
   line-ending-only files and ignored build probe.
9. Commit with an accurate message and push `main` to `origin` as the user
   previously requested.

## Compact mental model

There is one semantic transformation graph and one concordance. A source value
may pass through authored call result, comprehension element, LoopResult,
planning alias, resident sequence, SSA formal, and backend pointer, but these
are successive identities/representations of one fact-bearing path. Shape is
not rediscovered by looking at Python objects, names, NumPy, or a side cache.
The concordance proves which identity a stage means; the sequence descriptor
separates runtime length from fixed row layout; repository SSA then lowers
indexing to ordinary pointer/index arithmetic. The two remaining failure
groups are both places where that fact propagation currently stops one edge
too early.
